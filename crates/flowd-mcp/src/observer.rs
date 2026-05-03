//! [`PlanEventStore`]-backed implementation of [`PlanObserver`].
//!
//! The daemon installs a [`PlanEventObserver`] on the executor; every
//! lifecycle transition (plan submitted, started, per-step
//! completed/failed/refused/cancelled, plan finished) is durably
//! appended to the dedicated `plan_events` table introduced in HL-39.
//!
//! ## Why a dedicated event log instead of `MemoryService`?
//!
//! Plan events are *operational telemetry*, not semantic recall:
//!
//! - They benefit from a stable, queryable read path
//!   (`flowd plan events <id>`) rather than living mixed in with hybrid
//!   search results.
//! - Persisting them avoids paying ONNX + Qdrant cost for what is
//!   effectively structured logging.
//! - They warrant a longer, simpler retention than tier-aged memory.
//!
//! ## Bounded-queue dispatch (HL-40)
//!
//! [`PlanObserver::on_event`] is synchronous so the executor's hot path
//! stays `async_trait`-free and dyn-friendly. Storage I/O is real work,
//! so we cannot perform it inline. The observer therefore owns:
//!
//! 1. A bounded `tokio::sync::mpsc` channel (`capacity` configurable via
//!    `flowd start --plan-event-buffer N`, default 1024).
//! 2. A single background drain task that `await`s `rx.recv()` and
//!    calls `store.record(event).await`. Storage failures are logged and
//!    dropped; the executor never sees them.
//! 3. An `AtomicU64` *dropped-events* counter, incremented whenever
//!    `try_send` returns `Full`. This replaces the previous failure
//!    mode -- silent unbounded `tokio::spawn` per event -- with a
//!    visible, rate-limited warning + a counter operators can read via
//!    `flowd status`.
//!
//! ### Why drop-with-counter instead of block-with-timeout?
//!
//! Blocking the executor is the failure mode we explicitly need to
//! avoid (a slow `SQLite` write must never cascade into stalled plan
//! execution). Operators get the same visibility from the dropped-events
//! counter and the warn-rate, without trading executor responsiveness
//! for log durability. A future `--plan-event-mode strict` flag can
//! swap `try_send` -> `send_timeout` for compliance use cases. Out of
//! scope here.
//!
//! ### Graceful shutdown
//!
//! [`PlanEventObserver::shutdown`] notifies the drain task to flush
//! whatever is still buffered, then awaits its `JoinHandle` with a
//! caller-supplied deadline (the daemon uses 5s). If the deadline
//! expires we log a warning and return -- never block the daemon's
//! teardown indefinitely.
//!
//! ### Health snapshot
//!
//! When configured with a `health_file` path, the observer spawns a
//! second background task that ticks every `health_interval` and writes
//! an [`ObserverHealth`] JSON snapshot atomically (write-to-temp +
//! rename). `flowd status` reads that file out-of-band, so a separate
//! CLI process can render `plan_events: capacity=N in_flight=M
//! dropped=K` without IPC.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use flowd_core::error::FlowdError;
use flowd_core::orchestration::observer::{PlanEvent, PlanObserver};
use flowd_core::orchestration::plan_events::PlanEventStore;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{self, error::TrySendError};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use uuid::Uuid;

/// Default channel capacity. Picked so a hot plan emitting one event per
/// step can buffer ~hundreds of plan-runs of slack before storage
/// pressure starts dropping.
pub const DEFAULT_CAPACITY: usize = 1024;

/// Default cadence at which the health snapshot file is rewritten.
pub const DEFAULT_HEALTH_INTERVAL: Duration = Duration::from_secs(1);

/// Minimum gap between successive `Full`-channel warnings on the hot
/// path. Prevents log floods when storage stays slow.
const FULL_WARN_INTERVAL: Duration = Duration::from_secs(1);

/// Construction-time configuration for [`PlanEventObserver`].
#[derive(Debug, Clone)]
pub struct PlanEventObserverConfig {
    /// Bounded mpsc capacity. Must be >= 1.
    pub capacity: usize,
    /// Optional path to write the [`ObserverHealth`] JSON snapshot to.
    /// `None` disables the health task entirely (tests usually omit).
    pub health_file: Option<PathBuf>,
    /// How often the health task re-writes the snapshot. Ignored when
    /// `health_file` is `None`.
    pub health_interval: Duration,
}

impl Default for PlanEventObserverConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_CAPACITY,
            health_file: None,
            health_interval: DEFAULT_HEALTH_INTERVAL,
        }
    }
}

/// Operator-visible health snapshot of the observer pipeline.
///
/// Persisted as JSON so `flowd status` (a separate process) can render
/// it without any IPC. `updated_at` is unix seconds so consumers can
/// flag a stale snapshot when the daemon is no longer running.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObserverHealth {
    pub capacity: usize,
    /// Number of events currently buffered in the mpsc channel
    /// (`capacity - tx.capacity()`).
    pub in_flight: usize,
    /// Cumulative count of events the observer dropped because the
    /// channel was full at `try_send` time. Resets only on daemon
    /// restart.
    pub dropped: u64,
    /// Unix timestamp (seconds) when this snapshot was written.
    pub updated_at: u64,
}

/// Outcome of a [`PlanEventObserver::shutdown`] call.
#[derive(Debug, Clone, Copy)]
pub struct ShutdownReport {
    /// Total events dropped over the lifetime of the observer.
    pub dropped: u64,
    /// `true` when the drain task did not finish before the deadline
    /// expired. Some buffered events may still be unwritten.
    pub timed_out: bool,
}

/// Bounded-queue, fire-and-forget plan-event observer.
///
/// See module docs for the contract; specifically the trade-off between
/// drop-with-counter and block-with-timeout.
pub struct PlanEventObserver {
    /// Hot-path sender. `try_send` is lock-free and never blocks.
    tx: mpsc::Sender<PlanEvent>,
    capacity: usize,
    dropped: Arc<AtomicU64>,
    /// Rate-limit gate for `Full` warnings (only contended on drops).
    last_full_warn_at: Mutex<Option<Instant>>,
    /// Held under a mutex so [`Self::shutdown`] (which only takes
    /// `&self` because the observer lives behind `Arc`) can move the
    /// handles out exactly once. Always `Some` until shutdown runs.
    shutdown_state: Mutex<Option<ShutdownState>>,
}

/// Captures everything the observer needs to perform a graceful shutdown
/// from a `&self` method. `Option`s in the parent let us `take()` once.
struct ShutdownState {
    shutdown_signal: watch::Sender<bool>,
    drain_handle: JoinHandle<()>,
    health_handle: Option<JoinHandle<()>>,
}

impl PlanEventObserver {
    /// Build the observer, spawning the drain task and (optionally) the
    /// health writer. The store is consumed by the drain task only;
    /// callers retain `Arc`'d access elsewhere if needed.
    ///
    /// # Panics
    /// Panics if `config.capacity == 0`. A zero-capacity channel would
    /// drop *every* event; no caller wants that, so we surface it as a
    /// programmer error rather than silently degrade.
    #[must_use]
    pub fn new<S>(store: Arc<S>, config: PlanEventObserverConfig) -> Self
    where
        S: PlanEventStore + 'static,
    {
        let PlanEventObserverConfig {
            capacity,
            health_file,
            health_interval,
        } = config;
        assert!(capacity >= 1, "PlanEventObserver capacity must be >= 1");

        let (tx, rx) = mpsc::channel::<PlanEvent>(capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        // `watch::channel` retains the signal value, so a shutdown
        // raised before the drain task starts polling is still
        // observed -- avoiding the lost-wake race that `Notify`
        // suffers from.
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        // The drain task is the sole owner of the store from here on.
        let drain_handle = spawn_drain_task(rx, store, shutdown_rx.clone());

        let health_handle = health_file.map(|path| {
            spawn_health_task(
                path,
                health_interval,
                capacity,
                tx.clone(),
                Arc::clone(&dropped),
                shutdown_rx,
            )
        });

        Self {
            tx,
            capacity,
            dropped,
            last_full_warn_at: Mutex::new(None),
            shutdown_state: Mutex::new(Some(ShutdownState {
                shutdown_signal: shutdown_tx,
                drain_handle,
                health_handle,
            })),
        }
    }

    /// Cumulative dropped-events counter since construction.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Configured channel capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Live snapshot of pipeline health.
    #[must_use]
    pub fn health(&self) -> ObserverHealth {
        ObserverHealth {
            capacity: self.capacity,
            in_flight: in_flight(&self.tx, self.capacity),
            dropped: self.dropped(),
            updated_at: unix_seconds_now(),
        }
    }

    /// Signal the drain task to flush remaining queued events and exit,
    /// then wait up to `deadline` for it to do so. Idempotent: a second
    /// call is a no-op.
    ///
    /// On timeout we log a warning but do not block the caller; some
    /// buffered events may remain unwritten in that case.
    pub async fn shutdown(&self, deadline: Duration) -> ShutdownReport {
        // Recover from a poisoned lock rather than panicking: shutdown
        // is on the daemon's teardown path, panicking here would leak
        // the PID file. Held in a tight block so the guard is dropped
        // before any `.await`.
        let state = {
            let mut guard = match self.shutdown_state.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            guard.take()
        };
        let Some(state) = state else {
            return ShutdownReport {
                dropped: self.dropped(),
                timed_out: false,
            };
        };

        // Tell the drain & health tasks to exit. The drain task drains
        // whatever is currently queued before exiting; the health task
        // exits immediately. `send`'s only failure mode is "no receivers
        // remaining", which means the tasks already exited -- nothing to
        // do.
        let _ = state.shutdown_signal.send(true);
        if let Some(h) = state.health_handle {
            h.abort();
        }

        let timed_out = if let Ok(join_result) =
            tokio::time::timeout(deadline, state.drain_handle).await
        {
            if let Err(e) = join_result {
                tracing::warn!(error = %e, "plan event drain task join error");
            }
            false
        } else {
            tracing::warn!(
                deadline_secs = deadline.as_secs(),
                "plan event drain task did not finish before deadline; some buffered events may be lost"
            );
            true
        };

        ShutdownReport {
            dropped: self.dropped(),
            timed_out,
        }
    }
}

impl PlanObserver for PlanEventObserver {
    fn on_event(&self, event: PlanEvent) {
        match self.tx.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(dropped_event)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                self.maybe_warn_full(dropped_event.plan_id());
            }
            Err(TrySendError::Closed(dropped_event)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    plan_id = %dropped_event.plan_id(),
                    "plan event channel closed; observer is shutting down (event dropped)",
                );
            }
        }
    }
}

impl PlanEventObserver {
    /// Emit a `Full`-channel warning at most once per
    /// [`FULL_WARN_INTERVAL`]. Reduces log spam when storage stays slow
    /// (we still get the precise count via the `dropped` field).
    fn maybe_warn_full(&self, plan_id: Uuid) {
        let mut guard = match self.last_full_warn_at.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let now = Instant::now();
        let should_warn = match *guard {
            None => true,
            Some(prev) => now.duration_since(prev) >= FULL_WARN_INTERVAL,
        };
        if should_warn {
            *guard = Some(now);
            let dropped = self.dropped.load(Ordering::Relaxed);
            tracing::warn!(
                plan_id = %plan_id,
                capacity = self.capacity,
                dropped_total = dropped,
                "plan event channel full; dropping event (suppressed for {}s window)",
                FULL_WARN_INTERVAL.as_secs(),
            );
        }
    }
}

impl Drop for PlanEventObserver {
    fn drop(&mut self) {
        // Best-effort flush. Synchronous Drop cannot await, so we just
        // poke the signal and abort the tasks. Callers wanting a
        // bounded flush should call `shutdown` explicitly *before*
        // dropping the observer (the daemon does).
        if let Ok(mut guard) = self.shutdown_state.lock() {
            if let Some(state) = guard.take() {
                let _ = state.shutdown_signal.send(true);
                if let Some(h) = state.health_handle {
                    h.abort();
                }
                state.drain_handle.abort();
            }
        }
    }
}

/// Compute `in_flight` from the live sender. `tokio::mpsc::Sender::capacity`
/// returns *remaining* capacity, so the in-flight count is the
/// configured capacity minus what's still free.
fn in_flight<T>(tx: &mpsc::Sender<T>, capacity: usize) -> usize {
    capacity.saturating_sub(tx.capacity())
}

fn unix_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn spawn_drain_task<S>(
    mut rx: mpsc::Receiver<PlanEvent>,
    store: Arc<S>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()>
where
    S: PlanEventStore + 'static,
{
    tokio::spawn(async move {
        // Pre-check: shutdown might already be raised before we even
        // poll (race with very early `shutdown()` calls). Bail straight
        // to the drain branch in that case.
        if *shutdown.borrow() {
            while let Ok(event) = rx.try_recv() {
                record(store.as_ref(), &event).await;
            }
            return;
        }
        loop {
            tokio::select! {
                biased;
                _ = shutdown.changed() => {
                    // `changed` only resolves when the value changes;
                    // the only change we care about is `false -> true`.
                    while let Ok(event) = rx.try_recv() {
                        record(store.as_ref(), &event).await;
                    }
                    break;
                }
                maybe_event = rx.recv() => {
                    match maybe_event {
                        Some(event) => record(store.as_ref(), &event).await,
                        None => break, // all senders dropped
                    }
                }
            }
        }
    })
}

async fn record<S>(store: &S, event: &PlanEvent)
where
    S: PlanEventStore,
{
    if let Err(e) = store.record(event).await {
        log_record_failure(&e, event.plan_id());
    }
}

fn log_record_failure(e: &FlowdError, plan_id: Uuid) {
    tracing::warn!(
        plan_id = %plan_id,
        error = %e,
        "plan event observer dropped event due to storage failure",
    );
}

fn spawn_health_task(
    path: PathBuf,
    interval: Duration,
    capacity: usize,
    tx: mpsc::Sender<PlanEvent>,
    dropped: Arc<AtomicU64>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        // Write an initial snapshot so `flowd status` has something to
        // read immediately after `flowd start`.
        write_health_snapshot(&path, capacity, in_flight(&tx, capacity), &dropped).await;

        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker.tick().await;

        loop {
            tokio::select! {
                biased;
                _ = shutdown.changed() => break,
                _ = ticker.tick() => {
                    write_health_snapshot(&path, capacity, in_flight(&tx, capacity), &dropped)
                        .await;
                }
            }
        }
    })
}

async fn write_health_snapshot(
    path: &std::path::Path,
    capacity: usize,
    in_flight: usize,
    dropped: &AtomicU64,
) {
    let snapshot = ObserverHealth {
        capacity,
        in_flight,
        dropped: dropped.load(Ordering::Relaxed),
        updated_at: unix_seconds_now(),
    };
    if let Err(e) = persist_atomic(path, &snapshot).await {
        tracing::debug!(
            path = %path.display(),
            error = %e,
            "failed to write plan-event observer health snapshot",
        );
    }
}

/// Atomic-rename write so a concurrent reader (`flowd status`) never
/// observes a half-written file.
async fn persist_atomic(path: &std::path::Path, snapshot: &ObserverHealth) -> std::io::Result<()> {
    let json = serde_json::to_vec(snapshot).map_err(std::io::Error::other)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await?;
        }
    }
    let tmp = path.with_extension("health.tmp");
    tokio::fs::write(&tmp, &json).await?;
    tokio::fs::rename(&tmp, path).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use flowd_core::error::Result;
    use flowd_core::orchestration::PlanStatus;
    use flowd_core::orchestration::plan_events::{PlanEventQuery, StoredPlanEvent, kind};
    use std::sync::Mutex as StdMutex;

    /// Test double that records every event after sleeping `delay`. The
    /// sleep simulates slow storage so `try_send` backpressure becomes
    /// visible.
    struct SlowStore {
        events: StdMutex<Vec<PlanEvent>>,
        delay: Duration,
    }

    impl SlowStore {
        fn new(delay: Duration) -> Self {
            Self {
                events: StdMutex::new(Vec::new()),
                delay,
            }
        }

        fn recorded_count(&self) -> usize {
            self.events.lock().unwrap().len()
        }
    }

    impl PlanEventStore for SlowStore {
        fn record(
            &self,
            event: &PlanEvent,
        ) -> impl std::future::Future<Output = Result<()>> + Send {
            let cloned = event.clone();
            let delay = self.delay;
            let events = &self.events;
            async move {
                tokio::time::sleep(delay).await;
                events.lock().unwrap().push(cloned);
                Ok(())
            }
        }

        async fn list_for_plan(
            &self,
            _plan_id: Uuid,
            _query: PlanEventQuery,
        ) -> Result<Vec<StoredPlanEvent>> {
            Ok(Vec::new())
        }
    }

    fn ev(plan_id: Uuid) -> PlanEvent {
        PlanEvent::Started {
            plan_id,
            project: "demo".into(),
        }
    }

    #[tokio::test]
    async fn fans_every_variant_into_the_store() {
        let store = Arc::new(SlowStore::new(Duration::ZERO));
        let observer = PlanEventObserver::new(
            Arc::clone(&store),
            PlanEventObserverConfig {
                capacity: 64,
                ..PlanEventObserverConfig::default()
            },
        );

        let plan_id = Uuid::new_v4();
        let project = "demo".to_owned();
        let events = vec![
            PlanEvent::Submitted {
                plan_id,
                name: "p".into(),
                project: project.clone(),
            },
            PlanEvent::Started {
                plan_id,
                project: project.clone(),
            },
            PlanEvent::StepStarted {
                plan_id,
                project: project.clone(),
                step_id: "a".into(),
                agent_type: "echo".into(),
                started_at: Utc::now(),
            },
            PlanEvent::StepCompleted {
                plan_id,
                project: project.clone(),
                step_id: "a".into(),
                agent_type: "echo".into(),
                output: "ok".into(),
                metrics: None,
            },
            PlanEvent::StepFailed {
                plan_id,
                project: project.clone(),
                step_id: "b".into(),
                agent_type: "echo".into(),
                error: "boom".into(),
                metrics: None,
            },
            PlanEvent::StepRefused {
                plan_id,
                project: project.clone(),
                step_id: "c".into(),
                agent_type: "echo".into(),
                reason: "deny".into(),
            },
            PlanEvent::StepCancelled {
                plan_id,
                project: project.clone(),
                step_id: "d".into(),
                agent_type: "echo".into(),
            },
            PlanEvent::Finished {
                plan_id,
                project,
                status: PlanStatus::Completed,
                total_metrics: None,
                step_count: flowd_core::orchestration::observer::PlanStepCounts::default(),
                elapsed_ms: None,
            },
        ];
        let total = events.len();
        for e in events {
            observer.on_event(e);
        }

        let report = observer.shutdown(Duration::from_secs(2)).await;
        assert!(!report.timed_out);
        assert_eq!(store.recorded_count(), total);
        assert_eq!(report.dropped, 0);

        let recorded = store.events.lock().unwrap().clone();
        let kinds: Vec<&'static str> = recorded
            .iter()
            .map(flowd_core::orchestration::plan_events::event_kind)
            .collect();
        assert_eq!(
            kinds,
            vec![
                kind::SUBMITTED,
                kind::STARTED,
                kind::STEP_STARTED,
                kind::STEP_COMPLETED,
                kind::STEP_FAILED,
                kind::STEP_REFUSED,
                kind::STEP_CANCELLED,
                kind::FINISHED,
            ]
        );
    }

    #[tokio::test]
    async fn bounded_queue_drops_under_pressure_and_drains_remainder() {
        // Tiny capacity + slow store guarantees backpressure within the
        // 100-event burst. Storage takes 20ms per record so 100 events
        // emitted near-instantly will overflow the 8-slot buffer.
        let store = Arc::new(SlowStore::new(Duration::from_millis(20)));
        let observer = PlanEventObserver::new(
            Arc::clone(&store),
            PlanEventObserverConfig {
                capacity: 8,
                ..PlanEventObserverConfig::default()
            },
        );

        let total = 100u64;
        let plan_id = Uuid::new_v4();

        // Measure the worst-case `on_event` latency. Non-blocking means
        // each call returns in microseconds; we use a very loose 50ms
        // upper bound so CI noise doesn't make this flaky.
        let mut max_latency = Duration::ZERO;
        for _ in 0..total {
            let start = Instant::now();
            observer.on_event(ev(plan_id));
            let elapsed = start.elapsed();
            if elapsed > max_latency {
                max_latency = elapsed;
            }
        }
        assert!(
            max_latency < Duration::from_millis(50),
            "on_event should never block; max latency was {max_latency:?}",
        );

        // Health snapshot: dropped should already be > 0.
        let pre_shutdown = observer.health();
        assert!(
            pre_shutdown.dropped > 0,
            "expected backpressure to drop events"
        );
        assert_eq!(pre_shutdown.capacity, 8);

        // Drain whatever remains. With 8-slot buffer + 20ms per record,
        // we need at least ~200ms; give 2s for safety.
        let report = observer.shutdown(Duration::from_secs(2)).await;
        assert!(!report.timed_out);

        // Conservation law: every emitted event was either recorded or
        // counted as a drop. Nothing silently disappears.
        let recorded = store.recorded_count() as u64;
        assert_eq!(report.dropped + recorded, total);
        assert!(report.dropped > 0);
    }

    #[tokio::test]
    async fn shutdown_is_idempotent() {
        let store = Arc::new(SlowStore::new(Duration::ZERO));
        let observer =
            PlanEventObserver::new(Arc::clone(&store), PlanEventObserverConfig::default());
        let first = observer.shutdown(Duration::from_secs(1)).await;
        let second = observer.shutdown(Duration::from_secs(1)).await;
        assert_eq!(first.dropped, second.dropped);
        assert!(!first.timed_out);
        assert!(!second.timed_out);
    }

    #[tokio::test]
    async fn health_snapshot_file_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("observer.health");
        let store = Arc::new(SlowStore::new(Duration::ZERO));
        let observer = PlanEventObserver::new(
            Arc::clone(&store),
            PlanEventObserverConfig {
                capacity: 16,
                health_file: Some(path.clone()),
                health_interval: Duration::from_millis(50),
            },
        );
        // Wait for the initial write + at least one tick.
        for _ in 0..40 {
            if path.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(path.exists(), "health file never written");
        let raw = tokio::fs::read(&path).await.unwrap();
        let snapshot: ObserverHealth = serde_json::from_slice(&raw).unwrap();
        assert_eq!(snapshot.capacity, 16);
        assert_eq!(snapshot.dropped, 0);

        let _ = observer.shutdown(Duration::from_secs(1)).await;
    }

    #[tokio::test]
    async fn closed_channel_increments_dropped_counter() {
        let store = Arc::new(SlowStore::new(Duration::ZERO));
        let observer =
            PlanEventObserver::new(Arc::clone(&store), PlanEventObserverConfig::default());
        // Shut the drain task down. `shutdown` awaits the JoinHandle, so
        // by the time it returns the receiver half has been dropped and
        // the channel is closed for sends.
        let _ = observer.shutdown(Duration::from_secs(1)).await;

        let dropped_before = observer.dropped();
        observer.on_event(ev(Uuid::new_v4()));
        let dropped_after = observer.dropped();
        assert_eq!(
            dropped_after,
            dropped_before + 1,
            "post-shutdown on_event should bump the dropped counter via the Closed branch",
        );
    }
}
