//! End-to-end orchestrator tests.
//!
//! Exercises plan loading from disk, the full submit → confirm → execute
//! state machine through the public trait, and cancellation of an in-flight
//! plan via aborted `JoinHandle`s.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::{
    AgentOutput, AgentSpawner, InMemoryPlanExecutor, PlanExecutor, PlanStatus, PlanStep,
    StepStatus, load_plan,
};

fn temp_dir(prefix: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

struct EchoSpawner(Arc<AtomicUsize>);

impl AgentSpawner for EchoSpawner {
    async fn spawn(&self, step: &PlanStep) -> Result<AgentOutput> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(AgentOutput::success(format!("ran:{}", step.id)))
    }
}

/// Spawner that sleeps long enough for the test to issue a cancel before the
/// step naturally finishes.
struct SleepySpawner;

impl AgentSpawner for SleepySpawner {
    async fn spawn(&self, _: &PlanStep) -> Result<AgentOutput> {
        tokio::time::sleep(Duration::from_secs(30)).await;
        Ok(AgentOutput::success("late"))
    }
}

#[test]
fn load_plan_from_yaml_file_and_execute() {
    let dir = temp_dir("flowd-plans-yaml");
    let plan_path = dir.join("plan.yaml");
    fs::write(
        &plan_path,
        r#"
name: nightly
steps:
  - id: gather
    agent: summarizer
    prompt: "collect"
  - id: review
    agent: reviewer
    prompt: "review"
    depends_on: [gather]
"#,
    )
    .unwrap();

    let plan = load_plan(&plan_path).unwrap();
    assert_eq!(plan.name, "nightly");
    assert_eq!(plan.steps.len(), 2);

    let invocations = Arc::new(AtomicUsize::new(0));
    let exec = InMemoryPlanExecutor::new(EchoSpawner(Arc::clone(&invocations)));

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let id = exec.submit(plan).await.unwrap();
        let preview = exec.confirm(id).await.unwrap();
        assert_eq!(
            preview.execution_order,
            vec![vec!["gather".to_owned()], vec!["review".to_owned()]]
        );
        exec.execute(id).await.unwrap();
        let final_plan = exec.status(id).await.unwrap();
        assert_eq!(final_plan.status, PlanStatus::Completed);
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
    });

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_plan_from_json_file() {
    let dir = temp_dir("flowd-plans-json");
    let plan_path = dir.join("plan.json");
    fs::write(
        &plan_path,
        r#"{"name": "j", "steps": [{"id": "a", "agent_type": "echo", "prompt": "x"}]}"#,
    )
    .unwrap();
    let plan = load_plan(&plan_path).unwrap();
    assert_eq!(plan.steps.len(), 1);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn cancel_aborts_in_flight_plan() {
    let exec = Arc::new(InMemoryPlanExecutor::new(SleepySpawner));
    let plan = flowd_core::orchestration::Plan::new(
        "p",
        vec![PlanStep {
            id: "long".into(),
            agent_type: "echo".into(),
            prompt: "x".into(),
            depends_on: vec![],
            timeout_secs: None,
            retry_count: 0,
            status: StepStatus::Pending,
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        }],
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let id = exec.submit(plan).await.unwrap();
        exec.confirm(id).await.unwrap();

        let exec_for_task = Arc::clone(&exec);
        let exec_handle = tokio::spawn(async move { exec_for_task.execute(id).await });

        // Give the executor a moment to spawn the long-running step.
        tokio::time::sleep(Duration::from_millis(50)).await;
        exec.cancel(id).await.unwrap();

        // Bound the wait so a hung executor surfaces as a test failure.
        tokio::time::timeout(Duration::from_secs(5), exec_handle)
            .await
            .expect("executor should finish quickly after cancel")
            .unwrap()
            .unwrap();

        let final_plan = exec.status(id).await.unwrap();
        assert_eq!(final_plan.status, PlanStatus::Cancelled);
        assert_eq!(final_plan.steps[0].status, StepStatus::Skipped);
    });
}

#[test]
fn malformed_yaml_returns_validation_error() {
    let dir = temp_dir("flowd-plans-bad");
    let plan_path = dir.join("plan.yaml");
    fs::write(&plan_path, "::: not yaml :::").unwrap();
    let err = load_plan(&plan_path).unwrap_err();
    assert!(matches!(err, FlowdError::PlanValidation(_)));
    fs::remove_dir_all(&dir).ok();
}
