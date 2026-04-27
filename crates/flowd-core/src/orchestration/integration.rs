//! Pure v1 contract for `plan_integrate`.
//!
//! Locks the policy decisions made during the
//! audit-current-worktree-merge-flow / research-linear-history-options /
//! research-commit-policy-and-release-ux / synthesize-long-term-design
//! research rounds, **before** any git command lands. Concrete git
//! execution will arrive in a follow-up; this module deliberately
//! stays I/O-free so the rules can be unit-tested cheaply and
//! refactored independently of a working repository.
//!
//! # Behaviour summary (v1)
//!
//! `plan_integrate` proposes -- and on operator confirmation, applies
//! -- a single, linear-history merge of a [`super::PlanStatus::Completed`]
//! plan into a configured base branch.
//!
//! * **Mode**: manual confirm by default ([`IntegrationMode::Confirm`]).
//!   [`IntegrationMode::DryRun`] previews without staging anything.
//!   There is no auto-confirm and **no push** -- remote propagation
//!   stays out of the daemon and in the operator's hands.
//! * **Eligibility**: only [`super::PlanStatus::Completed`] plans are
//!   eligible. `Failed`, `Cancelled`, `Interrupted`, `Draft`,
//!   `Confirmed`, and `Running` are refused with
//!   [`IntegrationRefusal::PlanStatus`]. Partial-success integration
//!   is rejected on principle: if any step is not
//!   [`super::StepStatus::Completed`] (including `Skipped` /
//!   `Cancelled`), [`assess_eligibility`] returns
//!   [`IntegrationRefusal::PartialSuccess`]. The audit recommendation
//!   prefers a clean re-run over picking through a half-merged tree.
//! * **Topological tip-only cherry-pick**: only the *tip* steps
//!   (those with no dependents inside the plan's DAG) contribute
//!   commits to the integration branch. Their dependency closure is
//!   already merged into them by the worktree spawner's per-layer
//!   merges, so cherry-picking the tips in topological order produces
//!   a linear history equivalent to the full plan tree without
//!   re-applying interior commits.
//! * **Integration branch**: cherry-picks land on a *dedicated*
//!   integration branch ([`integration_branch_ref`]) that the
//!   operator can inspect, verify, and either keep or discard. The
//!   base branch is never touched until promotion.
//! * **Promotion to base**: fast-forward-only. If the configured base
//!   advanced after the integration branch was created (i.e. the
//!   integration branch is no longer a fast-forward of the base),
//!   the daemon refuses with [`IntegrationFailure::BaseAdvanced`] and
//!   the operator must rebase the integration branch and re-confirm.
//!   A dirty base ([`IntegrationFailure::DirtyBase`]) likewise blocks
//!   promotion -- the daemon refuses to stash or commit on the
//!   operator's behalf.
//! * **Conflict handling**: cherry-pick conflicts surface as
//!   [`IntegrationFailure::CherryPickConflict`] with the offending
//!   step id and the conflicting paths. The partial integration
//!   branch is left in the repo for human resolution; the daemon
//!   does not retry.
//! * **Cleanup**: see [`CleanupPolicy`]. Default
//!   [`CleanupPolicy::KeepOnFailure`] keeps the integration branch
//!   and per-step branches when anything went wrong, drops them on a
//!   clean promotion. [`CleanupPolicy::KeepAlways`] retains every
//!   artefact; [`CleanupPolicy::DropAlways`] is for ephemeral CI
//!   integrations that own their state externally.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Plan, PlanStatus, PlanStep, StepStatus};

/// Per-step branch ref the worktree spawner creates while a plan
/// runs. Mirrors the format produced by
/// `flowd_cli::spawner::step_branch`; pinned here (and re-used by
/// [`assess_eligibility`]) so the contract test suite can assert the
/// cherry-pick targets without depending on the CLI crate.
///
/// Inputs are sanitised with the same rules as the spawner: any
/// character outside `[A-Za-z0-9_.-]` collapses to `-`, leading and
/// trailing `-` are stripped, and an empty result becomes
/// `unnamed`. Keep in lock-step with the spawner; a divergence here
/// silently makes integration target the wrong branch.
#[must_use]
pub fn step_branch_ref(project: &str, plan_id: Uuid, step_id: &str) -> String {
    format!(
        "flowd/{}/{}/{}",
        sanitize_ref(project),
        plan_id.simple(),
        sanitize_ref(step_id),
    )
}

/// Dedicated integration branch where cherry-picks accumulate before
/// the operator confirms promotion to [`IntegrationConfig::base_branch`].
///
/// The `flowd-integrate/` prefix is distinct from the `flowd/`
/// per-step-branch namespace so a stray `git branch --list flowd/*`
/// in the operator's tooling does not pull integration branches into
/// per-plan cleanup loops.
#[must_use]
pub fn integration_branch_ref(project: &str, plan_id: Uuid) -> String {
    format!(
        "flowd-integrate/{}/{}",
        sanitize_ref(project),
        plan_id.simple(),
    )
}

/// Plan execution mode for `plan_integrate`.
///
/// V1 deliberately has no `Auto` / `AutoPush` variant. Promotion to
/// the configured base is always human-gated: the daemon stages the
/// integration branch, the operator inspects it, then re-invokes
/// `plan_integrate` with the same plan id to confirm. `Confirm` is
/// the default the front door selects when the caller does not
/// specify a mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntegrationMode {
    /// Stage the integration branch and stop. The operator must
    /// re-invoke to promote to base.
    #[default]
    Confirm,
    /// Compute the planned operations but apply nothing -- no branch
    /// is created, no cherry-picks run. Used to render a preview.
    DryRun,
}

/// What to do with the integration branch and per-step branches once
/// promotion succeeds (or definitively fails).
///
/// The choice is deliberately tri-state rather than a boolean: a
/// failed cherry-pick run wants the artefacts retained for triage
/// even when a successful run would clean up, and a CI flow that
/// owns its own branch lifecycle wants to drop everything either
/// way.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupPolicy {
    /// Default. Drop the integration branch and the per-step
    /// branches after a clean fast-forward to base. Keep them on
    /// any failure path so the operator can inspect what happened.
    #[default]
    KeepOnFailure,
    /// Retain every branch unconditionally. Useful when the
    /// integration is part of a longer audit trail or the operator
    /// wants to compare runs.
    KeepAlways,
    /// Drop every branch after the run finishes, success or
    /// failure. Suited to ephemeral CI environments where the
    /// branches outlive their utility immediately.
    DropAlways,
}

/// Operator-supplied policy for a single `plan_integrate` invocation.
///
/// Carries no per-plan identifiers (`plan_id`, project) -- those
/// derive from the plan being integrated. Keeps the type cheaply
/// clonable so the front door can pass it down without juggling
/// references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// The branch the integration eventually promotes to via
    /// fast-forward. Required: v1 has no notion of an inferred
    /// default ("main"-ish) base; the operator must declare it so a
    /// misconfigured workspace fails with a refusal instead of
    /// quietly merging into the wrong branch.
    pub base_branch: String,
    /// What the daemon does with branches once the run finishes.
    pub cleanup: CleanupPolicy,
}

impl IntegrationConfig {
    /// Build a config with a declared base branch and the
    /// [`CleanupPolicy`] default ([`CleanupPolicy::KeepOnFailure`]).
    #[must_use]
    pub fn with_base(base_branch: impl Into<String>) -> Self {
        Self {
            base_branch: base_branch.into(),
            cleanup: CleanupPolicy::default(),
        }
    }
}

/// Lifecycle status of a plan's integration, tracked independently of
/// the plan's terminal [`super::PlanStatus`].
///
/// A plan in [`super::PlanStatus::Completed`] keeps its terminal status
/// while its integration progresses through these states; conflating
/// the two would force the plan back into a non-terminal status
/// whenever the operator staged or retried integration, breaking the
/// eligibility contract [`assess_eligibility`] enforces.
///
/// Conceptual transitions:
///
/// ```text
/// Pending ─► InProgress ─► Staged ─► InProgress ─► Promoted
///                  │                       │
///                  └────────► Failed ◄─────┘
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegrationStatus {
    /// `plan_integrate` has not been invoked for this plan, or every
    /// prior attempt was reset. Default for plans that have never
    /// surfaced to the integration front door.
    #[default]
    Pending,
    /// Integration is in flight: cherry-picks running, or fast-forward
    /// to base in progress. Observed only between an
    /// `IntegrationStarted` event and the next terminal event.
    InProgress,
    /// Confirm-mode integration finished cleanly: the integration
    /// branch is ready, awaiting a follow-up `plan_integrate` call to
    /// promote.
    Staged,
    /// Fast-forward to base completed. The base branch now points at
    /// the integration tip.
    Promoted,
    /// A run failed (refusal *or* runtime failure). The structured
    /// cause lives on [`IntegrationMetadata::failure`]; this enum
    /// carries only the lifecycle marker so renderers can switch on
    /// status alone without pattern-matching the cause.
    Failed,
}

impl IntegrationStatus {
    /// True for states from which no further transition is expected
    /// without operator action ([`Self::Promoted`], [`Self::Failed`]).
    /// [`Self::Staged`] is *not* terminal: a staged branch waits for a
    /// follow-up promote call.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Promoted | Self::Failed)
    }
}

/// Per-plan integration progress. Persisted on [`super::Plan`] so a
/// daemon restart (or a reader who only has the plan struct) can render
/// the right integration status without replaying the event log.
///
/// `None` on the plan means "no integration attempt observed"; this is
/// distinct from `Some(IntegrationMetadata { status: Pending, .. })`,
/// which means an operator opened the integration surface (e.g. a
/// dry-run) and the daemon retains the chosen policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrationMetadata {
    pub status: IntegrationStatus,
    /// Integration branch this run is targeting / has produced.
    pub integration_branch: String,
    /// Base branch this run will (or did) promote to.
    pub base_branch: String,
    pub mode: IntegrationMode,
    pub cleanup: CleanupPolicy,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    /// Set when [`Self::status`] is [`IntegrationStatus::Failed`].
    /// Carries the structured runtime failure when one is available;
    /// pre-flight refusals are surfaced via
    /// [`IntegrationMetadata::refusal`] so the two stay typed
    /// separately.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<IntegrationFailure>,
    /// Set when [`Self::status`] is [`IntegrationStatus::Failed`] *and*
    /// the failure was a deterministic refusal rather than a runtime
    /// error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<IntegrationRefusal>,
}

impl IntegrationMetadata {
    /// Build a fresh `Pending` metadata record from a validated
    /// request. Used by the front door before any git work runs so the
    /// plan persists the operator-chosen policy alongside its terminal
    /// status.
    #[must_use]
    pub fn pending(plan_id: Uuid, project: &str, request: &PlanIntegrateRequest) -> Self {
        Self {
            status: IntegrationStatus::Pending,
            integration_branch: integration_branch_ref(project, plan_id),
            base_branch: request.config.base_branch.clone(),
            mode: request.mode,
            cleanup: request.config.cleanup,
            started_at: None,
            completed_at: None,
            failure: None,
            refusal: None,
        }
    }
}

/// Validated request to integrate a plan.
///
/// Built by the front door from raw caller input; constructing one
/// guarantees the [`IntegrationConfig::base_branch`] is non-empty
/// (other invariants live further down the pipeline).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanIntegrateRequest {
    pub plan_id: Uuid,
    pub mode: IntegrationMode,
    pub config: IntegrationConfig,
}

impl PlanIntegrateRequest {
    /// Validate the request shape (non-empty base branch, defaulted
    /// mode/cleanup if the caller left them unset). Pure -- runs no
    /// git, touches no plan state.
    ///
    /// # Errors
    /// Returns [`IntegrationRefusal::InvalidConfig`] when the
    /// configured base branch is empty / whitespace-only.
    pub fn new(
        plan_id: Uuid,
        mode: IntegrationMode,
        config: IntegrationConfig,
    ) -> Result<Self, IntegrationRefusal> {
        if config.base_branch.trim().is_empty() {
            return Err(IntegrationRefusal::InvalidConfig {
                field: "base_branch".into(),
                reason: "base_branch must be a non-empty git ref name".into(),
            });
        }
        Ok(Self {
            plan_id,
            mode,
            config,
        })
    }
}

/// One cherry-pick the integrator will apply, in apply order.
///
/// For v1 there is exactly one entry per *tip* step in the plan;
/// see [`assess_eligibility`] for the topological-tip rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CherryPick {
    pub step_id: String,
    /// Branch ref that holds the step's commit (and its dependency
    /// closure, by virtue of the per-layer merges the worktree
    /// spawner performs). The integrator cherry-picks the tip of
    /// this ref onto the integration branch.
    pub source_branch: String,
}

/// Concrete plan of operations the integrator will perform once the
/// operator confirms.
///
/// Pure data: this is what `assess_eligibility` returns, and what
/// the eventual git-driving layer consumes. No git state has been
/// touched at this point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrationPlan {
    pub plan_id: Uuid,
    pub project: String,
    pub base_branch: String,
    pub integration_branch: String,
    /// Topologically ordered cherry-picks (dependencies before
    /// dependents). One entry per tip step.
    pub cherry_picks: Vec<CherryPick>,
    pub mode: IntegrationMode,
    pub cleanup: CleanupPolicy,
}

/// Final outcome reported back to the caller.
///
/// `Promoted` is the success terminus; the rest are intermediate
/// states a `Confirm` flow might land on (preview only, conflict
/// surfaced, etc.). Constructed by the eventual git-driving layer;
/// pinned here so the protocol surface is stable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlanIntegrateOutcome {
    /// `mode == DryRun`. The plan was eligible and the operations
    /// are described in `intended`; nothing was applied.
    DryRun { intended: IntegrationPlan },
    /// `mode == Confirm`. The integration branch was created and
    /// every cherry-pick succeeded, but promotion to base is
    /// awaiting a follow-up confirm call.
    Staged {
        intended: IntegrationPlan,
        /// Tip commit on the integration branch after the last
        /// successful cherry-pick. Echoed back so the operator's
        /// front door can render an at-a-glance ref.
        integration_tip: String,
    },
    /// Fast-forward promotion succeeded. The base now points at
    /// the integration tip.
    Promoted {
        intended: IntegrationPlan,
        promoted_tip: String,
    },
}

/// Eligibility rejections. Returned **before** any git command runs.
///
/// These are the deterministic refusals: things knowable purely from
/// the plan struct + the request payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IntegrationRefusal {
    /// The plan is not in [`PlanStatus::Completed`]. Carries the
    /// observed status so the caller can render an actionable
    /// message ("Failed plans must be reset" vs. "Running plans
    /// must finish").
    #[error("plan status `{observed:?}` is not eligible for integration; expected `Completed`")]
    PlanStatus { observed: PlanStatus },
    /// The plan is `Completed` overall but at least one step is not
    /// itself `Completed` (e.g. `Skipped`, `Cancelled`). v1 refuses
    /// partial integration; the operator should re-run.
    #[error(
        "plan has {non_completed_count} step(s) that are not Completed (`{first_step_id}`: \
         `{first_step_status:?}`); refusing partial integration"
    )]
    PartialSuccess {
        non_completed_count: usize,
        first_step_id: String,
        first_step_status: StepStatus,
    },
    /// The plan has no steps. Should not happen for a `Completed`
    /// plan in practice (validation rejects empty plans), but the
    /// integrator carries its own guard rather than relying on
    /// upstream invariants.
    #[error("plan has no steps; nothing to integrate")]
    NoSteps,
    /// The plan has no tip steps. Indicates a malformed dependency
    /// graph; should be unreachable for a plan that passed
    /// [`super::validate_plan`], but covered defensively.
    #[error("plan has no tip steps (every step has a dependent); cannot pick an integration tip")]
    NoTipSteps,
    /// The request payload is internally invalid (e.g. empty base
    /// branch). Surfaced via [`PlanIntegrateRequest::new`].
    ///
    /// `field` is owned (not `&'static str`) so the enclosing refusal
    /// can round-trip through [`serde::Deserialize`]; embedding it in
    /// [`IntegrationMetadata`] requires that.
    #[error("invalid integration config field `{field}`: {reason}")]
    InvalidConfig { field: String, reason: String },
}

/// Runtime failures the git-driving layer can surface. Pinned here so
/// the protocol surface is stable before the apply code lands.
///
/// Distinct from [`IntegrationRefusal`]: those are pre-flight checks
/// against the plan struct, these can only be observed by
/// interacting with the working repo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IntegrationFailure {
    /// The configured base branch does not exist in the repo.
    #[error("base branch `{base}` does not exist in the repository")]
    MissingBase { base: String },
    /// One of the per-step branches the cherry-pick selection
    /// references is missing. Suggests cleanup ran prematurely or
    /// the plan was integrated previously.
    #[error("step branch `{branch}` for step `{step_id}` is missing")]
    MissingStepBranch { step_id: String, branch: String },
    /// The repo working tree (or the base branch checkout) has
    /// uncommitted changes. v1 refuses to stash or commit on the
    /// operator's behalf.
    #[error("base branch checkout has uncommitted changes; cannot promote without a clean tree")]
    DirtyBase,
    /// A cherry-pick produced merge conflicts. The integration
    /// branch is left at the offending step for human resolution.
    #[error("cherry-pick of step `{step_id}` produced conflicts: {conflicting_paths:?}")]
    CherryPickConflict {
        step_id: String,
        conflicting_paths: Vec<String>,
    },
    /// The configured base advanced after the integration branch
    /// was created, so the integration branch is no longer a
    /// fast-forward of base. The operator must rebase and re-confirm.
    #[error(
        "base branch `{base}` advanced from `{base_tip_at_assess}` to `{observed_base_tip}` \
         after the integration was staged; integration branch is not a fast-forward"
    )]
    BaseAdvanced {
        base: String,
        base_tip_at_assess: String,
        observed_base_tip: String,
    },
}

/// Pure eligibility + plan-of-operations check.
///
/// Decides whether the plan can be integrated and, if so, returns
/// the [`IntegrationPlan`] the git-driving layer should apply. Runs
/// no git commands; uses only the plan struct and the request.
///
/// # Errors
/// Returns an [`IntegrationRefusal`] for any deterministic
/// ineligibility reason -- non-`Completed` plan status, partial step
/// success, malformed graph.
pub fn assess_eligibility(
    plan: &Plan,
    request: &PlanIntegrateRequest,
) -> Result<IntegrationPlan, IntegrationRefusal> {
    if plan.status != PlanStatus::Completed {
        return Err(IntegrationRefusal::PlanStatus {
            observed: plan.status,
        });
    }
    if plan.steps.is_empty() {
        return Err(IntegrationRefusal::NoSteps);
    }
    let non_completed: Vec<&PlanStep> = plan
        .steps
        .iter()
        .filter(|s| s.status != StepStatus::Completed)
        .collect();
    if let Some(first) = non_completed.first() {
        return Err(IntegrationRefusal::PartialSuccess {
            non_completed_count: non_completed.len(),
            first_step_id: first.id.clone(),
            first_step_status: first.status,
        });
    }

    let cherry_picks = topological_tip_cherry_picks(plan)?;

    Ok(IntegrationPlan {
        plan_id: plan.id,
        project: plan.project.clone(),
        base_branch: request.config.base_branch.clone(),
        integration_branch: integration_branch_ref(&plan.project, plan.id),
        cherry_picks,
        mode: request.mode,
        cleanup: request.config.cleanup,
    })
}

/// Compute the topological tip-only cherry-pick order.
///
/// "Tip step" = a step with no dependents inside the plan's DAG.
/// Their commits already carry the dependency closure (the worktree
/// spawner merges every dep branch before invoking the agent), so a
/// cherry-pick of each tip's branch tip reproduces the full plan
/// tree linearly.
///
/// Order is the topological-layer order [`Plan::execution_layers`]
/// returns, restricted to tips, then sorted within a layer by step
/// id for determinism. This matches the order the worktree spawner
/// runs steps in, so reverting a cherry-pick maps cleanly to one
/// step's worth of work.
///
/// # Errors
/// [`IntegrationRefusal::NoTipSteps`] when the plan's DAG has no
/// leaves (defensive: should be unreachable post-validation).
pub fn topological_tip_cherry_picks(plan: &Plan) -> Result<Vec<CherryPick>, IntegrationRefusal> {
    let layers = plan.execution_layers().map_err(|_| {
        // execution_layers() only fails on a cycle, which validate_plan
        // already rejects. Map to NoTipSteps so callers get a stable
        // refusal variant instead of a leaked PlanValidation.
        IntegrationRefusal::NoTipSteps
    })?;

    // Build the dependent set: any step id that appears in another
    // step's `depends_on` is *not* a tip.
    let mut has_dependent: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for s in &plan.steps {
        for dep in &s.depends_on {
            has_dependent.insert(dep.as_str());
        }
    }

    let mut picks: Vec<CherryPick> = Vec::new();
    for layer in &layers {
        let mut tips_in_layer: Vec<&String> = layer
            .iter()
            .filter(|id| !has_dependent.contains(id.as_str()))
            .collect();
        tips_in_layer.sort_unstable();
        for tip in tips_in_layer {
            picks.push(CherryPick {
                step_id: tip.clone(),
                source_branch: step_branch_ref(&plan.project, plan.id, tip),
            });
        }
    }

    if picks.is_empty() {
        return Err(IntegrationRefusal::NoTipSteps);
    }
    Ok(picks)
}

/// Sanitisation rules for the per-step branch ref.
///
/// Mirrors `flowd_cli::spawner::sanitize_ref`: any character outside
/// `[A-Za-z0-9_.-]` becomes `-`, leading and trailing `-` are
/// stripped, and an empty result becomes `unnamed`.
fn sanitize_ref(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('-').to_owned();
    if trimmed.is_empty() {
        "unnamed".into()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::{Plan, PlanStatus, PlanStep, StepStatus};

    fn step(id: &str, deps: &[&str]) -> PlanStep {
        PlanStep {
            id: id.into(),
            agent_type: "echo".into(),
            prompt: format!("do {id}"),
            depends_on: deps.iter().map(|s| (*s).to_owned()).collect(),
            timeout_secs: None,
            retry_count: 0,
            status: StepStatus::Completed,
            output: Some("ok".into()),
            error: None,
            started_at: None,
            completed_at: None,
        }
    }

    fn completed_plan(steps: Vec<PlanStep>) -> Plan {
        let mut p = Plan::new("p", "proj", steps);
        p.status = PlanStatus::Completed;
        p
    }

    fn fixture_request(plan_id: Uuid) -> PlanIntegrateRequest {
        PlanIntegrateRequest::new(
            plan_id,
            IntegrationMode::Confirm,
            IntegrationConfig::with_base("main"),
        )
        .expect("fixture request must validate")
    }

    // -------- Branch ref naming -----------------------------------------

    #[test]
    fn step_branch_ref_uses_simple_uuid_and_sanitises_segments() {
        let id = Uuid::nil();
        let r = step_branch_ref("My Project!", id, "step/with spaces");
        assert_eq!(
            r,
            "flowd/My-Project/00000000000000000000000000000000/step-with-spaces"
        );
    }

    #[test]
    fn integration_branch_ref_uses_distinct_prefix() {
        let id = Uuid::nil();
        let r = integration_branch_ref("proj", id);
        assert_eq!(r, "flowd-integrate/proj/00000000000000000000000000000000");
        // Must not collide with the per-step namespace -- a shared
        // prefix would cause cleanup loops keyed on `flowd/*` to
        // sweep up integration branches by accident.
        assert!(!r.starts_with("flowd/"));
    }

    #[test]
    fn step_branch_ref_empty_segment_becomes_unnamed() {
        let id = Uuid::nil();
        let r = step_branch_ref("", id, "");
        assert_eq!(r, "flowd/unnamed/00000000000000000000000000000000/unnamed");
    }

    // -------- Request validation ----------------------------------------

    #[test]
    fn request_rejects_empty_base_branch() {
        let err = PlanIntegrateRequest::new(
            Uuid::new_v4(),
            IntegrationMode::Confirm,
            IntegrationConfig::with_base(""),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            IntegrationRefusal::InvalidConfig { ref field, .. } if field == "base_branch"
        ));
    }

    #[test]
    fn request_rejects_whitespace_base_branch() {
        let err = PlanIntegrateRequest::new(
            Uuid::new_v4(),
            IntegrationMode::Confirm,
            IntegrationConfig::with_base("   "),
        )
        .unwrap_err();
        assert!(matches!(err, IntegrationRefusal::InvalidConfig { .. }));
    }

    #[test]
    fn defaults_are_confirm_and_keep_on_failure() {
        // Pin the v1 documented defaults so a refactor that swaps
        // them silently flips real users' behaviour (e.g. an
        // accidental `Default::default()` returning DryRun would
        // make every integrate call a no-op).
        assert_eq!(IntegrationMode::default(), IntegrationMode::Confirm);
        assert_eq!(CleanupPolicy::default(), CleanupPolicy::KeepOnFailure);
    }

    // -------- Eligibility: plan status ----------------------------------

    #[test]
    fn refuses_draft_plan() {
        let mut plan = completed_plan(vec![step("a", &[])]);
        plan.status = PlanStatus::Draft;
        let err = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap_err();
        assert!(matches!(
            err,
            IntegrationRefusal::PlanStatus {
                observed: PlanStatus::Draft
            }
        ));
    }

    #[test]
    fn refuses_running_plan() {
        let mut plan = completed_plan(vec![step("a", &[])]);
        plan.status = PlanStatus::Running;
        let err = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap_err();
        assert!(matches!(
            err,
            IntegrationRefusal::PlanStatus {
                observed: PlanStatus::Running
            }
        ));
    }

    #[test]
    fn refuses_failed_plan() {
        let mut plan = completed_plan(vec![step("a", &[])]);
        plan.status = PlanStatus::Failed;
        let err = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap_err();
        assert!(matches!(
            err,
            IntegrationRefusal::PlanStatus {
                observed: PlanStatus::Failed
            }
        ));
    }

    #[test]
    fn refuses_cancelled_plan() {
        let mut plan = completed_plan(vec![step("a", &[])]);
        plan.status = PlanStatus::Cancelled;
        let err = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap_err();
        assert!(matches!(
            err,
            IntegrationRefusal::PlanStatus {
                observed: PlanStatus::Cancelled
            }
        ));
    }

    #[test]
    fn refuses_interrupted_plan() {
        let mut plan = completed_plan(vec![step("a", &[])]);
        plan.status = PlanStatus::Interrupted;
        let err = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap_err();
        assert!(matches!(
            err,
            IntegrationRefusal::PlanStatus {
                observed: PlanStatus::Interrupted
            }
        ));
    }

    // -------- Eligibility: partial success ------------------------------

    #[test]
    fn refuses_plan_with_skipped_step() {
        let a = step("a", &[]);
        let mut b = step("b", &[]);
        b.status = StepStatus::Skipped;
        let plan = completed_plan(vec![a, b]);
        let err = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap_err();
        match err {
            IntegrationRefusal::PartialSuccess {
                non_completed_count,
                first_step_id,
                first_step_status,
            } => {
                assert_eq!(non_completed_count, 1);
                assert_eq!(first_step_id, "b");
                assert_eq!(first_step_status, StepStatus::Skipped);
            }
            other => panic!("expected PartialSuccess, got {other:?}"),
        }
    }

    #[test]
    fn refuses_plan_with_cancelled_step() {
        let mut a = step("a", &[]);
        a.status = StepStatus::Cancelled;
        let plan = completed_plan(vec![a]);
        let err = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap_err();
        assert!(matches!(err, IntegrationRefusal::PartialSuccess { .. }));
    }

    #[test]
    fn refuses_plan_with_pending_step() {
        let mut a = step("a", &[]);
        a.status = StepStatus::Pending;
        let plan = completed_plan(vec![a]);
        let err = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap_err();
        assert!(matches!(err, IntegrationRefusal::PartialSuccess { .. }));
    }

    // -------- Topological tip-only cherry-pick selection ----------------

    #[test]
    fn single_step_plan_picks_only_step() {
        let plan = completed_plan(vec![step("a", &[])]);
        let req = fixture_request(plan.id);
        let result = assess_eligibility(&plan, &req).unwrap();
        assert_eq!(result.cherry_picks.len(), 1);
        assert_eq!(result.cherry_picks[0].step_id, "a");
        assert_eq!(
            result.cherry_picks[0].source_branch,
            step_branch_ref(&plan.project, plan.id, "a")
        );
    }

    #[test]
    fn linear_chain_picks_only_final_tip() {
        // a -> b -> c. Only `c` is a tip; its branch already merges
        // a and b in via the spawner's per-step dep merges.
        let plan = completed_plan(vec![step("a", &[]), step("b", &["a"]), step("c", &["b"])]);
        let result = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap();
        let ids: Vec<&str> = result
            .cherry_picks
            .iter()
            .map(|p| p.step_id.as_str())
            .collect();
        assert_eq!(ids, vec!["c"]);
    }

    #[test]
    fn diamond_plan_picks_only_join_step() {
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d   <- the only tip
        let plan = completed_plan(vec![
            step("a", &[]),
            step("b", &["a"]),
            step("c", &["a"]),
            step("d", &["b", "c"]),
        ]);
        let result = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap();
        let ids: Vec<&str> = result
            .cherry_picks
            .iter()
            .map(|p| p.step_id.as_str())
            .collect();
        assert_eq!(ids, vec!["d"]);
    }

    #[test]
    fn parallel_independent_tips_are_returned_in_layer_then_id_order() {
        // a -> c, b -> d. Tips are c and d; both live on the same
        // (final) topological layer, so id-sort wins for determinism.
        let plan = completed_plan(vec![
            step("a", &[]),
            step("b", &[]),
            step("c", &["a"]),
            step("d", &["b"]),
        ]);
        let result = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap();
        let ids: Vec<&str> = result
            .cherry_picks
            .iter()
            .map(|p| p.step_id.as_str())
            .collect();
        assert_eq!(ids, vec!["c", "d"]);
    }

    #[test]
    fn multi_layer_tips_order_dependencies_before_dependents() {
        // a (tip in layer 0) and b -> c (tip in layer 1). The
        // cherry-pick stream must place `a` before `c` so the
        // integration branch does not try to apply `c` against an
        // unprepared base.
        let plan = completed_plan(vec![step("a", &[]), step("b", &[]), step("c", &["b"])]);
        let result = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap();
        let ids: Vec<&str> = result
            .cherry_picks
            .iter()
            .map(|p| p.step_id.as_str())
            .collect();
        assert_eq!(ids, vec!["a", "c"]);
    }

    #[test]
    fn cherry_pick_branches_match_spawner_naming() {
        // Lock-step with `flowd_cli::spawner::step_branch`: any drift
        // means the integrator reaches for branches the spawner
        // never created. The string is reconstructed via the
        // contract's own `step_branch_ref` so the test fails loudly
        // rather than silently when the format changes.
        let plan = completed_plan(vec![step("a", &[])]);
        let result = assess_eligibility(&plan, &fixture_request(plan.id)).unwrap();
        let expected = format!("flowd/proj/{}/a", plan.id.simple());
        assert_eq!(result.cherry_picks[0].source_branch, expected);
    }

    // -------- Plan shape: integration branch, request echoes ------------

    #[test]
    fn integration_plan_echoes_request_fields_unchanged() {
        let plan = completed_plan(vec![step("a", &[])]);
        let req = PlanIntegrateRequest::new(
            plan.id,
            IntegrationMode::DryRun,
            IntegrationConfig {
                base_branch: "release/2026.04".into(),
                cleanup: CleanupPolicy::KeepAlways,
            },
        )
        .unwrap();
        let result = assess_eligibility(&plan, &req).unwrap();
        assert_eq!(result.plan_id, plan.id);
        assert_eq!(result.project, plan.project);
        assert_eq!(result.base_branch, "release/2026.04");
        assert_eq!(result.mode, IntegrationMode::DryRun);
        assert_eq!(result.cleanup, CleanupPolicy::KeepAlways);
        assert_eq!(
            result.integration_branch,
            integration_branch_ref(&plan.project, plan.id)
        );
    }

    // -------- Wire stability of the outcome / failure surface -----------
    //
    // The MCP / CLI layers serialise these enums in their JSON-RPC
    // payloads. Pinning the discriminants here means a rename that
    // would silently break a client surfaces in this crate's own
    // test suite first.

    #[test]
    fn refusal_serialises_with_stable_discriminants() {
        let r = IntegrationRefusal::PlanStatus {
            observed: PlanStatus::Failed,
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["kind"], "plan_status");
        assert_eq!(v["observed"], "failed");

        let r = IntegrationRefusal::PartialSuccess {
            non_completed_count: 2,
            first_step_id: "b".into(),
            first_step_status: StepStatus::Skipped,
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["kind"], "partial_success");
        assert_eq!(v["non_completed_count"], 2);
        assert_eq!(v["first_step_status"], "skipped");
    }

    #[test]
    fn failure_serialises_with_stable_discriminants() {
        let f = IntegrationFailure::DirtyBase;
        assert_eq!(serde_json::to_value(&f).unwrap()["kind"], "dirty_base");

        let f = IntegrationFailure::CherryPickConflict {
            step_id: "c".into(),
            conflicting_paths: vec!["src/lib.rs".into()],
        };
        let v = serde_json::to_value(&f).unwrap();
        assert_eq!(v["kind"], "cherry_pick_conflict");
        assert_eq!(v["step_id"], "c");

        let f = IntegrationFailure::BaseAdvanced {
            base: "main".into(),
            base_tip_at_assess: "abc".into(),
            observed_base_tip: "def".into(),
        };
        let v = serde_json::to_value(&f).unwrap();
        assert_eq!(v["kind"], "base_advanced");
    }

    #[test]
    fn outcome_serialises_with_stable_discriminants() {
        let plan = completed_plan(vec![step("a", &[])]);
        let req = fixture_request(plan.id);
        let intended = assess_eligibility(&plan, &req).unwrap();

        let dry = PlanIntegrateOutcome::DryRun {
            intended: intended.clone(),
        };
        assert_eq!(serde_json::to_value(&dry).unwrap()["kind"], "dry_run");

        let staged = PlanIntegrateOutcome::Staged {
            intended: intended.clone(),
            integration_tip: "deadbeef".into(),
        };
        assert_eq!(serde_json::to_value(&staged).unwrap()["kind"], "staged");

        let promoted = PlanIntegrateOutcome::Promoted {
            intended,
            promoted_tip: "deadbeef".into(),
        };
        assert_eq!(serde_json::to_value(&promoted).unwrap()["kind"], "promoted");
    }

    // -------- IntegrationStatus / IntegrationMetadata -------------------

    #[test]
    fn integration_status_default_is_pending() {
        // The plan-side metadata seeds with `Pending` until the first
        // run produces a real transition. A default flip would make
        // every freshly-loaded plan look mid-integration.
        assert_eq!(IntegrationStatus::default(), IntegrationStatus::Pending);
    }

    #[test]
    fn integration_status_serialises_snake_case() {
        // The wire format is checked here so a `rename_all` change to
        // `lowercase` (which would compile but flip
        // `in_progress` to `inprogress`) is caught at this layer.
        for (variant, expected) in [
            (IntegrationStatus::Pending, "pending"),
            (IntegrationStatus::InProgress, "in_progress"),
            (IntegrationStatus::Staged, "staged"),
            (IntegrationStatus::Promoted, "promoted"),
            (IntegrationStatus::Failed, "failed"),
        ] {
            let v = serde_json::to_value(variant).unwrap();
            assert_eq!(v.as_str(), Some(expected));
            let back: IntegrationStatus = serde_json::from_value(v).unwrap();
            assert_eq!(back, variant);
        }
    }

    #[test]
    fn integration_status_terminal_only_for_promoted_and_failed() {
        // `Staged` is deliberately *not* terminal: it awaits a follow-up
        // promote call, and a renderer that treated it as terminal
        // would hide the fact that the operator still has a step to take.
        assert!(!IntegrationStatus::Pending.is_terminal());
        assert!(!IntegrationStatus::InProgress.is_terminal());
        assert!(!IntegrationStatus::Staged.is_terminal());
        assert!(IntegrationStatus::Promoted.is_terminal());
        assert!(IntegrationStatus::Failed.is_terminal());
    }

    #[test]
    fn integration_metadata_round_trips_with_failure_set() {
        let meta = IntegrationMetadata {
            status: IntegrationStatus::Failed,
            integration_branch: "flowd-integrate/proj/abc".into(),
            base_branch: "main".into(),
            mode: IntegrationMode::Confirm,
            cleanup: CleanupPolicy::KeepAlways,
            started_at: None,
            completed_at: None,
            failure: Some(IntegrationFailure::DirtyBase),
            refusal: None,
        };
        let json = serde_json::to_value(&meta).unwrap();
        let back: IntegrationMetadata = serde_json::from_value(json).unwrap();
        assert_eq!(back, meta);
    }

    #[test]
    fn integration_metadata_omits_none_failure_and_refusal_fields() {
        // `skip_serializing_if = "Option::is_none"` keeps the row
        // compact for the common Pending / Staged / Promoted cases
        // where neither cause is meaningful. A leaked `null` would
        // bloat the events table and confuse downstream consumers.
        let meta = IntegrationMetadata::pending(
            Uuid::nil(),
            "proj",
            &PlanIntegrateRequest::new(
                Uuid::nil(),
                IntegrationMode::Confirm,
                IntegrationConfig::with_base("main"),
            )
            .unwrap(),
        );
        let json = serde_json::to_value(&meta).unwrap();
        let obj = json.as_object().unwrap();
        assert!(!obj.contains_key("failure"));
        assert!(!obj.contains_key("refusal"));
    }

    #[test]
    fn integration_metadata_pending_seeds_branch_from_helper() {
        // Pinning the seed against the branch helper makes sure the
        // metadata and the actual cherry-pick target stay in lock-step
        // -- a divergence here would have the plan advertise a branch
        // the integrator never produces.
        let plan_id = Uuid::new_v4();
        let req = PlanIntegrateRequest::new(
            plan_id,
            IntegrationMode::DryRun,
            IntegrationConfig::with_base("release"),
        )
        .unwrap();
        let meta = IntegrationMetadata::pending(plan_id, "proj", &req);
        assert_eq!(meta.status, IntegrationStatus::Pending);
        assert_eq!(meta.base_branch, "release");
        assert_eq!(meta.mode, IntegrationMode::DryRun);
        assert_eq!(meta.cleanup, CleanupPolicy::KeepOnFailure);
        assert_eq!(
            meta.integration_branch,
            integration_branch_ref("proj", plan_id)
        );
    }

    #[test]
    fn assess_eligibility_ignores_integration_metadata() {
        // The integration sub-state must not feed back into the
        // eligibility check: a plan that has been Staged or even
        // Promoted is still `PlanStatus::Completed`, and re-running
        // `assess_eligibility` on it (e.g. to re-render a dry-run)
        // must produce the same plan-of-operations rather than refuse.
        let mut plan = completed_plan(vec![step("a", &[])]);
        let req = fixture_request(plan.id);
        let baseline = assess_eligibility(&plan, &req).unwrap();

        plan.integration = Some(IntegrationMetadata {
            status: IntegrationStatus::Promoted,
            integration_branch: integration_branch_ref(&plan.project, plan.id),
            base_branch: "main".into(),
            mode: IntegrationMode::Confirm,
            cleanup: CleanupPolicy::KeepOnFailure,
            started_at: None,
            completed_at: None,
            failure: None,
            refusal: None,
        });
        let after = assess_eligibility(&plan, &req).unwrap();
        assert_eq!(after, baseline);
    }
}
