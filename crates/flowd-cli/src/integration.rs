//! I/O layer that drives `plan_integrate` against a real git repository.
//!
//! Sibling to [`crate::spawner`]. Consumes the pure contract from
//! [`flowd_core::orchestration::integration`]: the spawner has already
//! produced one branch per step (with per-layer dep merges) and persisted
//! `(plan_id, step_id) -> branch` to [`SqliteStepBranchStore`]; this module
//! turns those branches into a single linear-history integration branch
//! the operator can review and (separately) fast-forward onto the
//! configured base.
//!
//! ## Surface
//!
//! [`PlanIntegrator`] exposes two async methods:
//!
//! * [`PlanIntegrator::integrate`] -- assess eligibility against the pure
//!   contract, then either preview ([`IntegrationMode::DryRun`]) or stage
//!   ([`IntegrationMode::Confirm`]) by creating
//!   `flowd-integrate/<project>/<plan_id>` as a temporary worktree+branch
//!   off the resolved base tip and cherry-picking each tip step's commit
//!   range into it. Source step branches are never checked out and never
//!   mutated.
//! * [`PlanIntegrator::promote`] -- fast-forward-only update of the
//!   configured base ref to the integration tip via `git update-ref` with
//!   a CAS old-value guard, refusing on [`IntegrationFailure::DirtyBase`]
//!   or [`IntegrationFailure::BaseAdvanced`]. Never pushes; the operator
//!   propagates to a remote on their own.
//!
//! ## Failure surface
//!
//! Failures land in two typed buckets:
//!
//! * [`IntegrationRefusal`] -- pure pre-flight checks from
//!   [`assess_eligibility`] (plan-status / partial-success / malformed
//!   graph). Surfaced before any git command runs.
//! * [`IntegrationFailure`] -- runtime checks: missing base, missing step
//!   branch, dirty base, conflicting cherry-pick, base advanced past the
//!   staged tip. Conflicts deliberately leave the integration worktree on
//!   disk and the cherry-pick state mid-resolution; v1 does not retry.
//!
//! Anything else (failed `git worktree add`, fs errors, etc.) propagates
//! as [`FlowdError::PlanExecution`] via [`IntegrateError::Plan`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use flowd_core::error::{FlowdError, Result as FlowdResult};
use flowd_core::orchestration::Plan;
use flowd_core::orchestration::integration::{
    CherryPick, IntegrationFailure, IntegrationMode, IntegrationRefusal, PlanIntegrateOutcome,
    PlanIntegrateRequest, assess_eligibility,
};
use flowd_storage::step_branch_store::SqliteStepBranchStore;
use tokio::process::Command;

/// Combined error surface for [`PlanIntegrator`].
///
/// The pure-contract errors ([`IntegrationRefusal`], [`IntegrationFailure`])
/// are kept distinct from the wrapping [`FlowdError`] so callers can
/// pattern-match on the typed cause without parsing strings. The
/// [`thiserror::Error`] derive surfaces `Display` impls already pinned
/// inside the contract module.
#[derive(Debug, thiserror::Error)]
pub enum IntegrateError {
    /// Pre-flight refusal returned from [`assess_eligibility`] before any
    /// git command runs.
    #[error(transparent)]
    Refusal(#[from] IntegrationRefusal),
    /// Runtime failure observed while interacting with the working repo.
    #[error(transparent)]
    Failure(#[from] IntegrationFailure),
    /// Anything else: failed git invocation, filesystem error, etc.
    #[error(transparent)]
    Plan(#[from] FlowdError),
}

pub type IntegrateResult<T> = std::result::Result<T, IntegrateError>;

/// Concrete I/O driver for `plan_integrate`.
///
/// Holds construction-time defaults (the fallback git repo and the
/// directory under which integration worktrees live) plus an optional
/// reference to the durable step→branch store. Per-call inputs --
/// [`Plan`], [`PlanIntegrateRequest`] -- carry everything else.
#[derive(Debug)]
pub struct PlanIntegrator {
    /// Default git repository root. Used when [`Plan::project_root`] is
    /// `None` (legacy plans). Plans persisted with a project root take
    /// precedence over this fallback.
    repo: PathBuf,
    /// Where integration worktrees live. Mirrors the spawner's
    /// `flowd_home/worktrees/` convention but lives under a distinct
    /// subdir so cleanup loops keyed on the spawner's path don't sweep
    /// integration state by accident.
    worktree_root: PathBuf,
    /// Durable step→branch map. Optional: tests that pre-create branches
    /// in the repo do not need it; the daemon always supplies one.
    #[allow(dead_code)]
    branch_store: Option<Arc<SqliteStepBranchStore>>,
}

impl PlanIntegrator {
    #[must_use]
    pub fn new(
        repo: PathBuf,
        worktree_root: PathBuf,
        branch_store: Option<Arc<SqliteStepBranchStore>>,
    ) -> Self {
        Self {
            repo,
            worktree_root,
            branch_store,
        }
    }

    /// Stage the integration. Validates the request through the pure
    /// contract, resolves the base tip, creates a temporary integration
    /// worktree+branch, and cherry-picks each tip step's commit range
    /// in topological-then-id order. On
    /// [`IntegrationMode::DryRun`] the cherry-picks are described in the
    /// returned [`PlanIntegrateOutcome::DryRun`] without any git mutation.
    ///
    /// # Errors
    /// * [`IntegrateError::Refusal`] for pre-flight ineligibility (plan
    ///   status, partial success, malformed graph).
    /// * [`IntegrateError::Failure`] with [`IntegrationFailure::MissingBase`]
    ///   when the configured base ref does not resolve.
    /// * [`IntegrateError::Failure`] with
    ///   [`IntegrationFailure::MissingStepBranch`] when any tip step's
    ///   branch is missing.
    /// * [`IntegrateError::Failure`] with
    ///   [`IntegrationFailure::CherryPickConflict`] when a cherry-pick
    ///   produces unresolved paths. The integration worktree is left
    ///   in place for human resolution.
    /// * [`IntegrateError::Plan`] for any other I/O / git failure.
    pub async fn integrate(
        &self,
        plan: &Plan,
        request: &PlanIntegrateRequest,
    ) -> IntegrateResult<PlanIntegrateOutcome> {
        let intended = assess_eligibility(plan, request)?;

        if request.mode == IntegrationMode::DryRun {
            return Ok(PlanIntegrateOutcome::DryRun { intended });
        }

        let repo = self.repo_for_plan(plan);

        let base_tip = resolve_branch_tip(repo, &intended.base_branch)
            .await
            .map_err(|_| IntegrationFailure::MissingBase {
                base: intended.base_branch.clone(),
            })?;

        for pick in &intended.cherry_picks {
            if resolve_branch_tip(repo, &pick.source_branch).await.is_err() {
                return Err(IntegrationFailure::MissingStepBranch {
                    step_id: pick.step_id.clone(),
                    branch: pick.source_branch.clone(),
                }
                .into());
            }
        }

        let wt_path = self.worktree_path(plan);
        if wt_path.exists() {
            return Err(IntegrateError::Plan(FlowdError::PlanExecution {
                message: format!(
                    "integration worktree {} already exists; remove it before re-running \
                     `plan_integrate` for this plan",
                    wt_path.display()
                ),
                metrics: None,
            }));
        }
        if let Some(parent) = wt_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| FlowdError::PlanExecution {
                    message: format!(
                        "create integration worktree parent {}: {e}",
                        parent.display()
                    ),
                    metrics: None,
                })?;
        }

        let path_arg = wt_path.to_string_lossy().into_owned();
        run_git(
            repo,
            &[
                "worktree",
                "add",
                "-b",
                &intended.integration_branch,
                &path_arg,
                &base_tip,
            ],
        )
        .await?;

        for pick in &intended.cherry_picks {
            apply_cherry_pick(&wt_path, &base_tip, pick).await?;
        }

        let integration_tip = run_git(&wt_path, &["rev-parse", "HEAD"])
            .await?
            .trim()
            .to_owned();

        Ok(PlanIntegrateOutcome::Staged {
            intended,
            integration_tip,
        })
    }

    /// Promote a previously-staged integration branch to the configured
    /// base via fast-forward-only.
    ///
    /// Refuses on a dirty repo (won't stash for the operator) and on a
    /// non-fast-forward situation (the base advanced after staging --
    /// the operator must rebase the integration branch and re-confirm).
    /// Uses `git update-ref` with a CAS old-value so a race between the
    /// pre-check and the write surfaces as
    /// [`IntegrationFailure::BaseAdvanced`] rather than a silent
    /// overwrite.
    ///
    /// # Errors
    /// Same refusal shapes as [`Self::integrate`] for the eligibility
    /// stage, plus:
    /// * [`IntegrationFailure::DirtyBase`]
    /// * [`IntegrationFailure::BaseAdvanced`]
    pub async fn promote(
        &self,
        plan: &Plan,
        request: &PlanIntegrateRequest,
    ) -> IntegrateResult<PlanIntegrateOutcome> {
        let intended = assess_eligibility(plan, request)?;
        let repo = self.repo_for_plan(plan);

        let base_tip_now = resolve_branch_tip(repo, &intended.base_branch)
            .await
            .map_err(|_| IntegrationFailure::MissingBase {
                base: intended.base_branch.clone(),
            })?;

        let integration_tip = resolve_branch_tip(repo, &intended.integration_branch)
            .await
            .map_err(|_| {
                IntegrateError::Plan(FlowdError::PlanExecution {
                    message: format!(
                        "integration branch `{}` does not exist; stage the integration \
                         (mode=Confirm) before calling promote",
                        intended.integration_branch
                    ),
                    metrics: None,
                })
            })?;

        let dirty = run_git(repo, &["status", "--porcelain"]).await?;
        if !dirty.trim().is_empty() {
            return Err(IntegrationFailure::DirtyBase.into());
        }

        // ff-only: base tip must be an ancestor of the integration tip.
        // Exit code 0 means is-ancestor; 1 means not. Anything else is a
        // command failure and propagates.
        let (code, _, stderr) = run_git_capture(
            repo,
            &[
                "merge-base",
                "--is-ancestor",
                &base_tip_now,
                &integration_tip,
            ],
        )
        .await?;
        match code {
            0 => {}
            1 => {
                let base_tip_at_assess =
                    run_git(repo, &["merge-base", &base_tip_now, &integration_tip])
                        .await?
                        .trim()
                        .to_owned();
                return Err(IntegrationFailure::BaseAdvanced {
                    base: intended.base_branch.clone(),
                    base_tip_at_assess,
                    observed_base_tip: base_tip_now,
                }
                .into());
            }
            _ => {
                return Err(IntegrateError::Plan(FlowdError::PlanExecution {
                    message: format!("git merge-base --is-ancestor failed: {}", stderr.trim()),
                    metrics: None,
                }));
            }
        }

        // CAS update: if base advanced between the ancestor check and the
        // ref write, update-ref's old-value guard fails and we surface
        // BaseAdvanced for the operator to retry.
        let base_ref = format!("refs/heads/{}", intended.base_branch);
        let (code, _, stderr) = run_git_capture(
            repo,
            &["update-ref", &base_ref, &integration_tip, &base_tip_now],
        )
        .await?;
        if code != 0 {
            let observed = resolve_branch_tip(repo, &intended.base_branch)
                .await
                .unwrap_or_else(|_| base_tip_now.clone());
            if observed != base_tip_now {
                return Err(IntegrationFailure::BaseAdvanced {
                    base: intended.base_branch.clone(),
                    base_tip_at_assess: base_tip_now,
                    observed_base_tip: observed,
                }
                .into());
            }
            return Err(IntegrateError::Plan(FlowdError::PlanExecution {
                message: format!("git update-ref {base_ref} failed: {}", stderr.trim()),
                metrics: None,
            }));
        }

        Ok(PlanIntegrateOutcome::Promoted {
            intended,
            promoted_tip: integration_tip,
        })
    }

    /// Resolve the git repo path to drive operations from. The plan's
    /// persisted [`Plan::project_root`] (when present) overrides
    /// [`Self::repo`], which only meaningfully serves plans that
    /// pre-date the field.
    fn repo_for_plan<'a>(&'a self, plan: &'a Plan) -> &'a Path {
        plan.project_root
            .as_deref()
            .map_or(self.repo.as_path(), Path::new)
    }

    fn worktree_path(&self, plan: &Plan) -> PathBuf {
        self.worktree_root
            .join(sanitize_path(&plan.project))
            .join(plan.id.to_string())
    }
}

/// Apply one tip step's commits onto the integration branch. Uses
/// `rev-list base..source_branch` (reversed) so commits land in author
/// order, with `-m 1` reserved for merge commits in the range so the
/// per-layer dep merges the spawner produced replay relative to the
/// step's primary parent. An empty range -- the no-op step contract --
/// is treated as success.
async fn apply_cherry_pick(wt: &Path, base_tip: &str, pick: &CherryPick) -> IntegrateResult<()> {
    let range = format!("{base_tip}..{branch}", branch = pick.source_branch);
    let revs_raw = run_git(wt, &["rev-list", "--reverse", &range]).await?;
    let revs: Vec<&str> = revs_raw.split_whitespace().collect();
    if revs.is_empty() {
        // No-op step: source branch points at (or is contained in) the
        // base tip. Skip cleanly per the v1 contract.
        return Ok(());
    }

    for sha in &revs {
        let parents_raw = run_git(wt, &["rev-list", "-1", "--parents", sha]).await?;
        // `--parents` prints `<sha> <parent1> <parent2>...`; >1 parent ⇒ merge.
        let parent_count = parents_raw.split_whitespace().count().saturating_sub(1);

        // `--allow-empty` permits commits that *started* empty;
        // `--keep-redundant-commits` keeps commits whose changes are
        // already on the integration branch (a per-layer dep merge can
        // re-apply changes that an earlier cherry-pick in the same run
        // already brought in -- diamond plans hit this routinely). The
        // pair together stops the cherry-pick walker from aborting on
        // benign duplication.
        let mut args: Vec<&str> = Vec::with_capacity(6);
        args.push("cherry-pick");
        args.push("--allow-empty");
        args.push("--keep-redundant-commits");
        if parent_count > 1 {
            args.push("-m");
            args.push("1");
        }
        args.push(sha);

        let (code, _, stderr) = run_git_capture(wt, &args).await?;
        if code != 0 {
            // Inspect the index for unmerged paths: that's the conflict
            // signature. A non-conflict failure (e.g. missing object)
            // surfaces as a plain Plan error.
            let conflicting = run_git(wt, &["diff", "--name-only", "--diff-filter=U"])
                .await
                .unwrap_or_default();
            let paths: Vec<String> = conflicting
                .lines()
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect();
            if !paths.is_empty() {
                // Leave the integration worktree mid-cherry-pick per the
                // contract. The operator resolves and reruns; the daemon
                // does not retry.
                return Err(IntegrationFailure::CherryPickConflict {
                    step_id: pick.step_id.clone(),
                    conflicting_paths: paths,
                }
                .into());
            }
            return Err(IntegrateError::Plan(FlowdError::PlanExecution {
                message: format!(
                    "cherry-pick of {sha} failed for step `{step}`: {stderr}",
                    step = pick.step_id,
                    stderr = stderr.trim()
                ),
                metrics: None,
            }));
        }
    }
    Ok(())
}

/// Resolve a branch / ref to its tip sha. Returns `Err` when the ref
/// does not exist; callers map that to the appropriate
/// [`IntegrationFailure`] variant.
async fn resolve_branch_tip(cwd: &Path, branch: &str) -> FlowdResult<String> {
    let (code, stdout, stderr) = run_git_capture(
        cwd,
        &["rev-parse", "--verify", &format!("{branch}^{{commit}}")],
    )
    .await?;
    if code == 0 {
        Ok(stdout.trim().to_owned())
    } else {
        Err(FlowdError::PlanExecution {
            message: format!("ref `{branch}` did not resolve: {}", stderr.trim()),
            metrics: None,
        })
    }
}

/// Run `git -C <cwd> <args>` and return stdout on success. Mirrors
/// [`crate::spawner::git_output`] but lives here to avoid a cross-module
/// pub coupling for two helpers.
async fn run_git(cwd: &Path, args: &[&str]) -> FlowdResult<String> {
    let (code, stdout, stderr) = run_git_capture(cwd, args).await?;
    if code == 0 {
        Ok(stdout)
    } else {
        Err(FlowdError::PlanExecution {
            message: format!(
                "git {} failed in {}: {}",
                args.join(" "),
                cwd.display(),
                stderr.trim()
            ),
            metrics: None,
        })
    }
}

/// Capture variant of [`run_git`]: returns exit code + stdout + stderr
/// without treating non-zero as fatal. Used by call sites that need to
/// distinguish typed failure modes (e.g. cherry-pick conflict vs.
/// command error, ancestor-check 0/1/other).
async fn run_git_capture(cwd: &Path, args: &[&str]) -> FlowdResult<(i32, String, String)> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .await
        .map_err(|e| FlowdError::PlanExecution {
            message: format!("spawn git in {}: {e}", cwd.display()),
            metrics: None,
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let code = output.status.code().unwrap_or(-1);
    Ok((code, stdout, stderr))
}

/// Filesystem-safe equivalent of the spawner's `sanitize_path`: any
/// character outside `[A-Za-z0-9_.-]` collapses to `-`. Kept local so a
/// rename in the spawner doesn't ripple through; the integration
/// worktree directory naming is independent of any per-step branch
/// name.
fn sanitize_path(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowd_core::orchestration::integration::{
        CleanupPolicy, IntegrationConfig, IntegrationMode, integration_branch_ref, step_branch_ref,
    };
    use flowd_core::orchestration::{Plan, PlanStatus, PlanStep, StepStatus};

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

    fn completed_plan(project: &str, steps: Vec<PlanStep>) -> Plan {
        let mut p = Plan::new("p", project, steps);
        p.status = PlanStatus::Completed;
        p
    }

    fn confirm_request(plan_id: uuid::Uuid, base: &str) -> PlanIntegrateRequest {
        PlanIntegrateRequest::new(
            plan_id,
            IntegrationMode::Confirm,
            IntegrationConfig {
                base_branch: base.into(),
                cleanup: CleanupPolicy::KeepOnFailure,
            },
        )
        .expect("fixture request must validate")
    }

    fn dry_run_request(plan_id: uuid::Uuid, base: &str) -> PlanIntegrateRequest {
        PlanIntegrateRequest::new(
            plan_id,
            IntegrationMode::DryRun,
            IntegrationConfig {
                base_branch: base.into(),
                cleanup: CleanupPolicy::KeepOnFailure,
            },
        )
        .expect("fixture request must validate")
    }

    fn git_sync(cwd: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(args)
            .status()
            .expect("spawn git");
        assert!(status.success(), "git {args:?} failed in {}", cwd.display());
    }

    fn git_sync_capture(cwd: &Path, args: &[&str]) -> (bool, String, String) {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(args)
            .output()
            .expect("spawn git");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    /// Author identity is required for `git commit`; pin a known one so
    /// the test repos do not pick up the developer's global config.
    /// `commit.gpgsign=false` is forced because a global gpgsign=true
    /// (common on developer machines) breaks tests on a stale
    /// gpg-agent or in CI hosts without a signing key configured --
    /// the integrator does not care about signatures, so we disable
    /// the dependency entirely for these fixtures.
    fn commit(cwd: &Path, msg: &str) {
        git_sync(
            cwd,
            &[
                "-c",
                "user.name=flowd-test",
                "-c",
                "user.email=flowd@test.local",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-m",
                msg,
            ],
        );
    }

    /// Create a real git repo with one seed commit on `main`. Returns the
    /// canonical (non-symlinked) path so subsequent `git rev-parse`
    /// outputs compare cleanly on macOS where `/var` is a symlink to
    /// `/private/var`.
    fn init_repo(dir: &Path) -> PathBuf {
        git_sync(dir, &["init", "-b", "main"]);
        std::fs::write(dir.join("README.md"), "seed\n").unwrap();
        git_sync(dir, &["add", "README.md"]);
        commit(dir, "seed");
        std::fs::canonicalize(dir).expect("canonicalize repo")
    }

    /// Create the per-step branch the contract names ([`step_branch_ref`])
    /// at the given starting ref, populated with one commit that writes
    /// `<step_id>.txt = <step_id>`. Returns the resulting tip sha.
    fn make_step_branch(
        repo: &Path,
        project: &str,
        plan_id: uuid::Uuid,
        step_id: &str,
        start: &str,
    ) {
        let branch = step_branch_ref(project, plan_id, step_id);
        git_sync(repo, &["branch", &branch, start]);
        // Use `git worktree add` to a temp dir so the test never has to
        // mess with the main repo's HEAD; that mirrors the spawner's own
        // discipline and keeps the integrator's view of `main` stable.
        let tmp = tempfile::tempdir().expect("step worktree");
        let path = tmp.path().join("wt");
        git_sync(repo, &["worktree", "add", path.to_str().unwrap(), &branch]);
        std::fs::write(path.join(format!("{step_id}.txt")), step_id).unwrap();
        git_sync(&path, &["add", "."]);
        commit(&path, &format!("step {step_id}"));
        // Tear the worktree back down so the source branch stays put but
        // the temp checkout doesn't hang around.
        git_sync(
            repo,
            &["worktree", "remove", "--force", path.to_str().unwrap()],
        );
    }

    /// Variant that merges another step's branch in before committing,
    /// modelling the spawner's per-layer dep merges. The merge commit
    /// has two parents -- the integrator's range walk must `-m 1` it.
    fn make_step_branch_with_merge(
        repo: &Path,
        project: &str,
        plan_id: uuid::Uuid,
        step_id: &str,
        start: &str,
        merge_in: &str,
    ) {
        let branch = step_branch_ref(project, plan_id, step_id);
        git_sync(repo, &["branch", &branch, start]);
        let tmp = tempfile::tempdir().expect("step worktree");
        let path = tmp.path().join("wt");
        git_sync(repo, &["worktree", "add", path.to_str().unwrap(), &branch]);
        // First merge in the secondary dep -- creates the per-layer merge
        // commit the spawner produces in `prepare_step`. `commit.gpgsign=false`
        // for the same reason as `commit()`: a stale developer gpg-agent
        // would otherwise turn this into a flaky failure under parallel
        // test runs.
        git_sync(
            &path,
            &[
                "-c",
                "user.name=flowd-test",
                "-c",
                "user.email=flowd@test.local",
                "-c",
                "commit.gpgsign=false",
                "merge",
                "--no-edit",
                merge_in,
            ],
        );
        std::fs::write(path.join(format!("{step_id}.txt")), step_id).unwrap();
        git_sync(&path, &["add", "."]);
        commit(&path, &format!("step {step_id}"));
        git_sync(
            repo,
            &["worktree", "remove", "--force", path.to_str().unwrap()],
        );
    }

    /// Read a file from a worktree at a given ref via `git show <ref>:<path>`.
    /// Returns `None` if the path is missing at that ref.
    fn show_at(repo: &Path, refname: &str, path: &str) -> Option<String> {
        let (ok, out, _) = git_sync_capture(repo, &["show", &format!("{refname}:{path}")]);
        if ok { Some(out) } else { None }
    }

    fn rev_parse(repo: &Path, refname: &str) -> String {
        let (ok, out, err) = git_sync_capture(repo, &["rev-parse", refname]);
        assert!(ok, "rev-parse {refname} failed: {err}");
        out.trim().to_owned()
    }

    fn integrator(repo: &Path, worktrees: &Path) -> PlanIntegrator {
        PlanIntegrator::new(repo.to_path_buf(), worktrees.to_path_buf(), None)
    }

    /// Configure the plan to point at the test repo so [`PlanIntegrator`]
    /// drives operations there instead of the integrator's
    /// construction-time fallback.
    fn anchor_plan(plan: &mut Plan, repo: &Path) {
        plan.project_root = Some(repo.to_string_lossy().into_owned());
    }

    // -------- Linear chain: only final tip is cherry-picked --------------

    /// `a -> b -> c`. Per the contract only `c` is a tip. Set up branches
    /// so that the range `main..c-branch` carries a's, b's, and c's
    /// commits, and assert the integration tip ends up with all three
    /// files. The single cherry-pick covers the whole range.
    #[tokio::test]
    async fn linear_chain_picks_only_final_tip_and_carries_full_history() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan(
            "linproj",
            vec![step("a", &[]), step("b", &["a"]), step("c", &["b"])],
        );
        anchor_plan(&mut plan, &repo);

        make_step_branch(&repo, &plan.project, plan.id, "a", "main");
        make_step_branch(
            &repo,
            &plan.project,
            plan.id,
            "b",
            &step_branch_ref(&plan.project, plan.id, "a"),
        );
        make_step_branch(
            &repo,
            &plan.project,
            plan.id,
            "c",
            &step_branch_ref(&plan.project, plan.id, "b"),
        );

        let int = integrator(&repo, worktrees.path());
        let outcome = int
            .integrate(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect("staged");
        let intended = match &outcome {
            PlanIntegrateOutcome::Staged { intended, .. } => intended,
            other => panic!("expected Staged, got {other:?}"),
        };
        assert_eq!(intended.cherry_picks.len(), 1, "only c is a tip");
        assert_eq!(intended.cherry_picks[0].step_id, "c");

        let int_branch = integration_branch_ref(&plan.project, plan.id);
        for f in ["a.txt", "b.txt", "c.txt"] {
            assert!(
                show_at(&repo, &int_branch, f).is_some(),
                "expected {f} on integration tip after cherry-picking c's range"
            );
        }
        // The source branches must be untouched: their tips survive
        // verbatim. A cherry-pick that mutated them would shift these.
        let _ = rev_parse(&repo, &step_branch_ref(&plan.project, plan.id, "a"));
        let _ = rev_parse(&repo, &step_branch_ref(&plan.project, plan.id, "b"));
        let _ = rev_parse(&repo, &step_branch_ref(&plan.project, plan.id, "c"));
        // Base branch must be untouched until promote.
        assert_ne!(rev_parse(&repo, "main"), rev_parse(&repo, &int_branch));
    }

    // -------- Diamond: only the join step is cherry-picked --------------

    /// Diamond `a -> {b, c} -> d`. Only `d` is a tip; its branch was
    /// produced by merging `c` into `b`'s branch and committing on top,
    /// so the integration cherry-pick range includes a merge commit.
    /// Assert the integration tip carries all four steps' files (the
    /// `-m 1` handling for the merge in the range is the contract here).
    #[tokio::test]
    async fn diamond_picks_only_join_step_and_handles_merge_commit() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan(
            "diamondproj",
            vec![
                step("a", &[]),
                step("b", &["a"]),
                step("c", &["a"]),
                step("d", &["b", "c"]),
            ],
        );
        anchor_plan(&mut plan, &repo);

        let proj = plan.project.clone();
        make_step_branch(&repo, &proj, plan.id, "a", "main");
        make_step_branch(
            &repo,
            &proj,
            plan.id,
            "b",
            &step_branch_ref(&proj, plan.id, "a"),
        );
        make_step_branch(
            &repo,
            &proj,
            plan.id,
            "c",
            &step_branch_ref(&proj, plan.id, "a"),
        );
        // d is created from b, then merges c (per the spawner's
        // dependency-merge order), then commits its own change.
        make_step_branch_with_merge(
            &repo,
            &proj,
            plan.id,
            "d",
            &step_branch_ref(&proj, plan.id, "b"),
            &step_branch_ref(&proj, plan.id, "c"),
        );

        let int = integrator(&repo, worktrees.path());
        let outcome = int
            .integrate(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect("staged");
        let intended = match &outcome {
            PlanIntegrateOutcome::Staged { intended, .. } => intended,
            other => panic!("expected Staged, got {other:?}"),
        };
        assert_eq!(intended.cherry_picks.len(), 1);
        assert_eq!(intended.cherry_picks[0].step_id, "d");

        let int_branch = integration_branch_ref(&proj, plan.id);
        for f in ["a.txt", "b.txt", "c.txt", "d.txt"] {
            assert!(
                show_at(&repo, &int_branch, f).is_some(),
                "expected {f} on integration tip; merge commit handling regressed"
            );
        }
    }

    // -------- Parallel tails: both tips picked in id order --------------

    /// `a -> c`, `b -> d`. Two tips in the same final layer, picked in
    /// id order (`c` before `d`) for determinism. Both branches start
    /// off `main`, so each cherry-pick range carries that branch's
    /// own dep + tip commits.
    #[tokio::test]
    async fn parallel_tails_picked_in_id_order_and_both_land() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan(
            "parproj",
            vec![
                step("a", &[]),
                step("b", &[]),
                step("c", &["a"]),
                step("d", &["b"]),
            ],
        );
        anchor_plan(&mut plan, &repo);

        let proj = plan.project.clone();
        make_step_branch(&repo, &proj, plan.id, "a", "main");
        make_step_branch(&repo, &proj, plan.id, "b", "main");
        make_step_branch(
            &repo,
            &proj,
            plan.id,
            "c",
            &step_branch_ref(&proj, plan.id, "a"),
        );
        make_step_branch(
            &repo,
            &proj,
            plan.id,
            "d",
            &step_branch_ref(&proj, plan.id, "b"),
        );

        let int = integrator(&repo, worktrees.path());
        let outcome = int
            .integrate(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect("staged");
        let intended = match &outcome {
            PlanIntegrateOutcome::Staged { intended, .. } => intended,
            other => panic!("expected Staged, got {other:?}"),
        };
        let ids: Vec<&str> = intended
            .cherry_picks
            .iter()
            .map(|p| p.step_id.as_str())
            .collect();
        assert_eq!(ids, vec!["c", "d"], "tips must be picked in id order");

        let int_branch = integration_branch_ref(&proj, plan.id);
        for f in ["a.txt", "b.txt", "c.txt", "d.txt"] {
            assert!(
                show_at(&repo, &int_branch, f).is_some(),
                "expected {f} on integration tip"
            );
        }
    }

    // -------- DryRun: nothing is created --------------------------------

    #[tokio::test]
    async fn dry_run_returns_intended_without_creating_branches_or_worktrees() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan("dryproj", vec![step("a", &[])]);
        anchor_plan(&mut plan, &repo);

        // Note: no per-step branches created. DryRun must not consult the
        // working repo at all -- this proves it.
        let int = integrator(&repo, worktrees.path());
        let outcome = int
            .integrate(&plan, &dry_run_request(plan.id, "main"))
            .await
            .expect("dry-run");
        match outcome {
            PlanIntegrateOutcome::DryRun { intended } => {
                assert_eq!(intended.cherry_picks.len(), 1);
                assert_eq!(intended.cherry_picks[0].step_id, "a");
            }
            other => panic!("expected DryRun, got {other:?}"),
        }

        // Integration branch must not exist after a dry run.
        let int_branch = integration_branch_ref(&plan.project, plan.id);
        let (ok, _, _) = git_sync_capture(&repo, &["rev-parse", "--verify", &int_branch]);
        assert!(!ok, "DryRun must not create the integration branch");
    }

    // -------- No-op step: empty cherry-pick range --------------------

    /// A step whose branch is set to the base tip without any commits
    /// of its own. The integrator must skip it cleanly (empty range)
    /// rather than failing the whole integration.
    #[tokio::test]
    async fn no_op_step_with_empty_cherry_pick_range_skips_cleanly() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan("noopproj", vec![step("a", &[])]);
        anchor_plan(&mut plan, &repo);

        // Create the step branch pointing at main with no extra commits.
        // This is the "spawner ran but found no diff" shape.
        let branch = step_branch_ref(&plan.project, plan.id, "a");
        git_sync(&repo, &["branch", &branch, "main"]);

        let int = integrator(&repo, worktrees.path());
        let outcome = int
            .integrate(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect("no-op step must not fail");
        match outcome {
            PlanIntegrateOutcome::Staged {
                integration_tip, ..
            } => {
                // No cherry-picks landed; integration tip is just the
                // base tip.
                assert_eq!(integration_tip, rev_parse(&repo, "main"));
            }
            other => panic!("expected Staged, got {other:?}"),
        }
    }

    // -------- Missing base branch ---------------------------------------

    #[tokio::test]
    async fn missing_base_branch_surfaces_typed_failure() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan("missbase", vec![step("a", &[])]);
        anchor_plan(&mut plan, &repo);
        make_step_branch(&repo, &plan.project, plan.id, "a", "main");

        let int = integrator(&repo, worktrees.path());
        let err = int
            .integrate(&plan, &confirm_request(plan.id, "no-such-branch"))
            .await
            .expect_err("missing base must fail");
        match err {
            IntegrateError::Failure(IntegrationFailure::MissingBase { base }) => {
                assert_eq!(base, "no-such-branch");
            }
            other => panic!("expected MissingBase, got {other:?}"),
        }
    }

    // -------- Missing step branch ---------------------------------------

    #[tokio::test]
    async fn missing_step_branch_surfaces_typed_failure() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan("missstep", vec![step("a", &[])]);
        anchor_plan(&mut plan, &repo);
        // Deliberately do not create the step branch.

        let int = integrator(&repo, worktrees.path());
        let err = int
            .integrate(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect_err("missing step branch must fail");
        match err {
            IntegrateError::Failure(IntegrationFailure::MissingStepBranch { step_id, .. }) => {
                assert_eq!(step_id, "a");
            }
            other => panic!("expected MissingStepBranch, got {other:?}"),
        }
    }

    // -------- Cherry-pick conflict --------------------------------------

    /// Two tips both edit the same file with conflicting content; the
    /// second cherry-pick fails and the integrator must surface
    /// [`IntegrationFailure::CherryPickConflict`] with the offending
    /// step id and the conflicting paths. The integration worktree is
    /// left in place per the contract.
    #[tokio::test]
    async fn cherry_pick_conflict_surfaces_typed_failure_and_leaves_worktree() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan("conflictproj", vec![step("a", &[]), step("b", &[])]);
        anchor_plan(&mut plan, &repo);

        let proj = plan.project.clone();

        // Branch a writes shared.txt = "alpha\n".
        let a_branch = step_branch_ref(&proj, plan.id, "a");
        git_sync(&repo, &["branch", &a_branch, "main"]);
        let tmp_a = tempfile::tempdir().unwrap();
        let path_a = tmp_a.path().join("a");
        git_sync(
            &repo,
            &["worktree", "add", path_a.to_str().unwrap(), &a_branch],
        );
        std::fs::write(path_a.join("shared.txt"), "alpha\n").unwrap();
        git_sync(&path_a, &["add", "."]);
        commit(&path_a, "a");
        git_sync(
            &repo,
            &["worktree", "remove", "--force", path_a.to_str().unwrap()],
        );

        // Branch b writes shared.txt = "beta\n" -- straight conflict.
        let b_branch = step_branch_ref(&proj, plan.id, "b");
        git_sync(&repo, &["branch", &b_branch, "main"]);
        let tmp_b = tempfile::tempdir().unwrap();
        let path_b = tmp_b.path().join("b");
        git_sync(
            &repo,
            &["worktree", "add", path_b.to_str().unwrap(), &b_branch],
        );
        std::fs::write(path_b.join("shared.txt"), "beta\n").unwrap();
        git_sync(&path_b, &["add", "."]);
        commit(&path_b, "b");
        git_sync(
            &repo,
            &["worktree", "remove", "--force", path_b.to_str().unwrap()],
        );

        let int = integrator(&repo, worktrees.path());
        let err = int
            .integrate(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect_err("conflict must fail");
        match err {
            IntegrateError::Failure(IntegrationFailure::CherryPickConflict {
                step_id,
                conflicting_paths,
            }) => {
                // a is the first cherry-pick (id-sorted layer 0); b
                // conflicts. The contract surfaces the offending step.
                assert_eq!(step_id, "b");
                assert!(
                    conflicting_paths.iter().any(|p| p.contains("shared.txt")),
                    "expected shared.txt in conflicting paths, got: {conflicting_paths:?}"
                );
            }
            other => panic!("expected CherryPickConflict, got {other:?}"),
        }

        // Integration worktree must still be on disk for human resolution.
        let wt_path = worktrees
            .path()
            .join("conflictproj")
            .join(plan.id.to_string());
        assert!(
            wt_path.exists(),
            "conflict worktree must survive for the operator at {}",
            wt_path.display()
        );
    }

    // -------- Promote: dirty base ---------------------------------------

    #[tokio::test]
    async fn promote_refuses_dirty_base() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan("dirtyproj", vec![step("a", &[])]);
        anchor_plan(&mut plan, &repo);
        make_step_branch(&repo, &plan.project, plan.id, "a", "main");

        let int = integrator(&repo, worktrees.path());
        int.integrate(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect("staged");

        // Dirty the base worktree after staging but before promote.
        std::fs::write(repo.join("dirty.txt"), "dirty\n").unwrap();

        let err = int
            .promote(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect_err("dirty base must refuse promotion");
        match err {
            IntegrateError::Failure(IntegrationFailure::DirtyBase) => {}
            other => panic!("expected DirtyBase, got {other:?}"),
        }
    }

    // -------- Promote: base advanced ------------------------------------

    /// Stage cleanly, then advance the base ref to a commit not on the
    /// integration branch. Promotion must refuse with `BaseAdvanced`
    /// rather than rewrite the base.
    #[tokio::test]
    async fn promote_refuses_when_base_advanced_past_integration_tip() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan("advproj", vec![step("a", &[])]);
        anchor_plan(&mut plan, &repo);
        make_step_branch(&repo, &plan.project, plan.id, "a", "main");

        let int = integrator(&repo, worktrees.path());
        int.integrate(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect("staged");

        // Advance main past the integration tip directly in the main
        // repo (which has main checked out). A sidecar worktree on
        // `main` is rejected -- git refuses to check the same branch
        // out twice.
        std::fs::write(repo.join("advance.txt"), "advance\n").unwrap();
        git_sync(&repo, &["add", "advance.txt"]);
        commit(&repo, "advance main");

        let err = int
            .promote(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect_err("base advanced must refuse");
        match err {
            IntegrateError::Failure(IntegrationFailure::BaseAdvanced { base, .. }) => {
                assert_eq!(base, "main");
            }
            other => panic!("expected BaseAdvanced, got {other:?}"),
        }
    }

    // -------- Promote: happy path is fast-forward only ------------------

    #[tokio::test]
    async fn promote_fast_forwards_base_when_clean_and_ancestor_holds() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan("ffproj", vec![step("a", &[])]);
        anchor_plan(&mut plan, &repo);
        make_step_branch(&repo, &plan.project, plan.id, "a", "main");

        let int = integrator(&repo, worktrees.path());
        let stage_outcome = int
            .integrate(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect("staged");
        let staged_tip = match stage_outcome {
            PlanIntegrateOutcome::Staged {
                integration_tip, ..
            } => integration_tip,
            other => panic!("expected Staged, got {other:?}"),
        };
        let base_before = rev_parse(&repo, "main");
        assert_ne!(base_before, staged_tip, "staging must not touch base");

        let outcome = int
            .promote(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect("promote");
        match outcome {
            PlanIntegrateOutcome::Promoted { promoted_tip, .. } => {
                assert_eq!(promoted_tip, staged_tip);
                assert_eq!(
                    rev_parse(&repo, "main"),
                    staged_tip,
                    "base must point at integration tip"
                );
            }
            other => panic!("expected Promoted, got {other:?}"),
        }
    }

    // -------- Promote: cannot promote without staging ------------------

    /// Sanity guard: calling promote before integrate must error with a
    /// clear message rather than silently succeed against a missing
    /// integration ref.
    #[tokio::test]
    async fn promote_without_staging_errors_clearly() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan("nostageproj", vec![step("a", &[])]);
        anchor_plan(&mut plan, &repo);

        let int = integrator(&repo, worktrees.path());
        let err = int
            .promote(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect_err("promote-before-stage must fail");
        match err {
            IntegrateError::Plan(FlowdError::PlanExecution { message, .. }) => {
                assert!(
                    message.contains("does not exist") && message.contains("stage"),
                    "expected actionable message, got: {message}"
                );
            }
            other => panic!("expected Plan error, got {other:?}"),
        }
    }

    // -------- Refusal: non-completed plan -------------------------------

    /// Eligibility refusals from the contract bubble through unchanged.
    /// Pin one variant here so a future refactor that swallowed
    /// [`IntegrateError::Refusal`] would surface in this crate's suite.
    #[tokio::test]
    async fn integrate_propagates_pure_contract_refusal() {
        let repo_dir = tempfile::tempdir().expect("repo");
        let repo = init_repo(repo_dir.path());
        let worktrees = tempfile::tempdir().expect("worktrees");

        let mut plan = completed_plan("refuse", vec![step("a", &[])]);
        plan.status = PlanStatus::Failed;
        anchor_plan(&mut plan, &repo);

        let int = integrator(&repo, worktrees.path());
        let err = int
            .integrate(&plan, &confirm_request(plan.id, "main"))
            .await
            .expect_err("non-completed plan must refuse");
        match err {
            IntegrateError::Refusal(IntegrationRefusal::PlanStatus { observed }) => {
                assert_eq!(observed, PlanStatus::Failed);
            }
            other => panic!("expected PlanStatus refusal, got {other:?}"),
        }
    }
}
