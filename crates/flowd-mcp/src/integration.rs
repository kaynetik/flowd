//! Integration-driver abstraction for the `plan_integrate` MCP tool.
//!
//! The MCP layer never touches git directly: it borrows an
//! [`IntegrationDriver`] supplied at composition time. The daemon binds
//! this to `flowd_cli::integration::PlanIntegrator`; tests bind a stub.
//!
//! Splitting the driver out of `flowd-mcp` keeps the crate dependency
//! graph clean (`flowd-mcp` does not pull in `flowd-storage` or shell out
//! to git) while still letting the handler return the structured
//! [`PlanIntegrateOutcome`] / [`IntegrationFailure`] / [`IntegrationRefusal`]
//! shapes pinned in `flowd-core`.

use std::future::Future;
use std::pin::Pin;

use flowd_core::error::FlowdError;
use flowd_core::orchestration::Plan;
use flowd_core::orchestration::integration::{
    IntegrationFailure, IntegrationRefusal, PlanIntegrateOutcome, PlanIntegrateRequest,
};

/// Combined error surface for an [`IntegrationDriver`] call.
///
/// Pinned here (not re-imported from `flowd-cli`) because the MCP layer
/// must not depend on the CLI crate; the variants mirror
/// `flowd_cli::integration::IntegrateError` so the daemon's
/// `IntegrationDriver` impl can map across without information loss.
#[derive(Debug, thiserror::Error)]
pub enum IntegrationError {
    /// Pre-flight refusal returned from the pure contract before any git
    /// command runs.
    #[error(transparent)]
    Refusal(#[from] IntegrationRefusal),
    /// Runtime failure observed while interacting with the working repo.
    #[error(transparent)]
    Failure(#[from] IntegrationFailure),
    /// Anything else: failed git invocation, filesystem error, etc.
    #[error(transparent)]
    Plan(#[from] FlowdError),
}

/// Boxed async result for the dyn-compatible driver methods.
pub type IntegrationFuture<'a> =
    Pin<Box<dyn Future<Output = Result<PlanIntegrateOutcome, IntegrationError>> + Send + 'a>>;

/// Surface the MCP `plan_integrate` handler dispatches to.
///
/// Mirrors the two-phase CLI surface: stage (`integrate`) then
/// fast-forward (`promote`). Every implementation is responsible for its
/// own persistence -- the handler only routes the call and returns the
/// outcome verbatim.
pub trait IntegrationDriver: Send + Sync {
    fn integrate<'a>(
        &'a self,
        plan: &'a Plan,
        request: &'a PlanIntegrateRequest,
    ) -> IntegrationFuture<'a>;

    fn promote<'a>(
        &'a self,
        plan: &'a Plan,
        request: &'a PlanIntegrateRequest,
    ) -> IntegrationFuture<'a>;
}
