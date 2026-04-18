//! MCP tool handlers.
//!
//! [`McpHandlers`] is the application-side interface the server dispatches to.
//! Concrete implementation [`FlowdHandlers`] wires the trait to the three
//! flowd-core services -- `MemoryService`, a `PlanExecutor`, and a
//! `RuleEvaluator`. Tests can supply any other implementation (see the
//! integration test for a stub version).
//!
//! All handlers are `async` and return `serde_json::Value` payloads. The
//! server wraps these values into MCP tool-result envelopes; transport
//! details never leak into the handler layer.

use std::future::Future;
use std::str::FromStr;
use std::sync::Arc;

use serde_json::{Value, json};
use uuid::Uuid;

use flowd_core::error::{FlowdError, Result};
use flowd_core::memory::context::AutoContextQuery;
use flowd_core::memory::service::MemoryService;
use flowd_core::memory::{EmbeddingProvider, MemoryBackend, VectorIndex};
use flowd_core::orchestration::{PlanExecutor, PlanStatus, loader::PlanDefinition};
use flowd_core::rules::{ProposedAction, RuleEvaluator};
use flowd_core::types::SearchQuery;

use crate::tools::{
    MemoryContextParams, MemorySearchParams, MemoryStoreParams, PlanConfirmParams,
    PlanCreateParams, PlanStatusParams, RulesCheckParams, RulesListParams,
};

/// Surface exposed by the MCP server. Return values are arbitrary JSON;
/// errors bubble up via `Result` and are mapped to `isError: true` by the
/// dispatcher in [`crate::server`].
pub trait McpHandlers: Send + Sync {
    fn memory_store(&self, params: MemoryStoreParams)
    -> impl Future<Output = Result<Value>> + Send;

    fn memory_search(
        &self,
        params: MemorySearchParams,
    ) -> impl Future<Output = Result<Value>> + Send;

    fn memory_context(
        &self,
        params: MemoryContextParams,
    ) -> impl Future<Output = Result<Value>> + Send;

    fn plan_create(&self, params: PlanCreateParams) -> impl Future<Output = Result<Value>> + Send;

    fn plan_confirm(&self, params: PlanConfirmParams)
    -> impl Future<Output = Result<Value>> + Send;

    fn plan_status(&self, params: PlanStatusParams) -> impl Future<Output = Result<Value>> + Send;

    fn rules_check(&self, params: RulesCheckParams) -> impl Future<Output = Result<Value>> + Send;

    fn rules_list(&self, params: RulesListParams) -> impl Future<Output = Result<Value>> + Send;
}

/// Concrete handlers that compose the three flowd-core services.
///
/// Generic over the backend types so the concrete `SQLite` / `Qdrant` / ONNX
/// implementations (or test doubles) are chosen at composition time --
/// `flowd-mcp` itself never pulls in those heavy dependencies.
pub struct FlowdHandlers<M, V, E, PE, R>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
    PE: PlanExecutor + 'static,
    R: RuleEvaluator + 'static,
{
    memory: Arc<MemoryService<M, V, E>>,
    executor: Arc<PE>,
    rules: Arc<R>,
}

impl<M, V, E, PE, R> FlowdHandlers<M, V, E, PE, R>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
    PE: PlanExecutor + 'static,
    R: RuleEvaluator + 'static,
{
    pub fn new(memory: Arc<MemoryService<M, V, E>>, executor: Arc<PE>, rules: Arc<R>) -> Self {
        Self {
            memory,
            executor,
            rules,
        }
    }

    #[must_use]
    pub fn memory(&self) -> &Arc<MemoryService<M, V, E>> {
        &self.memory
    }

    #[must_use]
    pub fn executor(&self) -> &Arc<PE> {
        &self.executor
    }

    #[must_use]
    pub fn rules(&self) -> &Arc<R> {
        &self.rules
    }
}

fn parse_uuid(raw: &str) -> Result<Uuid> {
    Uuid::from_str(raw).map_err(|e| FlowdError::Internal(format!("invalid uuid `{raw}`: {e}")))
}

impl<M, V, E, PE, R> McpHandlers for FlowdHandlers<M, V, E, PE, R>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
    PE: PlanExecutor + 'static,
    R: RuleEvaluator + 'static,
{
    async fn memory_store(&self, p: MemoryStoreParams) -> Result<Value> {
        let session = parse_uuid(&p.session_id)?;
        // MCP clients hold the session UUID (Claude/Cursor assign it at
        // session start) but do not issue a separate start_session call.
        // Ensure the row exists before inserting an observation so the FK
        // on observations.session_id never trips.
        self.memory.ensure_session(&p.project, session).await?;
        let id = self
            .memory
            .record(&p.project, session, p.content, p.metadata)
            .await?;
        Ok(json!({ "id": id.to_string() }))
    }

    async fn memory_search(&self, p: MemorySearchParams) -> Result<Value> {
        let query = SearchQuery {
            text: p.query,
            project: p.project,
            since: None,
            limit: p.limit.unwrap_or(10),
        };
        let results = self.memory.search(&query).await?;
        serde_json::to_value(&results).map_err(|e| FlowdError::Serialization(e.to_string()))
    }

    async fn memory_context(&self, p: MemoryContextParams) -> Result<Value> {
        let mut ctx = AutoContextQuery::new(&p.project);
        if let Some(f) = p.file_path.as_deref() {
            ctx = ctx.with_file(f);
        }
        if let Some(s) = p.session_id.as_deref() {
            ctx = ctx.with_session(parse_uuid(s)?);
        }
        if let Some(h) = p.hint {
            ctx = ctx.with_hint(h);
        }
        if let Some(lim) = p.limit {
            ctx = ctx.with_limit(lim);
        }

        let observations = self.memory.auto_context(&ctx).await?;
        let rules = self
            .rules
            .matching_rules(Some(&p.project), p.file_path.as_deref());

        // Rules are cloned rather than borrowed-serialised because the
        // caller may outlive the lock held by `matching_rules`.
        let rules: Vec<_> = rules.into_iter().cloned().collect();

        Ok(json!({
            "observations": observations,
            "rules": rules,
        }))
    }

    async fn plan_create(&self, p: PlanCreateParams) -> Result<Value> {
        let def: PlanDefinition = serde_json::from_value(p.definition)
            .map_err(|e| FlowdError::PlanValidation(format!("invalid PlanDefinition: {e}")))?;
        let plan = def.into_plan();
        let preview = self.executor.preview(&plan)?;
        let plan_id = self.executor.submit(plan).await?;
        Ok(json!({
            "plan_id": plan_id.to_string(),
            "preview": preview,
        }))
    }

    async fn plan_confirm(&self, p: PlanConfirmParams) -> Result<Value> {
        let id = parse_uuid(&p.plan_id)?;
        let preview = self.executor.confirm(id).await?;
        // Kick off execution in the background; plan_status is the polling
        // entry point. We intentionally do not propagate errors from the
        // detached task -- they land in the plan's recorded state instead.
        let exec = Arc::clone(&self.executor);
        tokio::spawn(async move {
            if let Err(e) = exec.execute(id).await {
                tracing::warn!(plan_id = %id, error = %e, "plan execution failed");
            }
        });
        Ok(json!({
            "plan_id": id.to_string(),
            "status": "running",
            "preview": preview,
        }))
    }

    async fn plan_status(&self, p: PlanStatusParams) -> Result<Value> {
        let id = parse_uuid(&p.plan_id)?;
        let plan = self.executor.status(id).await?;
        serde_json::to_value(&plan).map_err(|e| FlowdError::Serialization(e.to_string()))
    }

    async fn rules_check(&self, p: RulesCheckParams) -> Result<Value> {
        let mut action = ProposedAction::new(p.tool);
        if let Some(f) = p.file_path {
            action = action.with_file(f);
        }
        if let Some(project) = p.project {
            action = action.with_project(project);
        }
        let result = self.rules.check(&action);
        serde_json::to_value(&result).map_err(|e| FlowdError::Serialization(e.to_string()))
    }

    async fn rules_list(&self, p: RulesListParams) -> Result<Value> {
        let matches = self
            .rules
            .matching_rules(p.project.as_deref(), p.file_path.as_deref());
        let matches: Vec<_> = matches.into_iter().cloned().collect();
        Ok(json!({ "rules": matches }))
    }
}

/// Mark [`PlanStatus`] as MCP-visible; the JSON value is what callers see.
#[must_use]
pub fn plan_status_label(status: PlanStatus) -> &'static str {
    match status {
        PlanStatus::Draft => "draft",
        PlanStatus::Confirmed => "confirmed",
        PlanStatus::Running => "running",
        PlanStatus::Completed => "completed",
        PlanStatus::Failed => "failed",
        PlanStatus::Cancelled => "cancelled",
    }
}
