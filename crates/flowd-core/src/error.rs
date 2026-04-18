use thiserror::Error;

#[derive(Debug, Error)]
pub enum FlowdError {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("vector index error: {0}")]
    Vector(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("rule violation ({level}): {description}")]
    RuleViolation {
        rule_id: String,
        level: RuleLevel,
        description: String,
    },

    #[error("rule load error: {0}")]
    RuleLoad(String),

    #[error("orchestration error: {0}")]
    Orchestration(String),

    #[error("plan validation failed: {0}")]
    PlanValidation(String),

    #[error("plan execution error: {0}")]
    PlanExecution(String),

    #[error("plan not found: {0}")]
    PlanNotFound(uuid::Uuid),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("{0}")]
    Internal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuleLevel {
    Warn,
    Deny,
}

impl std::fmt::Display for RuleLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Warn => write!(f, "warn"),
            Self::Deny => write!(f, "deny"),
        }
    }
}

pub type Result<T> = std::result::Result<T, FlowdError>;
