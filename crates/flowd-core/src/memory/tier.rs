//! Tiering policy for the memory subsystem.
//!
//! Maps observation age to `MemoryTier`. Policies are immutable value objects;
//! construct one with validated thresholds and consult `classify` to decide
//! where a given observation should live.

use crate::error::{FlowdError, Result};
use crate::types::MemoryTier;
use chrono::Duration;

/// Age thresholds that drive tier transitions.
///
/// Invariants (enforced by the constructor):
/// - `hot_max_age < warm_max_age`
/// - both durations are strictly positive
#[derive(Debug, Clone, Copy)]
pub struct TieringPolicy {
    hot_max_age: Duration,
    warm_max_age: Duration,
}

impl TieringPolicy {
    /// Build a policy with explicit thresholds.
    ///
    /// # Errors
    /// Returns `FlowdError::Internal` if `hot_max_age >= warm_max_age` or any
    /// threshold is non-positive.
    pub fn new(hot_max_age: Duration, warm_max_age: Duration) -> Result<Self> {
        if hot_max_age <= Duration::zero() || warm_max_age <= Duration::zero() {
            return Err(FlowdError::Internal(
                "tiering thresholds must be positive".into(),
            ));
        }
        if hot_max_age >= warm_max_age {
            return Err(FlowdError::Internal(
                "hot_max_age must be less than warm_max_age".into(),
            ));
        }
        Ok(Self {
            hot_max_age,
            warm_max_age,
        })
    }

    /// Reasonable defaults for a long-running daemon:
    /// - Hot: 24 hours
    /// - Warm: 30 days
    #[must_use]
    pub fn standard() -> Self {
        Self {
            hot_max_age: Duration::hours(24),
            warm_max_age: Duration::days(30),
        }
    }

    #[must_use]
    pub fn hot_max_age(&self) -> Duration {
        self.hot_max_age
    }

    #[must_use]
    pub fn warm_max_age(&self) -> Duration {
        self.warm_max_age
    }

    /// Return the tier an observation of the given `age` belongs in.
    #[must_use]
    pub fn classify(&self, age: Duration) -> MemoryTier {
        if age <= self.hot_max_age {
            MemoryTier::Hot
        } else if age <= self.warm_max_age {
            MemoryTier::Warm
        } else {
            MemoryTier::Cold
        }
    }
}

impl Default for TieringPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_respects_boundaries() {
        let p = TieringPolicy::standard();
        assert_eq!(p.classify(Duration::minutes(5)), MemoryTier::Hot);
        assert_eq!(p.classify(Duration::hours(24)), MemoryTier::Hot);
        assert_eq!(p.classify(Duration::hours(25)), MemoryTier::Warm);
        assert_eq!(p.classify(Duration::days(30)), MemoryTier::Warm);
        assert_eq!(p.classify(Duration::days(31)), MemoryTier::Cold);
    }

    #[test]
    fn rejects_inverted_thresholds() {
        let err = TieringPolicy::new(Duration::days(30), Duration::days(1));
        assert!(err.is_err());
    }

    #[test]
    fn rejects_non_positive() {
        assert!(TieringPolicy::new(Duration::zero(), Duration::days(1)).is_err());
        assert!(TieringPolicy::new(Duration::days(1), Duration::zero()).is_err());
    }
}
