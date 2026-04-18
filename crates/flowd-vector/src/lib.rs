//! Qdrant embedded vector index for flowd.
//!
//! Implements `VectorIndex` from `flowd-core` using Qdrant's embedded mode
//! for in-process HNSW-based approximate nearest neighbor search.

pub mod qdrant;
