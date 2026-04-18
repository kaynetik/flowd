//! ONNX Runtime embedding provider for flowd.
//!
//! Implements `EmbeddingProvider` from `flowd-core` using the `ort` crate
//! with a bundled all-MiniLM-L6-v2 model for local embedding generation.

pub mod provider;
