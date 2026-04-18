//! ONNX `EmbeddingProvider` implementation.
//!
//! Loads an all-MiniLM-L6-v2 ONNX model alongside its `HuggingFace` tokenizer
//! and runs inference on the rayon thread pool for CPU-parallel batch embedding.
//! Output vectors are L2-normalized after mean pooling.

use flowd_core::error::{FlowdError, Result};
use flowd_core::memory::EmbeddingProvider;
use ndarray::Array2;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

fn embed_err(e: impl std::fmt::Display) -> FlowdError {
    FlowdError::Embedding(e.to_string())
}

pub struct OnnxEmbedder {
    session: Mutex<ort::session::Session>,
    tokenizer: Tokenizer,
    dimensions: usize,
}

impl OnnxEmbedder {
    /// Load an ONNX model and its tokenizer from disk.
    ///
    /// `model_dir` must contain:
    /// - `model.onnx` (the sentence-transformer ONNX export)
    /// - `tokenizer.json` (`HuggingFace` fast tokenizer config)
    ///
    /// For all-MiniLM-L6-v2, output dimensionality is 384.
    ///
    /// # Errors
    /// Returns `FlowdError::Embedding` if the ONNX session or tokenizer cannot be loaded.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = ort::session::Session::builder()
            .map_err(embed_err)?
            .with_intra_threads(1)
            .map_err(embed_err)?
            .commit_from_file(&model_path)
            .map_err(embed_err)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| FlowdError::Embedding(format!("tokenizer load failed: {e}")))?;

        let dimensions = 384;

        tracing::info!(
            model = %model_path.display(),
            dimensions,
            "loaded ONNX embedding model"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dimensions,
        })
    }

    /// Tokenize, run inference, mean-pool, and L2-normalize a single text.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn embed_inner(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| FlowdError::Embedding(format!("tokenization failed: {e}")))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        let seq_len = ids.len();

        let input_ids =
            Array2::from_shape_vec((1, seq_len), ids.iter().map(|&v| i64::from(v)).collect())
                .map_err(embed_err)?;

        let attention_mask =
            Array2::from_shape_vec((1, seq_len), mask.iter().map(|&v| i64::from(v)).collect())
                .map_err(embed_err)?;

        let token_type_ids = Array2::from_shape_vec(
            (1, seq_len),
            type_ids.iter().map(|&v| i64::from(v)).collect(),
        )
        .map_err(embed_err)?;

        let ids_val = Tensor::from_array(input_ids).map_err(embed_err)?;
        let mask_val = Tensor::from_array(attention_mask).map_err(embed_err)?;
        let type_val = Tensor::from_array(token_type_ids).map_err(embed_err)?;

        let mut session = self.session.lock().map_err(embed_err)?;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_val,
                "attention_mask" => mask_val,
                "token_type_ids" => type_val,
            ])
            .map_err(embed_err)?;

        // last_hidden_state: shape [1, seq_len, hidden_dim]
        let (shape, hidden_data) = outputs[0].try_extract_tensor::<f32>().map_err(embed_err)?;

        if shape.len() != 3 || shape[0] != 1 {
            return Err(FlowdError::Embedding(format!(
                "unexpected output shape: {shape:?}, expected [1, seq_len, {dim}]",
                dim = self.dimensions
            )));
        }

        let hidden_dim = shape[2] as usize;

        // Mean pooling: attention-mask-weighted average over the sequence dimension.
        // hidden_data is row-major [1, seq_len, hidden_dim], so offset = t * hidden_dim + d
        let mut pooled = vec![0.0f32; hidden_dim];
        let mut mask_sum = 0.0f32;

        #[allow(clippy::cast_precision_loss)]
        for (t, &m_val) in mask.iter().enumerate().take(seq_len) {
            let m = m_val as f32;
            mask_sum += m;
            let offset = t * hidden_dim;
            for d in 0..hidden_dim {
                pooled[d] += hidden_data[offset + d] * m;
            }
        }

        let mask_sum = mask_sum.max(1e-9);
        for v in &mut pooled {
            *v /= mask_sum;
        }

        // L2 normalize
        let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in &mut pooled {
            *v /= norm;
        }

        Ok(pooled)
    }
}

impl EmbeddingProvider for OnnxEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_inner(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        use rayon::prelude::*;

        texts
            .par_iter()
            .map(|text| self.embed_inner(text))
            .collect()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
