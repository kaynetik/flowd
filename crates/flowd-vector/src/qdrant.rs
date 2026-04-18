//! Qdrant `VectorIndex` implementation.
//!
//! Connects to Qdrant (local or remote) and manages a single collection
//! with HNSW-tuned cosine similarity for observation embeddings.
//! Payload filtering on `project` enables scoped ANN search.

use flowd_core::error::{FlowdError, Result};
use flowd_core::memory::VectorIndex;
use flowd_core::types::Embedding;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter,
    HnswConfigDiffBuilder, PointStruct, PointsIdsList, SearchPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde_json::json;
use uuid::Uuid;

const COLLECTION_NAME: &str = "flowd_observations";

pub struct QdrantIndex {
    client: Qdrant,
    dimensions: usize,
}

/// Configuration for the Qdrant connection and HNSW tuning.
pub struct QdrantConfig {
    pub url: String,
    pub dimensions: usize,
    pub hnsw_m: u64,
    pub hnsw_ef_construct: u64,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_owned(),
            dimensions: 384,
            hnsw_m: 16,
            hnsw_ef_construct: 100,
        }
    }
}

impl QdrantIndex {
    /// Connect to Qdrant and ensure the observations collection exists.
    ///
    /// # Errors
    /// Returns `FlowdError::Vector` if the client or collection setup fails.
    pub async fn open(config: &QdrantConfig) -> Result<Self> {
        let client = Qdrant::from_url(&config.url)
            .build()
            .map_err(|e| FlowdError::Vector(e.to_string()))?;

        let exists = client
            .collection_exists(COLLECTION_NAME)
            .await
            .map_err(|e| FlowdError::Vector(e.to_string()))?;

        if !exists {
            client
                .create_collection(
                    CreateCollectionBuilder::new(COLLECTION_NAME)
                        .vectors_config(VectorParamsBuilder::new(
                            config.dimensions as u64,
                            Distance::Cosine,
                        ))
                        .hnsw_config(
                            HnswConfigDiffBuilder::default()
                                .m(config.hnsw_m)
                                .ef_construct(config.hnsw_ef_construct),
                        ),
                )
                .await
                .map_err(|e| FlowdError::Vector(e.to_string()))?;

            tracing::info!(
                collection = COLLECTION_NAME,
                dimensions = config.dimensions,
                hnsw_m = config.hnsw_m,
                hnsw_ef = config.hnsw_ef_construct,
                "created qdrant collection"
            );
        }

        Ok(Self {
            client,
            dimensions: config.dimensions,
        })
    }

    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl VectorIndex for QdrantIndex {
    async fn upsert(&self, embedding: &Embedding) -> Result<()> {
        let point_id: String = embedding.observation_id.to_string();

        let payload = Payload::try_from(json!({
            "observation_id": embedding.observation_id.to_string(),
            "project": embedding.project,
        }))
        .map_err(|e| FlowdError::Vector(e.to_string()))?;

        let point = PointStruct::new(point_id, embedding.vector.clone(), payload);

        self.client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, vec![point]).wait(true))
            .await
            .map_err(|e| FlowdError::Vector(e.to_string()))?;

        Ok(())
    }

    async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        project_filter: Option<&str>,
    ) -> Result<Vec<(Uuid, f64)>> {
        let mut builder =
            SearchPointsBuilder::new(COLLECTION_NAME, query_vector.to_vec(), limit as u64)
                .with_payload(true);

        if let Some(project) = project_filter {
            builder = builder.filter(Filter::all([Condition::matches(
                "project",
                project.to_owned(),
            )]));
        }

        let results = self
            .client
            .search_points(builder)
            .await
            .map_err(|e| FlowdError::Vector(e.to_string()))?;

        let mut out = Vec::with_capacity(results.result.len());
        for point in results.result {
            let obs_id_str = point
                .payload
                .get("observation_id")
                .and_then(|v| v.as_str().map(std::borrow::ToOwned::to_owned))
                .unwrap_or_default();

            if let Ok(obs_id) = Uuid::parse_str(&obs_id_str) {
                out.push((obs_id, f64::from(point.score)));
            }
        }

        Ok(out)
    }

    async fn delete(&self, observation_id: Uuid) -> Result<()> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(COLLECTION_NAME).points(PointsIdsList {
                    ids: vec![observation_id.to_string().into()],
                }),
            )
            .await
            .map_err(|e| FlowdError::Vector(e.to_string()))?;

        Ok(())
    }
}
