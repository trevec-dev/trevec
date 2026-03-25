use std::path::PathBuf;

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

/// Embedding dimension for BGESmallENV15 (384-dim).
pub const EMBEDDING_DIM: usize = 384;

/// Default batch size for CPU inference.
const BATCH_SIZE_CPU: usize = 256;
/// Larger batch size for GPU inference — short code texts leave headroom
/// for bigger batches that better amortize CUDA kernel launch overhead.
const BATCH_SIZE_GPU: usize = 512;

/// Wrapper around fastembed for generating local embeddings.
pub struct Embedder {
    model: TextEmbedding,
    batch_size: usize,
}

impl Embedder {
    /// Create a new embedder using the default model (BGESmallENV15).
    pub fn new() -> Result<Self> {
        Self::new_with_model(None, true, None, None)
    }

    /// Create a new embedder with configurable download progress.
    pub fn new_with_progress(show_progress: bool) -> Result<Self> {
        Self::new_with_model(None, show_progress, None, None)
    }

    /// Create a new embedder, optionally using a named model from config.
    ///
    /// When `cache_dir` is provided, the ONNX model is stored there instead
    /// of the default `.fastembed_cache/` in the working directory.
    ///
    /// When `device` is `Some("cuda")`, the embedder will attempt to use
    /// CUDA for GPU-accelerated inference. Requires the `cuda` feature and
    /// ONNX Runtime GPU libraries (set `ORT_DYLIB_PATH`). Falls back to CPU
    /// silently if CUDA is unavailable.
    ///
    /// All supported models produce 384-dimensional vectors (matching the
    /// LanceDB storage schema). Models with other dimensions are rejected
    /// to prevent index corruption.
    ///
    /// Supported model names:
    /// - `"BAAI/bge-small-en-v1.5"` (default)
    /// - `"sentence-transformers/all-MiniLM-L6-v2"`
    /// - `"sentence-transformers/all-MiniLM-L12-v2"`
    pub fn new_with_model(
        model_name: Option<&str>,
        show_progress: bool,
        cache_dir: Option<PathBuf>,
        device: Option<&str>,
    ) -> Result<Self> {
        let variant = match model_name {
            Some(name) => match name {
                "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
                "sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
                "sentence-transformers/all-MiniLM-L12-v2" => EmbeddingModel::AllMiniLML12V2,
                _ => anyhow::bail!(
                    "Unsupported embedding model '{name}'. \
                     Supported models (384-dim): BAAI/bge-small-en-v1.5, \
                     sentence-transformers/all-MiniLM-L6-v2, \
                     sentence-transformers/all-MiniLM-L12-v2"
                ),
            },
            None => EmbeddingModel::BGESmallENV15,
        };

        let mut opts = InitOptions::new(variant).with_show_download_progress(show_progress);
        if let Some(dir) = cache_dir {
            std::fs::create_dir_all(&dir).ok();
            opts = opts.with_cache_dir(dir);
        }

        // Configure GPU execution provider when requested
        #[cfg(feature = "cuda")]
        if matches!(device, Some("cuda")) {
            tracing::info!("Requesting CUDA execution provider for embeddings");
            opts = opts.with_execution_providers(vec![
                CUDAExecutionProvider::default().build(),
            ]);
        }

        #[cfg(not(feature = "cuda"))]
        if matches!(device, Some("cuda")) {
            tracing::warn!(
                "CUDA requested but binary built without `cuda` feature. \
                 Rebuild with `cargo build --release --features cuda`. Falling back to CPU."
            );
        }

        let batch_size = if matches!(device, Some("cuda")) {
            BATCH_SIZE_GPU
        } else {
            BATCH_SIZE_CPU
        };

        let model =
            TextEmbedding::try_new(opts).context("Failed to initialize embedding model")?;

        Ok(Self { model, batch_size })
    }

    /// Embed a batch of texts, returning one Vec<f32> per input.
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let embeddings = self
            .model
            .embed(texts.to_vec(), Some(self.batch_size))
            .context("Failed to generate embeddings")?;

        Ok(embeddings)
    }

    /// Embed a single text.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()])?;
        results
            .into_iter()
            .next()
            .context("No embedding returned")
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download
    fn test_embedder_dimensions() {
        let mut embedder = Embedder::new().unwrap();
        let embedding = embedder.embed("fn main() { println!(\"hello\"); }").unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_embedder_batch() {
        let mut embedder = Embedder::new().unwrap();
        let texts = vec![
            "fn add(a: i32, b: i32) -> i32".to_string(),
            "def greet(name): pass".to_string(),
        ];
        let embeddings = embedder.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), EMBEDDING_DIM);
    }
}
