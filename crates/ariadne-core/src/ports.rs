//! Ports to the optional AI models. The HTTP adapters live in the `ariadne` binary; tests use
//! in-memory adapters. With no models, the index is keyword-only.

use anyhow::Result;

/// Turns text into vectors. Implementations return one unit-length vector per input, in order.
pub trait Embedder: Send {
    /// Stable identity, stored with the index. A different name forces a full re-embed.
    fn name(&self) -> String;
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Scores documents against a query. Implementations return one score per document, in order.
pub trait Reranker: Send {
    fn name(&self) -> String;
    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>>;
}

/// The optional models an index uses.
#[derive(Default)]
pub struct Models {
    pub embedder: Option<Box<dyn Embedder>>,
    pub reranker: Option<Box<dyn Reranker>>,
}

pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}
