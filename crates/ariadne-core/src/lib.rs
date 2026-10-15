//! Ariadne core: ingestion, chunking, indexing, and hybrid retrieval over one source root.

pub mod chunk;
pub mod eval;
pub mod index;
pub mod models;
pub mod source;

pub use index::{BuildStats, Hit, Index, IndexStats, Mode, SearchOptions};
