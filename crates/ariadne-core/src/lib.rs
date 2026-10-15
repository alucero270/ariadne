//! Ariadne core: one source folder of Markdown, indexed for keyword and meaning-based search.
//!
//! The interface is [`Index`]: open it over a [`Source`] with optional [`Models`], `build` it,
//! then `search` and `read_document`. Everything else is implementation.

pub mod chunk;
pub mod eval;
mod index;
pub mod ports;
mod rank;
pub mod source;
mod store;

pub use index::{BuildStats, Hit, Index, IndexStats, Mode, SearchOptions, Source};
pub use ports::{Embedder, Models, Reranker};
