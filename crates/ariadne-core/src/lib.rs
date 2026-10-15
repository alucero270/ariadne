//! Ariadne core: one source folder of Markdown, indexed for keyword and meaning-based search.
//!
//! The interface is [`Index`]: open it over a [`Source`] with optional [`Models`], `build` it,
//! then `search`, `read_document`, and `evaluate`. Everything else is implementation.

mod chunk;
mod eval;
mod index;
mod ports;
mod rank;
mod source;
mod store;

pub use eval::{EvalReport, EvalSet, ModeScore, Question};
pub use index::{BuildStats, Hit, Index, IndexStats, Mode, SearchOptions, Source};
pub use ports::{Embedder, Models, Reranker};
