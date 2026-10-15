# ADR-003: Rename to Ariadne and Rebuild in Rust

## Status

Accepted. Decided by the operator on 2026-10-15. Tracked in #134.

## Context

Two name collisions make "Strata" and its internal name "Codex" ambiguous:

- **Strata** is also [Niko1221/Strata](https://github.com/Niko1221/Strata), a ggml inference engine
  for Qwen models that the operator evaluates on the same homelab.
- **Codex** is OpenAI Codex, an agent harness the operator also uses.

The project's role has also changed. It is now the knowledge-search service for the operator's
assistant stack (Jarvis), next to [Canonis](https://github.com/Loose-Arrow-Labs/canonis) for
governed memory and [Anemoi](https://github.com/Loose-Arrow-Labs/anemoi) for model governance. Both
are Rust. Agents call search as a tool, increasingly over MCP.

The .NET implementation had reached ingestion, Postgres full-text search, and index jobs.
Milestone 4 (embeddings and hybrid search) was not started: `Codex.Embedder` is the worker
template. Development had been idle since 2026-04-14.

A Python prototype of Milestone 4 (`docs-rag` in alucero270/pantheon#263) measured hybrid search
against a 30-question eval set and set the quality bar.

ADR-002 deferred a deep internal rename until a trigger applied. Two have: real confusion that
documentation can't fix (the collisions above), and a coordinated rebuild that absorbs the churn.

## Decision

- The product is renamed **Ariadne**: the thread through the labyrinth. The GitHub repository
  rename is a manual operator step.
- Ariadne is rebuilt in **Rust**, as a workspace of `ariadne-core` (ingestion, chunking, index,
  retrieval, eval) and `ariadne` (CLI, HTTP API, MCP server).
- Storage is **SQLite**: FTS5 for keyword search, with vectors stored per chunk. The index is
  derived data and rebuilds from the source folder. This follows Canonis's
  filesystem-truth-plus-rebuildable-index pattern and removes the Postgres dependency.
- The **AI-optional principle from ADR-001 stays.** With no embedding endpoint configured,
  Ariadne is a keyword search engine. Embeddings and reranking come from OpenAI-compatible HTTP
  endpoints (such as llama-swap) rather than being bundled.
- Retrieval is **hybrid**: BM25 plus dense vectors merged by reciprocal rank fusion, an optional
  reranker, and a cap of two chunks per file.
- Search is exposed over **MCP** (`search`, `read_document`) as well as HTTP, so any agent client
  can use it.
- The **source boundary** carries over and gets stricter: one root, symlinks never followed,
  reads limited to files currently in the index.
- The .NET code stays in the repository, untouched, until Ariadne reaches parity on what users
  rely on. Its removal is a separate, later change.

## Consequences

### Positive

- One language across Ariadne, Canonis, and Anemoi; patterns and code move between them.
- One small binary with no database server. The release build is about 8 MB and indexed 366
  Markdown files (keyword only) in 0.22 s.
- Retrieval quality is measured, not assumed. With the same embeddings on the same corpus,
  Ariadne matches the prototype: hybrid Recall@5 0.933, MRR 0.648 versus 0.649.

### Negative / Tradeoffs

- The working .NET slice is replaced rather than extended.
- SQLite with a single writer suits one operator and one source. Multi-tenant or high-concurrency
  use would need a different store.
- Dense search scans every vector in memory. That's fine for tens of thousands of chunks; much
  larger corpora would need an approximate index.
- The web UI is not rebuilt yet. Agents and the HTTP API are the interfaces for now.
- The repository and documentation still say Strata until the rename lands.

## Conditions for Revisiting

- More than one source root or multiple concurrent writers become requirements.
- Corpus size makes the in-memory vector scan too slow or too large.
- MCP stops being the common way agents call tools.
