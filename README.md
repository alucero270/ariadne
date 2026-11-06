# Ariadne

Ariadne indexes one folder of Markdown and answers questions about it: by keyword, and — when an
embedding endpoint is configured — by meaning. Agents call it over MCP, everything else over HTTP.
It ships as one Rust binary over SQLite, with no database server.

AI is optional. With no models configured, Ariadne is a keyword search engine.

See [ADR-003](docs/adr/ADR-003-ariadne-rust-rebuild.md) for the rename and rebuild, and
[GLOSSARY.md](GLOSSARY.md) for the terms used below (source, document, chunk, reference, hit).

## Quick start

```bash
cargo build --release -p ariadne

# keyword-only search over a folder of Markdown
./target/release/ariadne --root ~/notes --db ariadne.db index
./target/release/ariadne --root ~/notes --db ariadne.db search "which ports does the printer expose"

# HTTP API and MCP on 127.0.0.1:8090 (a token is required on any other address)
./target/release/ariadne --root ~/notes --db ariadne.db serve

# add embeddings from any OpenAI-compatible endpoint, e.g. llama-swap
./target/release/ariadne --embed-url http://127.0.0.1:8085 --embed-model bge-m3-embed ...

# contextual chunks: a chat model writes a context per chunk at index time (cached)
./target/release/ariadne --context-url http://127.0.0.1:8085 --context-model <chat model> ...
```

Every flag has an environment variable (`ARIADNE_ROOT`, `ARIADNE_DB`, `ARIADNE_EMBED_URL`,
`ARIADNE_RERANK_URL`, `ARIADNE_CONTEXT_URL`, `ARIADNE_BIND`, `ARIADNE_TOKEN`, …). `ariadne --help`
lists them.

## Commands

| command | what it does |
| --- | --- |
| `index` | Build or incrementally update the index from `--root`. Prints build stats as JSON. |
| `search <query>` | Update the index, then search with the server's defaults. |
| `eval <file> -k <n>` | Update the index, then report Recall@k and MRR per mode against an eval set. |
| `serve` | Update the index, then serve HTTP and MCP. |

`eval` takes `{"questions":[{"q","expect":[paths]}]}`. `k` is 1 to 20, default 5, on every front
end.

## Endpoints

- `GET /health`
- `GET /search?q=&k=&mode=hybrid|bm25|dense&rerank=true|false&expand=file|none`
- `GET /documents?path=` — 404 for anything not currently indexed
- `POST /reindex` — searches keep working while it runs
- `POST /mcp` — JSON-RPC; tools `search` and `read_document`

The reranker is off unless a request asks for it, and each file contributes one hit. Every
endpoint but `/health` requires the bearer token when one is set.

## Retrieval

- **bm25**: SQLite FTS5 over each chunk's heading trail, context, and body.
- **dense**: cosine similarity over the embedder's vectors.
- **hybrid**: both, merged by reciprocal rank fusion.
- **rerank**: optional cross-encoder pass over the fused candidates.
- **expand=file**: when the top file is short, its other sections are appended, so a match on one
  section can answer from the whole file.

Dense search loads every vector into memory and scans it. Measured on a 32-core machine with the
`scale` example at 1024 dimensions (the shape of `bge-m3`):

| chunks | keyword build | keyword DB | +vectors DB | vectors | cold dense | bm25 p50 | dense p50 | hybrid p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | 0.05 s | 3 MB | 7 MB | 4 MB | 6 ms | 0.7 ms | 0.5 ms | 1.0 ms |
| 10k | 0.6 s | 29 MB | 70 MB | 39 MB | 51 ms | 4.5 ms | 4.9 ms | 8.8 ms |
| 100k | 7.2 s | 283 MB | 733 MB | 393 MB | 552 ms | 57 ms | 50 ms | 103 ms |

The cost of dense search is memory: about `4 bytes × dimensions × chunks`, plus the index file. At
1024 dimensions that is 393 MB resident for 100k chunks and 3.9 GB for 1M. Hybrid search stays
under 110 ms at 100k chunks; past a million, an approximate index becomes necessary. The vector
write took 13 s at 100k chunks with a synthetic embedder — with a real model the HTTP calls
dominate, so measure one book before projecting a library.

## Correctness and limits

- One source root per index. Nothing outside it is read, and symlinks are never followed.
- Only files currently in the index can be read through Ariadne.
- Markdown only, files up to 2 MiB, chunks up to 1200 characters (`--max-chars`, minimum 100).
- The index is derived: delete the database and it rebuilds from the source folder.
- One writer and any number of readers may share a database; readers keep answering while a
  rebuild runs.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

To measure build time, index size, vector memory, and query latency on generated books:

```bash
cargo run --release --example scale -- --chunks 1000,10000,100000 --queries 50
```

CI runs the same three commands on Linux. The tests that create symlinks are `#[cfg(unix)]`
because Windows only grants the privilege sometimes; they run in CI and are the only part of the
suite not exercised on a Windows checkout.

## Layout

- `crates/ariadne-core` — ingestion, chunking, index, retrieval, eval
- `crates/ariadne` — CLI, HTTP API, MCP server
- `eval/` — the Pantheon eval set
- `ops/ariadne/` — Dockerfile and Compose for the container
- `docs/adr/` — decisions, oldest first

## Roadmap

- More file formats: `txt`, `html`, `epub`, then text PDFs as best effort (#145).
- A second instance, `ariadne-library`, indexing the ebook library keyword-only at first.

Deployment is one instance per corpus: the docs index and the library never share a database, so
a homelab question cannot be answered from a book.
