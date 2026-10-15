# Strata

> **Becoming Ariadne.** Strata is being renamed **Ariadne** and rebuilt in Rust, with hybrid
> search (keyword plus embeddings) and an MCP server for agents. See
> [ADR-003](docs/adr/ADR-003-ariadne-rust-rebuild.md). The Rust workspace lives in `crates/`;
> the .NET code under `src/` is unchanged until Ariadne reaches parity. Quick start below.
>
> ```bash
> cargo build --release -p ariadne
> # keyword-only search over a folder of Markdown
> ./target/release/ariadne --root ~/notes --db ariadne.db index
> ./target/release/ariadne --root ~/notes --db ariadne.db search "which ports does the printer expose"
> # HTTP API and MCP on 127.0.0.1:8090 (a token is required on any other address)
> ./target/release/ariadne --root ~/notes --db ariadne.db serve
> # add embeddings from any OpenAI-compatible endpoint, e.g. llama-swap
> ./target/release/ariadne --embed-url http://127.0.0.1:8085 --embed-model bge-m3-embed ...
> # contextual chunks: a chat model writes a context per chunk at index time (cached)
> ./target/release/ariadne --context-url http://127.0.0.1:8085 --context-model <chat model> ...
> ```
>
> Endpoints: `GET /health`, `GET /search?q=&k=&mode=hybrid|bm25|dense` (`k` from 1 to 20, default 5 on every
> front end),
> `GET /documents?path=` (404 for anything not indexed), `POST /reindex` (searches keep working
> while it runs), and `POST /mcp` (JSON-RPC: tools `search` and `read_document`). Every endpoint
> but `/health` needs the bearer token when one is set. Measure retrieval with
> `ariadne eval eval/pantheon-evalset.json --root <pantheon checkout>`. Docker: `ops/ariadne/` (the base file publishes no port; add
> `-f compose.local.yml` to reach it on `127.0.0.1:8090`).
> Terms are defined in [GLOSSARY.md](GLOSSARY.md).

Strata is a self-hosted knowledge ingestion and retrieval system for internal
content with explicit source boundaries and no required AI layer.

Strata is designed to be useful with filesystem ingestion, metadata storage,
and full-text retrieval alone. AI is an optional enhancement path, not a
prerequisite for core product value.

## The Problem

Teams store important knowledge across shared folders, docs, and internal
artifacts, but retrieving the right information quickly is still harder than it
should be.

Many tools make that problem worse by:

- blurring where data came from
- hiding boundary and safety assumptions
- making AI a requirement instead of an enhancement

Strata is intended to provide a more controlled foundation: explicit sources,
predictable ingestion, and traceable retrieval.

## Current Product Slice

The current repository implements a focused early slice of the product:

- filesystem-based ingestion from an explicitly configured source root
- markdown scanning with checksum-based sync into PostgreSQL
- asynchronous indexing through index-job endpoints and a background worker
- full-text search through the HTTP API
- document read access through the HTTP API
- a minimal web UI for search and document viewing
- Docker-based local and controlled-environment deployment

This is the current implemented slice, not the full product roadmap.

## Core Principles

- AI is optional, not foundational
- source boundaries are explicit and server-controlled
- source content is mounted read-only and is never edited in place
- retrieval should remain fast, predictable, and traceable

## What Strata Is Not

- not a chat-first interface
- not an AI-dependent product
- not a document editor
- not a general-purpose note-taking platform

Strata is a retrieval foundation for controlled internal knowledge access.

## Current Scope and Known Gaps

Today, Strata is still early and intentionally narrow.

Implemented now:

- one configured filesystem source boundary per deployment
- a first-class persisted source record for the configured filesystem source
- markdown ingestion and indexing
- retrieval APIs for search, document reads, and indexing jobs
- product-facing Docker and environment configuration

Planned or still maturing:

- multi-source modeling, source-scoped ingestion, and source management APIs
- stronger runtime boundary hardening, including symlink escape protection
- richer retrieval metadata, filtering, and pagination
- broader verification coverage and more complete operator runbooks
- optional AI enhancement, additional connectors, and multi-user access control

## Architecture

Current flow:

`Configured source root -> Indexer -> PostgreSQL -> Retrieval API -> Web UI`

Planned expansion follows the milestone roadmap in
[`docs/milestones.md`](docs/milestones.md).

## Getting Started

1. Copy `.env.example` to `.env`.
2. Set `STRATA_SOURCES` to a readable host directory that Strata is allowed to ingest.
3. Review `POSTGRES_*`, `STRATA_DB_CONNECTION`, `STRATA_API_PORT`, and optional embedding settings if you plan to enable AI enhancement services.
4. Start the stack:

```powershell
docker compose -f ops/docker-compose.yml --env-file .env up -d --build
```

This default path starts the core retrieval stack without the embedder.

Optional AI enhancement services:

```powershell
docker compose -f ops/docker-compose.yml --env-file .env --profile ai up -d --build
```

5. Apply the SQL migrations described in [`docs/operations.md`](docs/operations.md).
6. Verify API readiness at `/health` and smoke-test retrieval using the commands in [`docs/operations.md`](docs/operations.md).

## Documentation

- [`docs/charter.md`](docs/charter.md)
- [`docs/scope.md`](docs/scope.md)
- [`docs/requirements.md`](docs/requirements.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/data-model.md`](docs/data-model.md)
- [`docs/feasibility.md`](docs/feasibility.md)
- [`docs/quality-attributes.md`](docs/quality-attributes.md)
- [`docs/api.md`](docs/api.md)
- [`docs/api-surface-audit.md`](docs/api-surface-audit.md)
- [`docs/delivery-process.md`](docs/delivery-process.md)
- [`docs/operations.md`](docs/operations.md)
- [`docs/test-strategy.md`](docs/test-strategy.md)
- [`docs/traceability.md`](docs/traceability.md)
- [`docs/use-cases.md`](docs/use-cases.md)
- [`docs/milestones.md`](docs/milestones.md)
- [`docs/adr/ADR-001-project-identity.md`](docs/adr/ADR-001-project-identity.md)
- [`docs/adr/ADR-002-internal-rename-strategy.md`](docs/adr/ADR-002-internal-rename-strategy.md)

## Repository Layout

Strata now standardizes around a simple top-level layout:

- `src/` for product code and runtime projects
- `tests/` for test projects as coverage is added
- `docs/` for product, architecture, and operations documentation
- `build/` for repository-level build helpers and future build customizations
- `ops/` for deployment and runtime assets such as Compose, Dockerfiles,
  migrations, and operational validation scripts
- `artifacts/` as the reserved generated-output location for future build and
  packaging work

Current note:

- `ops/` remains separate from `build/` because its contents are deployment and
  runtime assets, not build customizations
- `Codex.slnx` is still the current solution file name during the internal
  rename transition
- empty layout placeholders do not imply finished test or packaging coverage

## Build Bootstrap

Use the root bootstrap scripts to run the current baseline validation flow:

```powershell
.\build.cmd
```

```bash
./build.sh
```

These entry points are bootstrap wrappers for the current readiness validation,
not a full packaging pipeline yet.

## Roadmap

- Milestone 2: source-aware ingestion and source-boundary hardening
- Milestone 3: richer retrieval behavior, provenance, and stable API contracts
- Milestone 4: optional AI-enhanced retrieval without breaking the non-AI baseline
- Milestone 5: connectors beyond filesystem ingestion
- Milestone 6: authenticated, source-scoped organizational usage

## Implementation Notes

- internal project names still use `Codex.*` in several runtime paths during the transition to Strata
- product-facing docs and environment configuration already use the Strata identity
- optional embedding infrastructure remains additive and should not be required for baseline retrieval
