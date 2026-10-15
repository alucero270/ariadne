# Glossary

Domain terms for Ariadne (formerly Strata). Code, tests, and docs use these names.

**Source**: the one folder Ariadne indexes, with the paths under it to skip and the largest
chunk size. Read-only. Nothing outside it is read, and symlinks are never followed.

**Document**: a Markdown file under the source. Only documents currently in the index can be
read through Ariadne.

**Chunk**: a section of a document, cut at headings and split further at paragraph boundaries
when long. It carries its document's title and its heading trail.

**Reference**: how a chunk is cited, in wiki-link form: the document path without `.md`, plus
`#` and the chunk's heading. Example: `systems/fabrication/devices/qidi-plus-4#Network Placement`.

**Index**: the searchable, derived copy of a source, stored in one SQLite file. Deleting it
loses nothing; `build` recreates it.

**Build**: bringing the index up to date with the source. Unchanged documents are skipped; a
different embedder, chunk size, or source root rebuilds everything.

**Hit**: one search result: a chunk, its reference, its score, and its rank from each
retriever that found it.

**Retriever**: one way of finding chunks: **keyword** (BM25 over full text) or **dense**
(similarity of embedding vectors). **Hybrid** search fuses both. In the API, the keyword
retriever is named `bm25`, both as a `mode` and in each hit's `ranks`.

**Models**: the optional AI behind the index: an **embedder** (text to vectors) and a
**reranker** (rescores hits against the query). Without models, the index is keyword-only.

**Eval set**: questions with the documents that answer them, used to measure retrieval as
Recall@k and MRR.
