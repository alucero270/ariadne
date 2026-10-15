//! The index: one source folder of Markdown, searchable by keyword and, when an embedder is
//! configured, by meaning. Derived data: delete the database and it rebuilds from the source.
//!
//! One writer and any number of readers may open the same database at once. Readers keep
//! answering while a writer rebuilds, and pick up the new data once the rebuild commits.

use crate::chunk::{chunk_markdown, reference};
use crate::ports::{normalize, Embedder, Models};
use crate::rank::{cap_per_key, fts_query, fuse};
use crate::source::{resolve_inside, walk_markdown, DEFAULT_EXCLUDES};
use crate::store;
use anyhow::{bail, Result};
use rusqlite::Connection;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// What to index: a root folder (read-only), the paths under it to skip, and the largest chunk
/// in characters. See GLOSSARY.md.
#[derive(Debug, Clone)]
pub struct Source {
    root: PathBuf,
    excludes: Vec<String>,
    max_chars: usize,
}

impl Source {
    /// A source with the default excludes (VCS and editor folders, `node_modules`, scratch
    /// artifacts) and chunks of at most 1200 characters.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            excludes: DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect(),
            max_chars: 1200,
        }
    }

    pub fn with_max_chars(self, max_chars: usize) -> Self {
        Self { max_chars, ..self }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Hybrid,
    Bm25,
    Dense,
}

impl Mode {
    pub const ALL: [Mode; 3] = [Mode::Bm25, Mode::Dense, Mode::Hybrid];

    pub fn name(self) -> &'static str {
        match self {
            Mode::Hybrid => "hybrid",
            Mode::Bm25 => "bm25",
            Mode::Dense => "dense",
        }
    }

    fn retrievers(self) -> &'static [Retriever] {
        match self {
            Mode::Hybrid => &[Retriever::Keyword, Retriever::Dense],
            Mode::Bm25 => &[Retriever::Keyword],
            Mode::Dense => &[Retriever::Dense],
        }
    }
}

/// One way of finding chunks. See GLOSSARY.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Retriever {
    Keyword,
    Dense,
}

impl Retriever {
    /// The name callers see in `Hit::ranks`, matching the `mode` names.
    fn label(self) -> &'static str {
        match self {
            Retriever::Keyword => "bm25",
            Retriever::Dense => "dense",
        }
    }
}

/// Hits kept from fusion before reranking and the per-document cap, so both have enough
/// candidates to choose from even when `k` is small.
const MIN_FUSION_POOL: usize = 20;

impl std::str::FromStr for Mode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "hybrid" => Ok(Mode::Hybrid),
            "bm25" => Ok(Mode::Bm25),
            "dense" => Ok(Mode::Dense),
            other => bail!("unknown mode {other}; use hybrid, bm25, or dense"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    k: usize,
    mode: Mode,
    rerank: bool,
    /// Candidates taken from each retriever before fusion.
    candidates: usize,
    /// At most this many hits per document; `None` for no cap.
    max_per_document: Option<usize>,
}

impl SearchOptions {
    /// `k` hits in `mode`; `rerank` applies the reranker when one is configured. Each document
    /// contributes at most two hits.
    pub fn new(k: usize, mode: Mode, rerank: bool) -> Self {
        Self {
            k,
            mode,
            rerank,
            candidates: 40,
            max_per_document: Some(2),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct BuildStats {
    pub files_seen: usize,
    pub files_changed: usize,
    pub files_removed: usize,
    pub chunks_written: usize,
    pub seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexStats {
    pub files: i64,
    pub chunks: i64,
    pub embedder: Option<String>,
    pub reranker: Option<String>,
    pub indexed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub id: i64,
    /// Citation: path without `.md`, plus `#heading`.
    #[serde(rename = "ref")]
    pub reference: String,
    pub path: String,
    pub title: String,
    pub heading: String,
    pub text: String,
    pub score: f64,
    /// 1-based rank from each retriever that found this chunk.
    pub ranks: HashMap<String, usize>,
}

/// Vectors held in memory for dense search, tagged with the database version they came from.
struct VectorCache {
    data_version: i64,
    ids: Vec<i64>,
    vectors: Vec<Vec<f32>>,
}

pub struct Index {
    conn: Connection,
    source: Source,
    models: Models,
    cache: Option<VectorCache>,
}

impl Index {
    pub fn open(db: &Path, source: Source, models: Models) -> Result<Self> {
        Ok(Self {
            conn: store::open(db)?,
            source,
            models,
            cache: None,
        })
    }

    fn embedder_name(&self) -> String {
        self.models
            .embedder
            .as_ref()
            .map(|e| e.name())
            .unwrap_or_else(|| "none".into())
    }

    /// Bring the index up to date with the source. Unchanged files are skipped; a different
    /// embedder, chunk size, or root rebuilds everything.
    pub fn build(&mut self) -> Result<BuildStats> {
        let start = Instant::now();
        let root = self.source.root.canonicalize()?;
        let files = walk_markdown(&root, &self.source.excludes)?;
        let settings = [
            ("embedder", self.embedder_name()),
            ("max_chars", self.source.max_chars.to_string()),
            ("root", root.to_string_lossy().into_owned()),
        ];
        let mut full = false;
        for (key, value) in &settings {
            full |= store::meta(&self.conn, key)?.as_deref() != Some(value.as_str());
        }

        let tx = self.conn.transaction()?;
        if full {
            store::clear(&tx)?;
        }
        let known = store::file_hashes(&tx)?;
        let mut dimensions: Option<usize> = match full {
            true => None,
            false => store::meta(&tx, "dimensions")?.and_then(|d| d.parse().ok()),
        };
        let mut stats = BuildStats {
            files_seen: files.len(),
            ..Default::default()
        };
        let mut seen = HashSet::new();
        for file in &files {
            seen.insert(file.rel.as_str());
            let raw = std::fs::read(&file.abs)?;
            let sha = format!("{:x}", Sha256::digest(&raw));
            if known.get(&file.rel) == Some(&sha) {
                continue;
            }
            let chunks = chunk_markdown(
                &file.rel,
                &String::from_utf8_lossy(&raw),
                self.source.max_chars,
            );
            let vectors = match &self.models.embedder {
                Some(e) if !chunks.is_empty() => {
                    let texts: Vec<String> = chunks.iter().map(|c| c.embed_text()).collect();
                    Some(embed_checked(e.as_ref(), &texts)?)
                }
                _ => None,
            };
            for v in vectors.iter().flatten() {
                match dimensions {
                    None => dimensions = Some(v.len()),
                    Some(d) if d != v.len() => bail!(
                        "embedding size changed from {d} to {} under the same embedder name; \
                         delete the index file to rebuild",
                        v.len()
                    ),
                    Some(_) => {}
                }
            }
            store::write_file(&tx, &file.rel, &sha, &chunks, vectors.as_deref())?;
            stats.files_changed += 1;
            stats.chunks_written += chunks.len();
        }
        for gone in known.keys().filter(|p| !seen.contains(p.as_str())) {
            store::delete_file(&tx, gone)?;
            stats.files_removed += 1;
        }
        for (key, value) in &settings {
            store::set_meta(&tx, key, value)?;
        }
        if let Some(d) = dimensions {
            store::set_meta(&tx, "dimensions", &d.to_string())?;
        }
        store::set_meta(&tx, "indexed_at", &now_utc())?;
        tx.commit()?;
        self.cache = None;
        stats.seconds = (start.elapsed().as_secs_f64() * 100.0).round() / 100.0;
        Ok(stats)
    }

    pub(crate) fn has_embedder(&self) -> bool {
        self.models.embedder.is_some()
    }

    pub(crate) fn has_reranker(&self) -> bool {
        self.models.reranker.is_some()
    }

    pub fn stats(&self) -> Result<IndexStats> {
        let (files, chunks) = store::counts(&self.conn)?;
        Ok(IndexStats {
            files,
            chunks,
            embedder: store::meta(&self.conn, "embedder")?,
            reranker: self.models.reranker.as_ref().map(|r| r.name()),
            indexed_at: store::meta(&self.conn, "indexed_at")?,
        })
    }

    /// Best chunks for `query`, cited. Hybrid mode fuses keyword and dense rankings; without an
    /// embedder it is keyword-only and dense mode returns nothing.
    pub fn search(&mut self, query: &str, opts: &SearchOptions) -> Result<Vec<Hit>> {
        let rankings = self.retrieve(query, opts)?;
        let fused = fuse(
            &rankings
                .iter()
                .map(|(_, r)| r.as_slice())
                .collect::<Vec<_>>(),
        );
        let mut hits = fused
            .into_iter()
            .take(opts.k.max(MIN_FUSION_POOL))
            .map(|(id, score)| self.to_hit(id, score, &rankings))
            .collect::<Result<Vec<_>>>()?;
        if opts.rerank {
            self.rerank(query, &mut hits)?;
        }
        if let Some(cap) = opts.max_per_document {
            hits = cap_per_key(hits, cap, |h| h.path.as_str());
        }
        hits.truncate(opts.k);
        Ok(hits)
    }

    /// The full text of an indexed document, or `None` if `path` isn't one. The `.md` suffix
    /// is optional. Nothing outside the index or the source root is ever read.
    pub fn read_document(&self, path: &str) -> Result<Option<String>> {
        let path = path.trim_start_matches('/');
        let path = if path.ends_with(".md") {
            path.to_string()
        } else {
            format!("{path}.md")
        };
        if !store::is_indexed(&self.conn, &path)? {
            return Ok(None);
        }
        match resolve_inside(&self.source.root, &path) {
            Ok(file) => Ok(Some(std::fs::read_to_string(file)?)),
            Err(_) => Ok(None),
        }
    }

    /// Ranked chunk ids from each retriever the mode uses; empty rankings are dropped.
    fn retrieve(
        &mut self,
        query: &str,
        opts: &SearchOptions,
    ) -> Result<Vec<(Retriever, Vec<i64>)>> {
        let mut rankings = Vec::new();
        for &retriever in opts.mode.retrievers() {
            let ids = match retriever {
                Retriever::Keyword => {
                    store::keyword_ids(&self.conn, &fts_query(query), opts.candidates)?
                }
                Retriever::Dense => self.dense_ids(query, opts.candidates)?,
            };
            if !ids.is_empty() {
                rankings.push((retriever, ids));
            }
        }
        Ok(rankings)
    }

    fn dense_ids(&mut self, query: &str, limit: usize) -> Result<Vec<i64>> {
        let Some(embedder) = &self.models.embedder else {
            return Ok(Vec::new());
        };
        if store::meta(&self.conn, "embedder")?.as_deref() != Some(embedder.name().as_str()) {
            return Ok(Vec::new());
        }
        let data_version = store::data_version(&self.conn)?;
        if self.cache.as_ref().map(|c| c.data_version) != Some(data_version) {
            let (ids, vectors) = store::vectors(&self.conn)?;
            self.cache = Some(VectorCache {
                data_version,
                ids,
                vectors,
            });
        }
        let cache = self.cache.as_ref().expect("cache filled above");
        if cache.ids.is_empty() {
            return Ok(Vec::new());
        }
        let query_vector = embed_checked(embedder.as_ref(), &[query.to_string()])?
            .pop()
            .expect("embed_checked returns one vector per input");
        let stored = cache.vectors[0].len();
        if query_vector.len() != stored {
            bail!(
                "query embedding has {} dimensions but the index has {stored}; \
                 delete the index file to rebuild",
                query_vector.len()
            );
        }
        let mut scored: Vec<(f32, i64)> = cache
            .vectors
            .iter()
            .zip(&cache.ids)
            .map(|(v, id)| (v.iter().zip(&query_vector).map(|(a, b)| a * b).sum(), *id))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        Ok(scored.into_iter().take(limit).map(|(_, id)| id).collect())
    }

    fn to_hit(&self, id: i64, score: f64, rankings: &[(Retriever, Vec<i64>)]) -> Result<Hit> {
        let c = store::chunk(&self.conn, id)?;
        let ranks = rankings
            .iter()
            .filter_map(|(retriever, r)| {
                r.iter()
                    .position(|x| *x == id)
                    .map(|p| (retriever.label().to_string(), p + 1))
            })
            .collect();
        Ok(Hit {
            id,
            reference: reference(&c.path, &c.anchor),
            path: c.path,
            title: c.title,
            heading: c.heading,
            text: c.body,
            score,
            ranks,
        })
    }

    fn rerank(&self, query: &str, hits: &mut [Hit]) -> Result<()> {
        let Some(reranker) = &self.models.reranker else {
            return Ok(());
        };
        if hits.is_empty() {
            return Ok(());
        }
        let docs: Vec<String> = hits
            .iter()
            .map(|h| format!("{} > {}\n\n{}", h.title, h.heading, h.text))
            .collect();
        let scores = reranker.rerank(query, &docs)?;
        for (h, s) in hits.iter_mut().zip(scores) {
            h.score = f64::from(s);
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(())
    }
}

/// Hold the `Embedder` contract here rather than trusting each adapter: one vector per input,
/// and unit length so a dot product is cosine similarity.
fn embed_checked(embedder: &dyn Embedder, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let vectors = embedder.embed(texts)?;
    if vectors.len() != texts.len() {
        bail!(
            "embedder {} returned {} vectors for {} texts",
            embedder.name(),
            vectors.len(),
            texts.len()
        );
    }
    Ok(vectors.into_iter().map(normalize).collect())
}

fn now_utc() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let (h, m, s) = (secs % 86_400 / 3600, secs % 3600 / 60, secs % 60);
    // Civil-from-days (Howard Hinnant), to avoid a date dependency for one timestamp.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(mo <= 2);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ports::Reranker;
    use std::fs;

    /// Deterministic bag-of-words embedder so dense search is testable without a model.
    pub struct HashEmbedder;

    impl Embedder for HashEmbedder {
        fn name(&self) -> String {
            "hash:64".into()
        }
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0f32; 64];
                    for tok in t.to_lowercase().split_whitespace() {
                        v[usize::from(Sha256::digest(tok.as_bytes())[0]) % 64] += 1.0;
                    }
                    normalize(v)
                })
                .collect())
        }
    }

    struct ReverseReranker;

    impl Reranker for ReverseReranker {
        fn name(&self) -> String {
            "rerank:reverse".into()
        }
        fn rerank(&self, _: &str, docs: &[String]) -> Result<Vec<f32>> {
            Ok((0..docs.len()).map(|i| i as f32).collect())
        }
    }

    pub const PRINTER: &str = "# Qidi Plus 4\n\nIntro paragraph about the printer.\n\n## Network Placement\n\nIOT VLAN 30 at 192.168.30.10. Moonraker listens on port 7125.\n\n### Firewall\n\nOnly USER VLAN 20 may reach it.\n";
    pub const VOICE: &str = "# Voice Agent\n\n## Latency\n\nWhisper large-v3-turbo transcribes speech; Qwen3-TTS speaks the reply.\n";

    pub struct Fixture {
        pub src: tempfile::TempDir,
        pub db: tempfile::TempDir,
    }

    impl Fixture {
        pub fn new() -> Self {
            let src = tempfile::tempdir().unwrap();
            fs::create_dir_all(src.path().join("systems/fabrication")).unwrap();
            fs::write(src.path().join("systems/fabrication/qidi.md"), PRINTER).unwrap();
            fs::write(src.path().join("voice.md"), VOICE).unwrap();
            Self {
                src,
                db: tempfile::tempdir().unwrap(),
            }
        }

        pub fn open(&self, embedder: bool) -> Index {
            let models = Models {
                embedder: embedder.then(|| Box::new(HashEmbedder) as Box<dyn Embedder>),
                reranker: None,
            };
            self.open_with(models)
        }

        pub fn open_with(&self, models: Models) -> Index {
            Index::open(
                &self.db.path().join("i.db"),
                Source::new(self.src.path()),
                models,
            )
            .unwrap()
        }

        pub fn write(&self, rel: &str, text: &str) {
            fs::write(self.src.path().join(rel), text).unwrap();
        }
    }

    fn opts(mode: Mode, k: usize) -> SearchOptions {
        SearchOptions::new(k, mode, true)
    }

    #[test]
    fn timestamp_format() {
        let t = now_utc();
        assert_eq!(t.len(), 20);
        assert!(t.starts_with("20") && t.ends_with('Z'));
    }

    #[test]
    fn build_and_search_all_modes() {
        let f = Fixture::new();
        let mut idx = f.open(true);
        let stats = idx.build().unwrap();
        assert_eq!(
            (stats.files_seen, stats.files_changed, stats.chunks_written),
            (2, 2, 4)
        );
        for mode in [Mode::Bm25, Mode::Dense, Mode::Hybrid] {
            let hits = idx.search("moonraker port 7125", &opts(mode, 3)).unwrap();
            assert_eq!(
                hits[0].reference, "systems/fabrication/qidi#Network Placement",
                "{mode:?}"
            );
        }
        let hit = &idx
            .search("whisper speech", &opts(Mode::Hybrid, 1))
            .unwrap()[0];
        assert_eq!(hit.path, "voice.md");
        assert!(hit.ranks.keys().all(|k| k == "bm25" || k == "dense"));
    }

    #[test]
    fn query_syntax_cannot_break_search() {
        let f = Fixture::new();
        let mut idx = f.open(false);
        idx.build().unwrap();
        for q in ["\"unbalanced", "NEAR(", "a AND OR", "*", "the of"] {
            assert!(idx.search(q, &opts(Mode::Hybrid, 5)).is_ok(), "{q}");
        }
    }

    #[test]
    fn incremental_reindex_and_removal() {
        let f = Fixture::new();
        let mut idx = f.open(true);
        idx.build().unwrap();
        assert_eq!(idx.build().unwrap().files_changed, 0);

        f.write(
            "voice.md",
            &format!("{VOICE}\n## Barge-in\n\nInterruptions are supported.\n"),
        );
        assert_eq!(idx.build().unwrap().files_changed, 1);
        assert_eq!(
            idx.search("interruptions", &opts(Mode::Hybrid, 1)).unwrap()[0].heading,
            "Barge-in"
        );

        fs::remove_file(f.src.path().join("voice.md")).unwrap();
        assert_eq!(idx.build().unwrap().files_removed, 1);
        assert!(idx
            .search("whisper", &opts(Mode::Hybrid, 5))
            .unwrap()
            .iter()
            .all(|h| h.path != "voice.md"));
        assert_eq!(idx.stats().unwrap().files, 1);
    }

    #[test]
    fn embedder_change_forces_full_rebuild() {
        let f = Fixture::new();
        f.open(false).build().unwrap();
        let mut idx = f.open(true);
        assert_eq!(idx.build().unwrap().files_changed, 2);
        assert!(!idx
            .search("whisper", &opts(Mode::Dense, 1))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn keyword_only_without_embedder() {
        let f = Fixture::new();
        let mut idx = f.open(false);
        idx.build().unwrap();
        assert!(idx
            .search("whisper", &opts(Mode::Dense, 1))
            .unwrap()
            .is_empty());
        assert_eq!(
            idx.search("whisper", &opts(Mode::Hybrid, 1)).unwrap()[0].path,
            "voice.md"
        );
        assert_eq!(idx.stats().unwrap().embedder.as_deref(), Some("none"));
    }

    #[test]
    fn reranker_reorders_and_document_cap_applies() {
        let f = Fixture::new();
        let mut idx = f.open_with(Models {
            embedder: Some(Box::new(HashEmbedder)),
            reranker: Some(Box::new(ReverseReranker)),
        });
        idx.build().unwrap();
        let base = SearchOptions {
            k: 20,
            max_per_document: None,
            ..SearchOptions::new(8, Mode::Hybrid, true)
        };
        let plain = idx
            .search(
                "printer vlan",
                &SearchOptions {
                    rerank: false,
                    ..base.clone()
                },
            )
            .unwrap();
        let reranked = idx.search("printer vlan", &base).unwrap();
        let reversed: Vec<_> = plain.iter().rev().map(|h| h.id).collect();
        assert_eq!(reranked.iter().map(|h| h.id).collect::<Vec<_>>(), reversed);

        let capped = idx
            .search(
                "printer vlan moonraker whisper",
                &SearchOptions {
                    k: 10,
                    max_per_document: Some(1),
                    ..SearchOptions::new(8, Mode::Hybrid, true)
                },
            )
            .unwrap();
        let paths: HashSet<_> = capped.iter().map(|h| h.path.clone()).collect();
        assert_eq!(paths.len(), capped.len());
    }

    #[test]
    fn read_document_only_reads_indexed_files_inside_root() {
        let f = Fixture::new();
        f.write("notes.txt", "not indexed");
        let mut idx = f.open(false);
        idx.build().unwrap();
        assert!(idx
            .read_document("systems/fabrication/qidi")
            .unwrap()
            .unwrap()
            .contains("Moonraker"));
        assert!(idx
            .read_document("/voice.md")
            .unwrap()
            .unwrap()
            .contains("Whisper"));
        for bad in ["notes.txt", "../etc/passwd", "missing.md"] {
            assert!(
                idx.read_document(bad).unwrap().is_none(),
                "{bad} should not be readable"
            );
        }
    }

    /// Same name as `HashEmbedder`, different size: a model swapped behind an unchanged name.
    struct ResizedEmbedder;

    impl Embedder for ResizedEmbedder {
        fn name(&self) -> String {
            "hash:64".into()
        }
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| normalize(vec![1.0; 32])).collect())
        }
    }

    #[test]
    fn a_resized_embedder_is_an_error_not_garbage() {
        let f = Fixture::new();
        f.open(true).build().unwrap();
        let mut resized = f.open_with(Models {
            embedder: Some(Box::new(ResizedEmbedder)),
            reranker: None,
        });
        let err = resized
            .search("whisper", &opts(Mode::Dense, 1))
            .unwrap_err();
        assert!(err.to_string().contains("dimensions"), "{err}");
        f.write("new.md", "# New\n\nfresh text\n");
        let err = resized.build().unwrap_err();
        assert!(err.to_string().contains("embedding size changed"), "{err}");
    }

    /// Breaks the port contract both ways: unnormalized vectors, and one vector too few.
    struct SloppyEmbedder {
        short: bool,
    }

    impl Embedder for SloppyEmbedder {
        fn name(&self) -> String {
            "sloppy".into()
        }
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let n = if self.short {
                texts.len() - 1
            } else {
                texts.len()
            };
            let hashed = HashEmbedder.embed(&texts[..n])?;
            Ok(hashed
                .into_iter()
                .map(|v| v.iter().map(|x| x * 7.0).collect())
                .collect())
        }
    }

    #[test]
    fn the_index_holds_the_embedder_contract() {
        let f = Fixture::new();
        let mut unnormalized = f.open_with(Models {
            embedder: Some(Box::new(SloppyEmbedder { short: false })),
            reranker: None,
        });
        unnormalized.build().unwrap();
        let hits = unnormalized
            .search("moonraker port 7125", &opts(Mode::Dense, 1))
            .unwrap();
        assert_eq!(
            hits[0].reference,
            "systems/fabrication/qidi#Network Placement"
        );

        let g = Fixture::new();
        let mut short = g.open_with(Models {
            embedder: Some(Box::new(SloppyEmbedder { short: true })),
            reranker: None,
        });
        let err = short.build().unwrap_err();
        assert!(err.to_string().contains("returned"), "{err}");
    }

    #[test]
    fn readers_see_a_writers_rebuild_without_reopening() {
        let f = Fixture::new();
        let mut writer = f.open(true);
        let mut reader = f.open(true);
        writer.build().unwrap();
        assert!(reader
            .search("sourdough", &opts(Mode::Dense, 5))
            .unwrap()
            .iter()
            .all(|h| h.path != "bread.md"));

        f.write(
            "bread.md",
            "# Bread\n\n## Sourdough\n\nFeed the sourdough starter daily.\n",
        );
        writer.build().unwrap();
        let hits = reader
            .search("sourdough starter", &opts(Mode::Dense, 1))
            .unwrap();
        assert_eq!(hits[0].path, "bread.md");
    }
}
