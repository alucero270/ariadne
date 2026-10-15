//! SQLite index: FTS5 keyword search plus stored vectors. Derived data, rebuildable from the source.

use crate::chunk::{chunk_markdown, reference};
use crate::models::{Embedder, Reranker};
use crate::source::{resolve_inside, walk_markdown};
use anyhow::{anyhow, bail, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, sha TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL,
  title TEXT NOT NULL,
  heading TEXT NOT NULL,
  anchor TEXT NOT NULL,
  ord INTEGER NOT NULL,
  body TEXT NOT NULL,
  vec BLOB
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(context, body, tokenize='porter unicode61');
";

pub const RRF_K: f64 = 60.0;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "should", "so", "that",
    "the", "their", "there", "this", "to", "was", "we", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Hybrid,
    Bm25,
    Dense,
}

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
    pub k: usize,
    pub mode: Mode,
    pub candidates: usize,
    pub rerank: bool,
    pub max_per_file: Option<usize>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            k: 8,
            mode: Mode::Hybrid,
            candidates: 40,
            rerank: true,
            max_per_file: Some(2),
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
    #[serde(rename = "ref")]
    pub reference: String,
    pub path: String,
    pub title: String,
    pub heading: String,
    pub text: String,
    pub score: f64,
    pub ranks: HashMap<String, usize>,
}

pub fn fts_query(query: &str) -> String {
    let lower = query.to_lowercase();
    let mut seen: Vec<&str> = Vec::new();
    for term in lower.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if !term.is_empty() && !STOPWORDS.contains(&term) && !seen.contains(&term) {
            seen.push(term);
        }
    }
    seen.iter()
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

pub fn rrf(rankings: &[Vec<i64>]) -> HashMap<i64, f64> {
    let mut scores = HashMap::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }
    scores
}

fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_blob(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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

pub struct Index {
    conn: Connection,
    embedder: Option<Box<dyn Embedder>>,
    reranker: Option<Box<dyn Reranker>>,
    matrix: Option<(Vec<i64>, Vec<Vec<f32>>)>,
}

impl Index {
    pub fn open(
        db: &Path,
        embedder: Option<Box<dyn Embedder>>,
        reranker: Option<Box<dyn Reranker>>,
    ) -> Result<Self> {
        let conn = Connection::open(db)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn,
            embedder,
            reranker,
            matrix: None,
        })
    }

    fn meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0))
            .optional()?)
    }

    fn embedder_name(&self) -> String {
        self.embedder
            .as_ref()
            .map(|e| e.name())
            .unwrap_or_else(|| "none".into())
    }

    pub fn root(&self) -> Result<Option<PathBuf>> {
        Ok(self.meta("root")?.map(PathBuf::from))
    }

    pub fn build(
        &mut self,
        root: &Path,
        excludes: &[String],
        max_chars: usize,
    ) -> Result<BuildStats> {
        let start = Instant::now();
        let root = root.canonicalize()?;
        let files = walk_markdown(&root, excludes)?;
        let embedder_name = self.embedder_name();
        let full = self.meta("embedder")?.as_deref() != Some(embedder_name.as_str())
            || self.meta("max_chars")?.as_deref() != Some(max_chars.to_string().as_str())
            || self.meta("root")?.as_deref() != Some(root.to_string_lossy().as_ref());

        let tx = self.conn.transaction()?;
        if full {
            tx.execute_batch("DELETE FROM chunks; DELETE FROM chunks_fts; DELETE FROM files;")?;
        }
        let known: HashMap<String, String> = {
            let mut stmt = tx.prepare("SELECT path, sha FROM files")?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        let mut stats = BuildStats {
            files_seen: files.len(),
            ..Default::default()
        };
        let mut seen = HashSet::new();

        for file in &files {
            seen.insert(file.rel.clone());
            let raw = std::fs::read(&file.abs)?;
            let sha = format!("{:x}", Sha256::digest(&raw));
            if known.get(&file.rel) == Some(&sha) {
                continue;
            }
            stats.files_changed += 1;
            delete_path(&tx, &file.rel)?;
            let chunks = chunk_markdown(&file.rel, &String::from_utf8_lossy(&raw), max_chars);
            let vecs = match (&self.embedder, chunks.is_empty()) {
                (Some(e), false) => {
                    Some(e.embed(&chunks.iter().map(|c| c.embed_text()).collect::<Vec<_>>())?)
                }
                _ => None,
            };
            for (i, c) in chunks.iter().enumerate() {
                let blob = vecs.as_ref().map(|v| to_blob(&v[i]));
                tx.execute(
                    "INSERT INTO chunks(path, title, heading, anchor, ord, body, vec) VALUES (?1,?2,?3,?4,?5,?6,?7)",
                    params![c.path, c.title, c.heading, c.anchor, c.ord as i64, c.body, blob],
                )?;
                let id = tx.last_insert_rowid();
                tx.execute(
                    "INSERT INTO chunks_fts(rowid, context, body) VALUES (?1, ?2, ?3)",
                    params![id, format!("{} {} {}", c.title, c.heading, c.path), c.body],
                )?;
            }
            stats.chunks_written += chunks.len();
            tx.execute(
                "INSERT INTO files(path, sha) VALUES (?1, ?2)",
                params![file.rel, sha],
            )?;
        }
        for gone in known.keys().filter(|p| !seen.contains(*p)) {
            delete_path(&tx, gone)?;
            stats.files_removed += 1;
        }
        for (k, v) in [
            ("embedder", embedder_name),
            ("max_chars", max_chars.to_string()),
            ("root", root.to_string_lossy().into_owned()),
            ("indexed_at", now_utc()),
        ] {
            tx.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?1, ?2)",
                params![k, v],
            )?;
        }
        tx.commit()?;
        self.matrix = None;
        stats.seconds = (start.elapsed().as_secs_f64() * 100.0).round() / 100.0;
        Ok(stats)
    }

    pub fn stats(&self) -> Result<IndexStats> {
        Ok(IndexStats {
            files: self
                .conn
                .query_row("SELECT count(*) FROM files", [], |r| r.get(0))?,
            chunks: self
                .conn
                .query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?,
            embedder: self.meta("embedder")?,
            reranker: self.reranker.as_ref().map(|r| r.name()),
            indexed_at: self.meta("indexed_at")?,
        })
    }

    fn bm25(&self, query: &str, limit: usize) -> Result<Vec<i64>> {
        let q = fts_query(query);
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1 ORDER BY bm25(chunks_fts, 3.0, 1.0) LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![q, limit as i64], |r| r.get(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    fn dense(&mut self, query: &str, limit: usize) -> Result<Vec<i64>> {
        let Some(embedder) = &self.embedder else {
            return Ok(Vec::new());
        };
        if self.meta("embedder")?.as_deref() != Some(embedder.name().as_str()) {
            return Ok(Vec::new());
        }
        if self.matrix.is_none() {
            let mut stmt = self
                .conn
                .prepare("SELECT id, vec FROM chunks WHERE vec IS NOT NULL ORDER BY id")?;
            let rows =
                stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
            let (mut ids, mut vecs) = (Vec::new(), Vec::new());
            for row in rows {
                let (id, blob) = row?;
                ids.push(id);
                vecs.push(from_blob(&blob));
            }
            self.matrix = Some((ids, vecs));
        }
        let (ids, vecs) = self.matrix.as_ref().expect("matrix loaded above");
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let qv = embedder
            .embed(&[query.to_string()])?
            .pop()
            .ok_or_else(|| anyhow!("embedder returned no vector for the query"))?;
        let mut scored: Vec<(f32, i64)> = vecs
            .iter()
            .zip(ids)
            .map(|(v, id)| (v.iter().zip(&qv).map(|(a, b)| a * b).sum(), *id))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        Ok(scored.into_iter().take(limit).map(|(_, id)| id).collect())
    }

    pub fn search(&mut self, query: &str, opts: &SearchOptions) -> Result<Vec<Hit>> {
        let mut rankings: Vec<(&str, Vec<i64>)> = Vec::new();
        if matches!(opts.mode, Mode::Hybrid | Mode::Bm25) {
            rankings.push(("bm25", self.bm25(query, opts.candidates)?));
        }
        if matches!(opts.mode, Mode::Hybrid | Mode::Dense) {
            rankings.push(("dense", self.dense(query, opts.candidates)?));
        }
        let fused = rrf(&rankings
            .iter()
            .filter(|(_, r)| !r.is_empty())
            .map(|(_, r)| r.clone())
            .collect::<Vec<_>>());
        let mut order: Vec<i64> = fused.keys().copied().collect();
        order.sort_by(|a, b| fused[b].total_cmp(&fused[a]).then(a.cmp(b)));
        order.truncate(opts.k.max(20));

        let mut hits = Vec::with_capacity(order.len());
        {
            let mut stmt = self
                .conn
                .prepare("SELECT path, title, heading, anchor, body FROM chunks WHERE id=?1")?;
            for id in &order {
                let (path, title, heading, anchor, body): (String, String, String, String, String) =
                    stmt.query_row([id], |r| {
                        Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
                    })?;
                let ranks = rankings
                    .iter()
                    .filter_map(|(name, r)| {
                        r.iter()
                            .position(|x| x == id)
                            .map(|p| (name.to_string(), p + 1))
                    })
                    .collect();
                hits.push(Hit {
                    id: *id,
                    reference: reference(&path, &anchor),
                    path,
                    title,
                    heading,
                    text: body,
                    score: fused[id],
                    ranks,
                });
            }
        }

        if opts.rerank && !hits.is_empty() {
            if let Some(reranker) = &self.reranker {
                let docs: Vec<String> = hits
                    .iter()
                    .map(|h| format!("{} > {}\n\n{}", h.title, h.heading, h.text))
                    .collect();
                let scores = reranker.rerank(query, &docs)?;
                for (h, s) in hits.iter_mut().zip(scores) {
                    h.score = f64::from(s);
                }
                hits.sort_by(|a, b| b.score.total_cmp(&a.score));
            }
        }
        if let Some(cap) = opts.max_per_file {
            let mut per_file: HashMap<String, usize> = HashMap::new();
            hits.retain(|h| {
                let n = per_file.entry(h.path.clone()).or_insert(0);
                *n += 1;
                *n <= cap
            });
        }
        hits.truncate(opts.k);
        Ok(hits)
    }

    /// Read a whole indexed document. Only files currently in the index are readable.
    pub fn read_document(&self, rel: &str) -> Result<String> {
        let rel = rel.trim_start_matches('/');
        let rel = if rel.ends_with(".md") {
            rel.to_string()
        } else {
            format!("{rel}.md")
        };
        let indexed: Option<String> = self
            .conn
            .query_row("SELECT path FROM files WHERE path=?1", [&rel], |r| r.get(0))
            .optional()?;
        if indexed.is_none() {
            bail!("not an indexed document: {rel}");
        }
        let root = self
            .root()?
            .ok_or_else(|| anyhow!("index has no source root"))?;
        Ok(std::fs::read_to_string(resolve_inside(&root, &rel)?)?)
    }
}

fn delete_path(tx: &rusqlite::Transaction, path: &str) -> Result<()> {
    tx.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE path=?1)",
        [path],
    )?;
    tx.execute("DELETE FROM chunks WHERE path=?1", [path])?;
    tx.execute("DELETE FROM files WHERE path=?1", [path])?;
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::source::DEFAULT_EXCLUDES;
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
                        let h = Sha256::digest(tok.as_bytes());
                        v[usize::from(h[0]) % 64] += 1.0;
                    }
                    crate::models::normalize(v)
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

    pub fn repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("systems/fabrication")).unwrap();
        fs::write(dir.path().join("systems/fabrication/qidi.md"), PRINTER).unwrap();
        fs::write(dir.path().join("voice.md"), VOICE).unwrap();
        dir
    }

    pub fn excludes() -> Vec<String> {
        DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect()
    }

    fn opts(mode: Mode, k: usize) -> SearchOptions {
        SearchOptions {
            k,
            mode,
            ..Default::default()
        }
    }

    #[test]
    fn fts_query_sanitizes() {
        assert_eq!(
            fts_query("Which ports does the \"printer\" expose?"),
            "\"ports\" OR \"printer\" OR \"expose\""
        );
        assert_eq!(fts_query("the of and"), "");
        assert_eq!(fts_query("llama-swap llama"), "\"llama\" OR \"swap\"");
    }

    #[test]
    fn rrf_rewards_agreement() {
        let s = rrf(&[vec![1, 2, 3], vec![3, 2, 4]]);
        assert!(s[&2] > s[&1] && s[&2] > s[&4]);
    }

    #[test]
    fn timestamp_format() {
        let t = now_utc();
        assert_eq!(t.len(), 20);
        assert!(t.starts_with("20") && t.ends_with('Z'));
    }

    #[test]
    fn build_and_search_all_modes() {
        let src = repo();
        let db = tempfile::tempdir().unwrap();
        let mut idx =
            Index::open(&db.path().join("i.db"), Some(Box::new(HashEmbedder)), None).unwrap();
        let stats = idx.build(src.path(), &excludes(), 1200).unwrap();
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
    fn incremental_reindex_and_removal() {
        let src = repo();
        let db = tempfile::tempdir().unwrap();
        let mut idx =
            Index::open(&db.path().join("i.db"), Some(Box::new(HashEmbedder)), None).unwrap();
        idx.build(src.path(), &excludes(), 1200).unwrap();
        assert_eq!(
            idx.build(src.path(), &excludes(), 1200)
                .unwrap()
                .files_changed,
            0
        );

        fs::write(
            src.path().join("voice.md"),
            format!("{VOICE}\n## Barge-in\n\nInterruptions are supported.\n"),
        )
        .unwrap();
        assert_eq!(
            idx.build(src.path(), &excludes(), 1200)
                .unwrap()
                .files_changed,
            1
        );
        assert_eq!(
            idx.search("interruptions", &opts(Mode::Hybrid, 1)).unwrap()[0].heading,
            "Barge-in"
        );

        fs::remove_file(src.path().join("voice.md")).unwrap();
        assert_eq!(
            idx.build(src.path(), &excludes(), 1200)
                .unwrap()
                .files_removed,
            1
        );
        assert!(idx
            .search("whisper", &opts(Mode::Hybrid, 5))
            .unwrap()
            .iter()
            .all(|h| h.path != "voice.md"));
        let fts: i64 = idx
            .conn
            .query_row("SELECT count(*) FROM chunks_fts", [], |r| r.get(0))
            .unwrap();
        assert_eq!(fts, idx.stats().unwrap().chunks);
    }

    #[test]
    fn embedder_change_forces_full_rebuild() {
        let src = repo();
        let db = tempfile::tempdir().unwrap();
        Index::open(&db.path().join("i.db"), None, None)
            .unwrap()
            .build(src.path(), &excludes(), 1200)
            .unwrap();
        let mut idx =
            Index::open(&db.path().join("i.db"), Some(Box::new(HashEmbedder)), None).unwrap();
        assert_eq!(
            idx.build(src.path(), &excludes(), 1200)
                .unwrap()
                .files_changed,
            2
        );
        assert!(!idx
            .search("whisper", &opts(Mode::Dense, 1))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn keyword_only_without_embedder() {
        let src = repo();
        let db = tempfile::tempdir().unwrap();
        let mut idx = Index::open(&db.path().join("i.db"), None, None).unwrap();
        idx.build(src.path(), &excludes(), 1200).unwrap();
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
    fn reranker_reorders_and_file_cap_applies() {
        let src = repo();
        let db = tempfile::tempdir().unwrap();
        let mut idx = Index::open(
            &db.path().join("i.db"),
            Some(Box::new(HashEmbedder)),
            Some(Box::new(ReverseReranker)),
        )
        .unwrap();
        idx.build(src.path(), &excludes(), 1200).unwrap();
        let base = SearchOptions {
            k: 20,
            max_per_file: None,
            ..Default::default()
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
        let rev: Vec<_> = plain.iter().rev().map(|h| h.id).collect();
        assert_eq!(reranked.iter().map(|h| h.id).collect::<Vec<_>>(), rev);

        let capped = idx
            .search(
                "printer vlan moonraker whisper",
                &SearchOptions {
                    k: 10,
                    max_per_file: Some(1),
                    ..Default::default()
                },
            )
            .unwrap();
        let paths: HashSet<_> = capped.iter().map(|h| h.path.clone()).collect();
        assert_eq!(paths.len(), capped.len());
    }

    #[test]
    fn read_document_only_reads_indexed_files_inside_root() {
        let src = repo();
        fs::write(src.path().join("notes.txt"), "not indexed").unwrap();
        let db = tempfile::tempdir().unwrap();
        let mut idx = Index::open(&db.path().join("i.db"), None, None).unwrap();
        idx.build(src.path(), &excludes(), 1200).unwrap();
        assert!(idx
            .read_document("systems/fabrication/qidi")
            .unwrap()
            .contains("Moonraker"));
        assert!(idx.read_document("/voice.md").unwrap().contains("Whisper"));
        for bad in ["notes.txt", "../etc/passwd", "missing.md"] {
            assert!(idx.read_document(bad).is_err(), "{bad} should be refused");
        }
    }
}
