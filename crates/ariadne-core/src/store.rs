//! SQLite persistence for the index. Private to the crate: `Index` is the only caller.

use crate::chunk::Chunk;
use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

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

/// A stored chunk as search needs it.
pub(crate) struct StoredChunk {
    pub path: String,
    pub title: String,
    pub heading: String,
    pub anchor: String,
    pub body: String,
}

/// Open the database in WAL mode so one writer and many readers don't block each other.
pub(crate) fn open(db: &Path) -> Result<Connection> {
    let conn = Connection::open(db)?;
    conn.busy_timeout(Duration::from_secs(10))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.execute_batch(SCHEMA)?;
    Ok(conn)
}

pub(crate) fn meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0))
        .optional()?)
}

pub(crate) fn set_meta(tx: &Transaction, key: &str, value: &str) -> Result<()> {
    tx.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?1, ?2)",
        params![key, value],
    )?;
    Ok(())
}

pub(crate) fn clear(tx: &Transaction) -> Result<()> {
    tx.execute_batch("DELETE FROM chunks; DELETE FROM chunks_fts; DELETE FROM files;")?;
    Ok(())
}

pub(crate) fn file_hashes(conn: &Connection) -> Result<HashMap<String, String>> {
    let mut stmt = conn.prepare("SELECT path, sha FROM files")?;
    let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub(crate) fn is_indexed(conn: &Connection, path: &str) -> Result<bool> {
    Ok(conn
        .query_row("SELECT 1 FROM files WHERE path=?1", [path], |_| Ok(()))
        .optional()?
        .is_some())
}

pub(crate) fn delete_file(tx: &Transaction, path: &str) -> Result<()> {
    tx.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE path=?1)",
        [path],
    )?;
    tx.execute("DELETE FROM chunks WHERE path=?1", [path])?;
    tx.execute("DELETE FROM files WHERE path=?1", [path])?;
    Ok(())
}

/// Replace a file's chunks. `vectors`, when present, holds one vector per chunk.
pub(crate) fn write_file(
    tx: &Transaction,
    path: &str,
    sha: &str,
    chunks: &[Chunk],
    vectors: Option<&[Vec<f32>]>,
) -> Result<()> {
    delete_file(tx, path)?;
    for (i, c) in chunks.iter().enumerate() {
        let blob = vectors.map(|v| to_blob(&v[i]));
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
    tx.execute(
        "INSERT INTO files(path, sha) VALUES (?1, ?2)",
        params![path, sha],
    )?;
    Ok(())
}

/// Chunk ids best-first by BM25, weighting the title/heading column 3x the body.
pub(crate) fn keyword_ids(conn: &Connection, fts_query: &str, limit: usize) -> Result<Vec<i64>> {
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1 ORDER BY bm25(chunks_fts, 3.0, 1.0) LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![fts_query, limit as i64], |r| r.get(0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

pub(crate) fn vectors(conn: &Connection) -> Result<(Vec<i64>, Vec<Vec<f32>>)> {
    let mut stmt = conn.prepare("SELECT id, vec FROM chunks WHERE vec IS NOT NULL ORDER BY id")?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
    let (mut ids, mut vecs) = (Vec::new(), Vec::new());
    for row in rows {
        let (id, blob) = row?;
        ids.push(id);
        vecs.push(from_blob(&blob));
    }
    Ok((ids, vecs))
}

pub(crate) fn chunk(conn: &Connection, id: i64) -> Result<StoredChunk> {
    Ok(conn.query_row(
        "SELECT path, title, heading, anchor, body FROM chunks WHERE id=?1",
        [id],
        |r| {
            Ok(StoredChunk {
                path: r.get(0)?,
                title: r.get(1)?,
                heading: r.get(2)?,
                anchor: r.get(3)?,
                body: r.get(4)?,
            })
        },
    )?)
}

/// Changes whenever another connection commits; stable across this connection's own writes.
pub(crate) fn data_version(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("PRAGMA data_version", [], |r| r.get(0))?)
}

pub(crate) fn counts(conn: &Connection) -> Result<(i64, i64)> {
    Ok((
        conn.query_row("SELECT count(*) FROM files", [], |r| r.get(0))?,
        conn.query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?,
    ))
}

fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_blob(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
