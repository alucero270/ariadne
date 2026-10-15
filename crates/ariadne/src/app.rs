//! The application module: everything a front end (HTTP, MCP) can ask of Ariadne, validated
//! once, with errors typed by who is at fault.

use ariadne_core::{BuildStats, Hit, Index, IndexStats, Mode, SearchOptions};
use std::fmt;
use std::sync::{Arc, Mutex};

pub const MAX_K: u64 = 20;
pub const MAX_QUERY_CHARS: usize = 500;

#[derive(Debug)]
pub enum AppError {
    /// The caller sent something invalid. The message is safe to show them.
    BadRequest(String),
    /// What the caller asked for doesn't exist. The message is safe to show them.
    NotFound(String),
    /// Ariadne or one of its models failed. Log it; don't show the details to callers.
    Internal(anyhow::Error),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::BadRequest(m) | AppError::NotFound(m) => f.write_str(m),
            AppError::Internal(e) => write!(f, "internal error: {e:#}"),
        }
    }
}

/// A search as callers send it. Every field but the query is optional.
#[derive(Debug, Default)]
pub struct SearchRequest {
    pub query: String,
    pub k: Option<u64>,
    pub mode: Option<String>,
    pub rerank: Option<bool>,
}

impl SearchRequest {
    fn validate(self) -> Result<(String, SearchOptions), AppError> {
        let query = self.query.trim().to_string();
        if query.is_empty() || query.chars().count() > MAX_QUERY_CHARS {
            return Err(AppError::BadRequest(format!(
                "query must be 1 to {MAX_QUERY_CHARS} characters"
            )));
        }
        let k = self.k.unwrap_or(SearchOptions::default().k as u64);
        if !(1..=MAX_K).contains(&k) {
            return Err(AppError::BadRequest(format!("k must be from 1 to {MAX_K}")));
        }
        let mode = match self.mode.as_deref() {
            None => Mode::Hybrid,
            Some(m) => m
                .parse::<Mode>()
                .map_err(|e| AppError::BadRequest(e.to_string()))?,
        };
        let opts = SearchOptions {
            k: k as usize,
            mode,
            rerank: self.rerank.unwrap_or(true),
            ..Default::default()
        };
        Ok((query, opts))
    }
}

/// Two handles on one index: reads go to `reader`, rebuilds to `writer`, so a rebuild never
/// blocks searches.
pub struct App {
    reader: Mutex<Index>,
    writer: Mutex<Index>,
}

impl App {
    pub fn new(reader: Index, writer: Index) -> Arc<Self> {
        Arc::new(Self {
            reader: Mutex::new(reader),
            writer: Mutex::new(writer),
        })
    }

    pub async fn stats(self: &Arc<Self>) -> Result<IndexStats, AppError> {
        self.with_reader(|idx| idx.stats().map_err(AppError::Internal))
            .await
    }

    pub async fn search(self: &Arc<Self>, req: SearchRequest) -> Result<Vec<Hit>, AppError> {
        let (query, opts) = req.validate()?;
        self.with_reader(move |idx| idx.search(&query, &opts).map_err(AppError::Internal))
            .await
    }

    pub async fn read_document(self: &Arc<Self>, path: String) -> Result<String, AppError> {
        self.with_reader(move |idx| {
            idx.read_document(&path)
                .map_err(AppError::Internal)?
                .ok_or_else(|| AppError::NotFound(format!("no indexed document at {path}")))
        })
        .await
    }

    pub async fn reindex(self: &Arc<Self>) -> Result<BuildStats, AppError> {
        let app = self.clone();
        blocking(move || {
            let mut writer = app.writer.lock().map_err(poisoned)?;
            writer.build().map_err(AppError::Internal)
        })
        .await
    }

    async fn with_reader<T, F>(self: &Arc<Self>, f: F) -> Result<T, AppError>
    where
        T: Send + 'static,
        F: FnOnce(&mut Index) -> Result<T, AppError> + Send + 'static,
    {
        let app = self.clone();
        blocking(move || {
            let mut reader = app.reader.lock().map_err(poisoned)?;
            f(&mut reader)
        })
        .await
    }
}

fn poisoned<T>(_: T) -> AppError {
    AppError::Internal(anyhow::anyhow!("index lock poisoned by an earlier panic"))
}

async fn blocking<T, F>(f: F) -> Result<T, AppError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, AppError> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| AppError::Internal(e.into()))?
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use ariadne_core::{Models, Source};
    use std::fs;

    pub struct Fixture {
        pub src: tempfile::TempDir,
        _db: tempfile::TempDir,
        pub app: Arc<App>,
    }

    pub fn fixture() -> Fixture {
        let src = tempfile::tempdir().unwrap();
        fs::create_dir_all(src.path().join("systems/fabrication")).unwrap();
        fs::write(
            src.path().join("systems/fabrication/qidi.md"),
            "# Qidi Plus 4\n\n## Network Placement\n\nMoonraker listens on port 7125.\n",
        )
        .unwrap();
        fs::write(src.path().join("secret.txt"), "not indexed").unwrap();
        let db = tempfile::tempdir().unwrap();
        let open = || {
            Index::open(
                &db.path().join("i.db"),
                Source::new(src.path()),
                Models::default(),
            )
            .unwrap()
        };
        let mut writer = open();
        writer.build().unwrap();
        let app = App::new(open(), writer);
        Fixture { src, _db: db, app }
    }

    fn req(query: &str) -> SearchRequest {
        SearchRequest {
            query: query.into(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn search_validates_once_for_every_front_end() {
        let f = fixture();
        for bad in [
            req("  "),
            req(&"x".repeat(MAX_QUERY_CHARS + 1)),
            SearchRequest {
                k: Some(0),
                ..req("x")
            },
            SearchRequest {
                k: Some(MAX_K + 1),
                ..req("x")
            },
            SearchRequest {
                mode: Some("bogus".into()),
                ..req("x")
            },
        ] {
            assert!(matches!(
                f.app.search(bad).await,
                Err(AppError::BadRequest(_))
            ));
        }
        let hits = f.app.search(req("moonraker port")).await.unwrap();
        assert_eq!(
            hits[0].reference,
            "systems/fabrication/qidi#Network Placement"
        );
    }

    #[tokio::test]
    async fn missing_documents_are_not_found() {
        let f = fixture();
        assert!(f
            .app
            .read_document("systems/fabrication/qidi".into())
            .await
            .unwrap()
            .contains("7125"));
        for path in ["secret.txt", "../etc/passwd", "missing"] {
            assert!(matches!(
                f.app.read_document(path.into()).await,
                Err(AppError::NotFound(_))
            ));
        }
    }

    #[tokio::test]
    async fn reindex_is_visible_to_searches() {
        let f = fixture();
        fs::write(
            f.src.path().join("bread.md"),
            "# Bread\n\n## Sourdough\n\nFeed the starter.\n",
        )
        .unwrap();
        assert_eq!(f.app.reindex().await.unwrap().files_changed, 1);
        assert_eq!(
            f.app.search(req("sourdough starter")).await.unwrap()[0].path,
            "bread.md"
        );
    }

    #[tokio::test]
    async fn searches_run_while_the_writer_is_held() {
        let f = fixture();
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let app = f.app.clone();
        let holder = std::thread::spawn(move || {
            let _writer = app.writer.lock().unwrap();
            locked_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        locked_rx.recv().unwrap();
        let hits = f.app.search(req("moonraker")).await.unwrap();
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        assert!(!hits.is_empty());
    }
}
