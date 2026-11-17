//! The application module: everything a front end (CLI, HTTP, MCP) can ask of Ariadne. It owns
//! the defaults and limits, validates once, and types errors by who is at fault. Front ends
//! only choose how to present results.

use ariadne_core::{
    BuildStats, EvalReport, EvalSet, FileState, FileStatus, Hit, Index, IndexStats, Mode,
    SearchOptions,
};
use serde::Serialize;
use std::fmt;
use std::sync::{Arc, Mutex};

pub const DEFAULT_K: u64 = 5;
pub const MAX_K: u64 = 20;
pub const DEFAULT_FILE_LIMIT: u64 = 200;
pub const MAX_FILE_LIMIT: u64 = 1000;
pub const MAX_QUERY_CHARS: usize = 500;
pub const MAX_PATH_CHARS: usize = 400;
/// Served searches skip the reranker unless the caller asks: on a CPU reranker it costs seconds
/// per query and, on the Pantheon eval, bought no recall (#143). `ariadne eval` reports both.
const DEFAULT_RERANK: bool = false;

#[derive(Debug)]
pub enum AppError {
    /// The caller sent something invalid. The message is safe to show them.
    BadRequest(String),
    /// What the caller asked for doesn't exist. The message is safe to show them.
    NotFound(String),
    /// Ariadne or one of its models failed. Log it; don't show the details to callers.
    Internal(anyhow::Error),
}

impl std::error::Error for AppError {}

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
    /// `"file"` to return the rest of the top hit's file when it is short (parent-document
    /// retrieval); omitted or `"none"` for single sections.
    pub expand: Option<String>,
}

impl SearchRequest {
    fn validate(self) -> Result<(String, SearchOptions), AppError> {
        let query = self.query.trim().to_string();
        if query.is_empty() || query.chars().count() > MAX_QUERY_CHARS {
            return Err(AppError::BadRequest(format!(
                "query must be 1 to {MAX_QUERY_CHARS} characters"
            )));
        }
        let k = validate_k(self.k)?;
        let mode = match self.mode.as_deref() {
            None => Mode::Hybrid,
            Some(m) => m
                .parse::<Mode>()
                .map_err(|e| AppError::BadRequest(e.to_string()))?,
        };
        let expand_file = match self.expand.as_deref() {
            None | Some("none") => false,
            Some("file") => true,
            Some(other) => {
                return Err(AppError::BadRequest(format!(
                    "unknown expand {other}; use file or none"
                )))
            }
        };
        let opts = SearchOptions::new(k, mode, self.rerank.unwrap_or(DEFAULT_RERANK))
            .with_expand_file(expand_file);
        Ok((query, opts))
    }
}

/// Parse a `status` filter value (`indexed` / `skipped`) into a [`FileState`].
pub fn parse_file_status(s: &str) -> Result<FileState, AppError> {
    match s {
        "indexed" => Ok(FileState::Indexed),
        "skipped" => Ok(FileState::Skipped),
        other => Err(AppError::BadRequest(format!(
            "unknown status {other}; use indexed or skipped"
        ))),
    }
}

/// `limit` for the file listings: default 200, at least 1, values above [`MAX_FILE_LIMIT`]
/// are clamped rather than refused.
fn validate_file_limit(limit: Option<u64>) -> Result<u64, AppError> {
    let limit = limit.unwrap_or(DEFAULT_FILE_LIMIT);
    if limit == 0 {
        return Err(AppError::BadRequest(format!(
            "limit must be from 1 to {MAX_FILE_LIMIT}"
        )));
    }
    Ok(limit.min(MAX_FILE_LIMIT))
}

fn validate_k(k: Option<u64>) -> Result<usize, AppError> {
    let k = k.unwrap_or(DEFAULT_K);
    if !(1..=MAX_K).contains(&k) {
        return Err(AppError::BadRequest(format!("k must be from 1 to {MAX_K}")));
    }
    Ok(k as usize)
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
        if path.is_empty() || path.chars().count() > MAX_PATH_CHARS {
            return Err(AppError::BadRequest(format!(
                "path must be 1 to {MAX_PATH_CHARS} characters"
            )));
        }
        self.with_reader(move |idx| {
            idx.read_document(&path)
                .map_err(AppError::Internal)?
                .ok_or_else(|| AppError::NotFound(format!("no indexed document at {path}")))
        })
        .await
    }

    /// Every file the index knows about (indexed and skipped), filtered by status and path
    /// prefix, sorted by path, then paginated: `total` counts the filtered list before
    /// pagination, `files` is the `[offset .. offset + limit]` slice of it. Shared by the
    /// HTTP `/files` route and the MCP `list_documents` tool.
    pub async fn files(
        self: &Arc<Self>,
        status: Option<FileState>,
        prefix: Option<String>,
        limit: Option<u64>,
        offset: Option<u64>,
    ) -> Result<FileList, AppError> {
        let limit = validate_file_limit(limit)?;
        let offset = offset.unwrap_or(0);
        self.with_reader(move |idx| {
            let filtered: Vec<FileStatus> = idx
                .files()
                .map_err(AppError::Internal)?
                .into_iter()
                .filter(|f| {
                    let status_ok = match status {
                        None => true,
                        Some(s) => f.status == s,
                    };
                    let prefix_ok = match prefix.as_deref() {
                        None => true,
                        Some(p) => f.path.starts_with(p),
                    };
                    status_ok && prefix_ok
                })
                .collect();
            let total = filtered.len() as u64;
            let files = filtered
                .into_iter()
                .skip(offset as usize)
                .take(limit as usize)
                .collect();
            Ok(FileList { total, files })
        })
        .await
    }

    /// Measure retrieval in every mode, plus hybrid with the reranker when one is configured.
    pub async fn evaluate(
        self: &Arc<Self>,
        set: EvalSet,
        k: Option<u64>,
    ) -> Result<EvalReport, AppError> {
        let k = validate_k(k)?;
        self.with_reader(move |idx| idx.evaluate(&set, k).map_err(AppError::Internal))
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

/// One page of the file listing, as the HTTP `/files` route answers it.
#[derive(Debug, Serialize)]
pub struct FileList {
    /// Files matching the status and prefix filters, before pagination.
    pub total: u64,
    pub files: Vec<FileStatus>,
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

    /// Like `fixture`, with a made-up EPUB beside the markdown and the `md,epub` formats
    /// enabled, so the front ends can be tested against a non-Markdown document (#154).
    pub fn fixture_with_epub() -> Fixture {
        let src = tempfile::tempdir().unwrap();
        fs::create_dir_all(src.path().join("systems/fabrication")).unwrap();
        fs::write(
            src.path().join("systems/fabrication/qidi.md"),
            "# Qidi Plus 4\n\n## Network Placement\n\nMoonraker listens on port 7125.\n",
        )
        .unwrap();
        ariadne_core::epub::fixtures::write_epub(
            src.path(),
            "Library/Some Book.epub",
            &ariadne_core::epub::fixtures::epub3_nav(),
        );
        let db = tempfile::tempdir().unwrap();
        let config = crate::config::Config::new(crate::config::Settings {
            root: Some(src.path().to_path_buf()),
            db: db.path().join("i.db"),
            formats: Some("md,epub".into()),
            ..Default::default()
        })
        .unwrap();
        let (app, stats) = config.open_app().unwrap();
        assert_eq!(stats.files_changed, 2, "the markdown and the book");
        Fixture { src, _db: db, app }
    }

    fn req(query: &str) -> SearchRequest {
        SearchRequest {
            query: query.into(),
            ..Default::default()
        }
    }

    /// Like `fixture`, plus an unreadable `.md` the last build recorded as skipped, and a
    /// second directory so prefix filters have something to bite on.
    #[cfg(unix)]
    pub fn fixture_with_skipped() -> Fixture {
        use std::os::unix::fs::PermissionsExt;
        let src = tempfile::tempdir().unwrap();
        fs::create_dir_all(src.path().join("systems/fabrication")).unwrap();
        fs::write(
            src.path().join("systems/fabrication/qidi.md"),
            "# Qidi Plus 4\n\n## Network Placement\n\nMoonraker listens on port 7125.\n",
        )
        .unwrap();
        fs::write(
            src.path().join("systems/locked.md"),
            "# Locked\n\nVault passphrase.\n",
        )
        .unwrap();
        fs::create_dir_all(src.path().join("tools")).unwrap();
        fs::write(
            src.path().join("tools/bench.md"),
            "# Bench\n\n## Router\n\nClamp the panel.\n",
        )
        .unwrap();
        fs::set_permissions(
            src.path().join("systems/locked.md"),
            fs::Permissions::from_mode(0o0),
        )
        .unwrap();
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

    #[tokio::test]
    async fn file_limits_are_validated_and_clamped() {
        assert_eq!(validate_file_limit(None).unwrap(), DEFAULT_FILE_LIMIT);
        assert_eq!(
            validate_file_limit(Some(MAX_FILE_LIMIT + 1)).unwrap(),
            MAX_FILE_LIMIT
        );
        assert_eq!(validate_file_limit(Some(1)).unwrap(), 1);
        assert!(matches!(
            validate_file_limit(Some(0)),
            Err(AppError::BadRequest(_))
        ));
        assert_eq!(parse_file_status("indexed").unwrap(), FileState::Indexed);
        assert_eq!(parse_file_status("skipped").unwrap(), FileState::Skipped);
        for bad in ["bogus", "", "Indexed"] {
            assert!(matches!(
                parse_file_status(bad),
                Err(AppError::BadRequest(_))
            ));
        }
        let f = fixture();
        assert!(matches!(
            f.app.files(None, None, Some(0), None).await,
            Err(AppError::BadRequest(_))
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn files_lists_indexed_and_skipped_with_filters() {
        let f = fixture_with_skipped();
        let all = f.app.files(None, None, None, None).await.unwrap();
        assert_eq!(all.total, 3);
        let paths: Vec<_> = all.files.iter().map(|x| x.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "systems/fabrication/qidi.md",
                "systems/locked.md",
                "tools/bench.md"
            ]
        );
        assert_eq!(all.files[0].status, FileState::Indexed);
        assert!(all.files[0].chunks > 0 && all.files[0].reason.is_none());
        assert_eq!(all.files[1].status, FileState::Skipped);
        assert!(all.files[1].reason.is_some());

        let skipped = f
            .app
            .files(Some(FileState::Skipped), None, None, None)
            .await
            .unwrap();
        assert_eq!(skipped.total, 1);
        assert_eq!(skipped.files[0].path, "systems/locked.md");
        let indexed = f
            .app
            .files(Some(FileState::Indexed), None, None, None)
            .await
            .unwrap();
        assert_eq!(indexed.total, 2);

        let under = f
            .app
            .files(None, Some("systems/".into()), None, None)
            .await
            .unwrap();
        assert_eq!(under.total, 2);
        let none = f
            .app
            .files(Some(FileState::Skipped), Some("tools/".into()), None, None)
            .await
            .unwrap();
        assert_eq!(none.total, 0);
        assert!(none.files.is_empty());

        // Pagination slices the filtered, sorted list; total is the pre-pagination count.
        let page = f.app.files(None, None, Some(2), Some(1)).await.unwrap();
        assert_eq!(page.total, 3);
        assert_eq!(page.files.len(), 2);
        assert_eq!(page.files[0].path, "systems/locked.md");
        let past_end = f.app.files(None, None, Some(2), Some(3)).await.unwrap();
        assert_eq!(past_end.total, 3);
        assert!(past_end.files.is_empty());
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
            SearchRequest {
                expand: Some("bogus".into()),
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
    async fn evaluate_measures_with_the_served_defaults() {
        let f = fixture();
        let set: EvalSet = serde_json::from_str(
            r#"{"questions":[{"q":"moonraker","expect":["systems/fabrication/qidi.md"]}]}"#,
        )
        .unwrap();
        let report = f.app.evaluate(set, None).await.unwrap();
        assert_eq!(report.k, DEFAULT_K as usize);
        assert!(
            !report.modes.contains_key("dense"),
            "no embedder configured"
        );
        assert_eq!(report.modes["hybrid"].recall_at_k, 1.0);
    }

    #[tokio::test]
    async fn overlong_paths_are_bad_requests() {
        let f = fixture();
        let long = "a/".repeat(MAX_PATH_CHARS);
        assert!(matches!(
            f.app.read_document(long).await,
            Err(AppError::BadRequest(_))
        ));
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
    async fn an_epub_reads_as_rendered_text_and_not_as_bytes() {
        let f = fixture_with_epub();
        let text = f
            .app
            .read_document("Library/Some Book.epub".into())
            .await
            .unwrap();
        assert!(text.starts_with("# Replication Basics"), "{text}");
        assert!(text.contains("## Chapter 5 Replication"), "{text}");
        assert!(text.contains("Print a calibration cube first."), "{text}");
        assert!(!text.bytes().any(|b| b == 0), "container bytes: {text:?}");

        // Markdown is still its source, and an extension-less epub path is not a document.
        assert!(f
            .app
            .read_document("systems/fabrication/qidi".into())
            .await
            .unwrap()
            .starts_with("# Qidi Plus 4"));
        for path in ["Library/Some Book", "../etc/passwd", "missing"] {
            assert!(
                matches!(
                    f.app.read_document(path.into()).await,
                    Err(AppError::NotFound(_))
                ),
                "{path}"
            );
        }

        // `expand=file` works on an epub hit: the top section carries the rest of the book.
        let hits = f
            .app
            .search(SearchRequest {
                query: "calibration cube".into(),
                expand: Some("file".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(hits[0].path, "Library/Some Book.epub");
        assert!(
            hits[0]
                .text
                .contains("[also in this file: Library/Some Book.epub#Chapter 1 Setup]"),
            "{}",
            hits[0].text
        );
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
