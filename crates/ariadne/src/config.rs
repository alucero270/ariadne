//! Configuration: the settings every command shares, validated once, and the wiring that turns
//! them into a ready index or app.

use crate::app::App;
use crate::upstream::{HttpContextualizer, HttpEmbedder, HttpReranker};
use anyhow::{bail, Context, Result};
use ariadne_core::{BuildStats, Index, Models, Source};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

const MIN_TOKEN_CHARS: usize = 16;
/// Well under the HTTP layer's 4 KiB cap on the Authorization header, so a valid token can
/// always be sent.
const MAX_TOKEN_CHARS: usize = 1024;

/// Flags and environment variables shared by every command.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct Settings {
    /// Folder to index. Read-only; nothing outside it is read and symlinks are skipped.
    #[arg(long, env = "ARIADNE_ROOT", global = true)]
    pub root: Option<PathBuf>,
    /// SQLite index file. Derived data: delete it and it rebuilds from the root.
    #[arg(long, env = "ARIADNE_DB", default_value = "ariadne.db", global = true)]
    pub db: PathBuf,
    /// OpenAI-compatible base URL for embeddings (e.g. llama-swap). Unset means keyword search only.
    #[arg(long, env = "ARIADNE_EMBED_URL", global = true)]
    pub embed_url: Option<String>,
    #[arg(long, env = "ARIADNE_EMBED_MODEL", global = true)]
    pub embed_model: Option<String>,
    /// Base URL of a llama.cpp-style /v1/rerank endpoint. Unset disables reranking.
    #[arg(long, env = "ARIADNE_RERANK_URL", global = true)]
    pub rerank_url: Option<String>,
    #[arg(long, env = "ARIADNE_RERANK_MODEL", global = true)]
    pub rerank_model: Option<String>,
    /// OpenAI-compatible base URL for a chat model that writes a short context for each chunk at
    /// index time (contextual retrieval). Unset disables it.
    #[arg(long, env = "ARIADNE_CONTEXT_URL", global = true)]
    pub context_url: Option<String>,
    #[arg(long, env = "ARIADNE_CONTEXT_MODEL", global = true)]
    pub context_model: Option<String>,
    /// Chat requests in flight at once while indexing.
    #[arg(long, env = "ARIADNE_CONTEXT_WORKERS", global = true)]
    pub context_workers: Option<usize>,
    /// Bearer token sent to the embedding, rerank, and context endpoints.
    #[arg(
        long,
        env = "ARIADNE_UPSTREAM_KEY",
        hide_env_values = true,
        global = true,
        default_value = ""
    )]
    pub upstream_key: String,
    /// Largest chunk, in characters, before a section is split at paragraph boundaries.
    #[arg(long, global = true)]
    pub max_chars: Option<usize>,
}

/// An OpenAI-compatible endpoint and the model to ask for.
#[derive(Debug, Clone, PartialEq)]
struct Endpoint {
    url: String,
    model: String,
}

/// Validated settings.
#[derive(Debug, Clone)]
pub struct Config {
    db: PathBuf,
    source: Source,
    embedder: Option<Endpoint>,
    reranker: Option<Endpoint>,
    contextualizer: Option<Endpoint>,
    context_workers: usize,
    upstream_key: String,
}

impl Config {
    pub fn new(settings: Settings) -> Result<Self> {
        let root = settings
            .root
            .context("--root (or ARIADNE_ROOT) is required")?;
        let source = match settings.max_chars {
            Some(n) if n < 100 => bail!("--max-chars must be at least 100"),
            Some(n) => Source::new(root).with_max_chars(n),
            None => Source::new(root),
        };
        Ok(Self {
            db: settings.db,
            source,
            embedder: endpoint(settings.embed_url, settings.embed_model, "EMBED")?,
            reranker: endpoint(settings.rerank_url, settings.rerank_model, "RERANK")?,
            contextualizer: endpoint(settings.context_url, settings.context_model, "CONTEXT")?,
            context_workers: match settings.context_workers {
                Some(n) if !(1..=32).contains(&n) => bail!("--context-workers must be 1 to 32"),
                Some(n) => n,
                None => 4,
            },
            upstream_key: settings.upstream_key,
        })
    }

    fn models(&self) -> Models {
        let key = &self.upstream_key;
        Models {
            embedder: self
                .embedder
                .as_ref()
                .map(|e| Box::new(HttpEmbedder::new(&e.url, &e.model, key)) as _),
            reranker: self
                .reranker
                .as_ref()
                .map(|r| Box::new(HttpReranker::new(&r.url, &r.model, key)) as _),
            contextualizer: self.contextualizer.as_ref().map(|c| {
                Box::new(HttpContextualizer::new(
                    &c.url,
                    &c.model,
                    key,
                    self.context_workers,
                )) as _
            }),
        }
    }

    pub fn open_index(&self) -> Result<Index> {
        Index::open(&self.db, self.source.clone(), self.models())
            .with_context(|| format!("opening index {}", self.db.display()))
    }

    /// Bring the index up to date, then hand back an app with separate reader and writer
    /// handles, so later rebuilds never block searches.
    pub fn open_app(&self) -> Result<(Arc<App>, BuildStats)> {
        let mut writer = self.open_index()?;
        let stats = writer.build()?;
        Ok((App::new(self.open_index()?, writer), stats))
    }
}

fn endpoint(url: Option<String>, model: Option<String>, name: &str) -> Result<Option<Endpoint>> {
    // `NAME=` in an env file arrives as Some(""); treat it as unset, like a missing line.
    let set = |v: Option<String>| v.filter(|s| !s.trim().is_empty());
    match (set(url), set(model)) {
        (Some(url), Some(model)) => Ok(Some(Endpoint { url, model })),
        (None, None) => Ok(None),
        _ => bail!("set both ARIADNE_{name}_URL and ARIADNE_{name}_MODEL, or neither"),
    }
}

/// Refuse to serve on the network without a usable token.
pub fn check_serve_auth(bind: &SocketAddr, token: Option<&str>) -> Result<()> {
    match token {
        Some(t) if !(MIN_TOKEN_CHARS..=MAX_TOKEN_CHARS).contains(&t.chars().count()) => {
            bail!("ARIADNE_TOKEN must be {MIN_TOKEN_CHARS} to {MAX_TOKEN_CHARS} characters")
        }
        None if !bind.ip().is_loopback() => {
            bail!("refusing to listen on {bind} without ARIADNE_TOKEN; set a token or bind to 127.0.0.1")
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::SearchRequest;
    use std::fs;

    fn settings(root: Option<PathBuf>, db: PathBuf) -> Settings {
        Settings {
            root,
            db,
            ..Default::default()
        }
    }

    #[test]
    fn root_is_required_and_models_come_in_pairs() {
        let db = PathBuf::from("unused.db");
        assert!(Config::new(settings(None, db.clone())).is_err());
        let half = Settings {
            embed_url: Some("http://x".into()),
            ..settings(Some("/tmp".into()), db.clone())
        };
        let err = Config::new(half).unwrap_err().to_string();
        assert!(err.contains("ARIADNE_EMBED_URL"), "{err}");
        let half_context = Settings {
            context_model: Some("m".into()),
            ..settings(Some("/tmp".into()), db.clone())
        };
        assert!(Config::new(half_context).is_err());
        let too_many = Settings {
            context_workers: Some(64),
            ..settings(Some("/tmp".into()), db.clone())
        };
        assert!(Config::new(too_many).is_err());
        let blank = Settings {
            context_url: Some(String::new()),
            context_model: Some(" ".into()),
            ..settings(Some("/tmp".into()), db.clone())
        };
        assert!(
            Config::new(blank).unwrap().contextualizer.is_none(),
            "empty env values mean unset"
        );
        let tiny = Settings {
            max_chars: Some(10),
            ..settings(Some("/tmp".into()), db)
        };
        assert!(Config::new(tiny).is_err());
    }

    #[tokio::test]
    async fn open_app_builds_and_serves_from_one_config() {
        let src = tempfile::tempdir().unwrap();
        fs::write(
            src.path().join("a.md"),
            "# A\n\n## Ports\n\nMoonraker on 7125.\n",
        )
        .unwrap();
        let db = tempfile::tempdir().unwrap();
        let config = Config::new(settings(
            Some(src.path().to_path_buf()),
            db.path().join("i.db"),
        ))
        .unwrap();
        let (app, stats) = config.open_app().unwrap();
        assert_eq!(stats.files_changed, 1);
        let hits = app
            .search(SearchRequest {
                query: "moonraker".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(hits[0].reference, "a#Ports");
        assert_eq!(config.open_app().unwrap().1.files_changed, 0);
    }

    #[test]
    fn serve_auth_rules() {
        let local: SocketAddr = "127.0.0.1:8090".parse().unwrap();
        let open: SocketAddr = "0.0.0.0:8090".parse().unwrap();
        assert!(check_serve_auth(&local, None).is_ok());
        assert!(check_serve_auth(&open, None).is_err());
        assert!(check_serve_auth(&open, Some("short")).is_err());
        assert!(check_serve_auth(&open, Some("0123456789abcdef")).is_ok());
        assert!(check_serve_auth(&open, Some(&"x".repeat(MAX_TOKEN_CHARS + 1))).is_err());
    }
}
