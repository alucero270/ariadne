//! HTTP front end: the REST routes, the MCP route, and bearer-token auth.

use crate::app::{parse_file_status, App, AppError, FileList, SearchRequest};
use crate::mcp;
use axum::{
    extract::{DefaultBodyLimit, Query, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

const BODY_LIMIT: usize = 1024 * 1024;
const MAX_AUTH_HEADER: usize = 4096;

#[derive(Clone)]
struct HttpState {
    app: Arc<App>,
    expected_auth: Option<String>,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            AppError::NotFound(m) => (StatusCode::NOT_FOUND, m),
            internal => {
                eprintln!("request failed: {internal}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal error".to_string(),
                )
            }
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}

/// Everything but `/health` requires `Authorization: Bearer <token>` when a token is set.
pub fn router(app: Arc<App>, token: Option<String>) -> Router {
    let state = HttpState {
        app,
        expected_auth: token.map(|t| format!("Bearer {t}")),
    };
    let protected = Router::new()
        .route("/search", get(search))
        .route("/documents", get(document))
        .route("/files", get(files))
        .route("/reindex", post(reindex))
        .route(
            "/mcp",
            post(mcp_route).get(|| async { StatusCode::METHOD_NOT_ALLOWED }),
        )
        .layer(middleware::from_fn_with_state(state.clone(), require_token));
    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(state)
}

async fn require_token(
    State(state): State<HttpState>,
    headers: HeaderMap,
    req: Request,
    next: Next,
) -> Response {
    if !authorized(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "invalid or missing bearer token" })),
        )
            .into_response();
    }
    next.run(req).await
}

fn authorized(state: &HttpState, headers: &HeaderMap) -> bool {
    let Some(expected) = &state.expected_auth else {
        return true;
    };
    let supplied = headers
        .get("authorization")
        .map(|v| v.as_bytes())
        .unwrap_or_default();
    constant_time_eq(supplied, expected.as_bytes())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() > MAX_AUTH_HEADER || b.len() > MAX_AUTH_HEADER {
        return false;
    }
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0));
    }
    diff == 0
}

/// Open to everyone so load balancers can probe it. Index details only for authorized callers.
async fn health(State(s): State<HttpState>, headers: HeaderMap) -> Result<Json<Value>, AppError> {
    let mut body = json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") });
    if authorized(&s, &headers) {
        body["index"] = json!(s.app.stats().await?);
    }
    Ok(Json(body))
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    k: Option<u64>,
    mode: Option<String>,
    rerank: Option<bool>,
    expand: Option<String>,
}

async fn search(
    State(s): State<HttpState>,
    Query(p): Query<SearchParams>,
) -> Result<Json<Value>, AppError> {
    let hits = s
        .app
        .search(SearchRequest {
            query: p.q,
            k: p.k,
            mode: p.mode,
            rerank: p.rerank,
            expand: p.expand,
        })
        .await?;
    Ok(Json(json!({ "results": hits })))
}

#[derive(Deserialize)]
struct DocumentParams {
    path: String,
}

async fn document(
    State(s): State<HttpState>,
    Query(p): Query<DocumentParams>,
) -> Result<Json<Value>, AppError> {
    let text = s.app.read_document(p.path.clone()).await?;
    Ok(Json(json!({ "path": p.path, "text": text })))
}

async fn reindex(State(s): State<HttpState>) -> Result<Json<Value>, AppError> {
    let stats = s.app.reindex().await?;
    Ok(Json(json!({ "status": "ok", "stats": stats })))
}

#[derive(Deserialize)]
struct FilesParams {
    status: Option<String>,
    prefix: Option<String>,
    // u64 so axum's query deserializer rejects non-integer and negative values with a 400.
    limit: Option<u64>,
    offset: Option<u64>,
}

/// What is indexed and what was skipped: `{ total, files }` filtered by status and path
/// prefix, sorted by path, paginated by limit (default 200, capped at 1000) and offset.
async fn files(
    State(s): State<HttpState>,
    Query(p): Query<FilesParams>,
) -> Result<Json<FileList>, AppError> {
    let status = match p.status.as_deref() {
        None => None,
        Some(v) => Some(parse_file_status(v)?),
    };
    Ok(Json(
        s.app.files(status, p.prefix, p.limit, p.offset).await?,
    ))
}

async fn mcp_route(State(s): State<HttpState>, body: Option<Json<Value>>) -> Response {
    mcp::handle(&s.app, body).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests::fixture;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    const TOKEN: &str = "t0ken-t0ken-t0ken";

    async fn send(
        router: &Router,
        method: &str,
        uri: &str,
        token: Option<&str>,
    ) -> (StatusCode, Value) {
        let mut req = HttpRequest::builder().method(method).uri(uri);
        if let Some(t) = token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        let resp = router
            .clone()
            .oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    #[tokio::test]
    async fn health_is_open_everything_else_needs_the_token() {
        let f = fixture();
        let router = router(f.app.clone(), Some(TOKEN.into()));
        let (status, body) = send(&router, "GET", "/health", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert!(body.get("index").is_none(), "index details leaked: {body}");
        let (_, body) = send(&router, "GET", "/health", Some(TOKEN)).await;
        assert_eq!(body["index"]["files"], 1);
        for (method, uri) in [
            ("GET", "/search?q=moonraker"),
            ("GET", "/documents?path=x"),
            ("POST", "/reindex"),
            ("POST", "/mcp"),
        ] {
            assert_eq!(
                send(&router, method, uri, None).await.0,
                StatusCode::UNAUTHORIZED,
                "{uri}"
            );
            assert_eq!(
                send(&router, method, uri, Some("wrong")).await.0,
                StatusCode::UNAUTHORIZED,
                "{uri}"
            );
        }
        let (status, body) = send(&router, "GET", "/search?q=moonraker", Some(TOKEN)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["results"][0]["ref"],
            "systems/fabrication/qidi#Network Placement"
        );
    }

    #[tokio::test]
    async fn errors_map_to_status_codes() {
        let f = fixture();
        let router = router(f.app.clone(), None);
        assert_eq!(
            send(&router, "GET", "/search?q=x&k=0", None).await.0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            send(&router, "GET", "/search?q=x&mode=bogus", None).await.0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            send(&router, "GET", "/documents?path=secret.txt", None)
                .await
                .0,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            send(
                &router,
                "GET",
                "/documents?path=systems/fabrication/qidi",
                None
            )
            .await
            .0,
            StatusCode::OK
        );
        assert_eq!(
            send(&router, "GET", "/mcp", None).await.0,
            StatusCode::METHOD_NOT_ALLOWED
        );
        assert_eq!(
            send(&router, "POST", "/reindex", None).await.1["stats"]["files_changed"],
            0
        );
    }

    #[tokio::test]
    async fn documents_route_returns_an_epub_as_text() {
        let f = crate::app::tests::fixture_with_epub();
        let router = router(f.app.clone(), None);
        let (status, body) = send(
            &router,
            "GET",
            "/documents?path=Library/Some%20Book.epub",
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["path"], "Library/Some Book.epub");
        let text = body["text"].as_str().unwrap();
        assert!(text.starts_with("# Replication Basics"), "{text}");
        assert!(text.contains("## Chapter 1 Setup"), "{text}");
        assert!(text.contains("level the bed"), "{text}");
        assert!(!text.bytes().any(|b| b == 0), "zip bytes: {text:?}");

        // A markdown path is still served as its source, and an epub path without its
        // extension is 404 like any other unreadable path.
        let (status, body) = send(
            &router,
            "GET",
            "/documents?path=systems/fabrication/qidi",
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["text"].as_str().unwrap().starts_with("# Qidi Plus 4"));
        for path in ["Library/Some%20Book", "../etc/passwd"] {
            assert_eq!(
                send(&router, "GET", &format!("/documents?path={path}"), None)
                    .await
                    .0,
                StatusCode::NOT_FOUND,
                "{path}"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn files_route_lists_filters_paginates_and_validates() {
        let f = crate::app::tests::fixture_with_skipped();
        let router = router(f.app.clone(), Some(TOKEN.into()));

        // Token-protected like /search.
        assert_eq!(
            send(&router, "GET", "/files", None).await.0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            send(&router, "GET", "/files", Some("wrong")).await.0,
            StatusCode::UNAUTHORIZED
        );

        let (status, body) = send(&router, "GET", "/files", Some(TOKEN)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total"], 3);
        let files = body["files"].as_array().unwrap();
        let paths: Vec<_> = files.iter().map(|f| f["path"].as_str().unwrap()).collect();
        assert_eq!(
            paths,
            [
                "systems/fabrication/qidi.md",
                "systems/locked.md",
                "tools/bench.md"
            ]
        );
        assert_eq!(files[0]["status"], "indexed");
        assert!(files[0]["chunks"].as_i64().unwrap() > 0);
        assert!(files[0].get("reason").is_none(), "reason only when Some");
        assert_eq!(files[1]["status"], "skipped");
        assert!(files[1]["reason"]
            .as_str()
            .unwrap()
            .contains("Permission denied"));

        // Status and prefix filters change total, not just the slice.
        let (_, body) = send(&router, "GET", "/files?status=skipped", Some(TOKEN)).await;
        assert_eq!(body["total"], 1);
        assert_eq!(body["files"][0]["path"], "systems/locked.md");
        let (_, body) = send(&router, "GET", "/files?status=indexed", Some(TOKEN)).await;
        assert_eq!(body["total"], 2);
        let (_, body) = send(&router, "GET", "/files?prefix=systems/", Some(TOKEN)).await;
        assert_eq!(body["total"], 2);
        let (_, body) = send(
            &router,
            "GET",
            "/files?status=indexed&prefix=tools/",
            Some(TOKEN),
        )
        .await;
        assert_eq!(body["total"], 1);
        assert_eq!(body["files"][0]["path"], "tools/bench.md");

        // Pagination: total is pre-pagination, files is the slice.
        let (_, body) = send(&router, "GET", "/files?limit=1&offset=1", Some(TOKEN)).await;
        assert_eq!(body["total"], 3);
        assert_eq!(body["files"].as_array().unwrap().len(), 1);
        assert_eq!(body["files"][0]["path"], "systems/locked.md");
        let (_, body) = send(&router, "GET", "/files?limit=2&offset=3", Some(TOKEN)).await;
        assert_eq!(body["total"], 3);
        assert!(body["files"].as_array().unwrap().is_empty());
        // A limit above the cap is clamped, not refused; with only three files the whole
        // list comes back. The clamp itself (1001 -> 1000) is asserted in app::tests.
        let (status, body) = send(&router, "GET", "/files?limit=1001", Some(TOKEN)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["files"].as_array().unwrap().len(), 3);

        // Caller mistakes: bad status values and bad limit/offset are 400s.
        for bad in [
            "/files?status=bogus",
            "/files?limit=0",
            "/files?limit=-1",
            "/files?limit=abc",
            "/files?offset=-1",
            "/files?offset=abc",
        ] {
            assert_eq!(
                send(&router, "GET", bad, Some(TOKEN)).await.0,
                StatusCode::BAD_REQUEST,
                "{bad}"
            );
        }
    }

    #[test]
    fn constant_time_eq_basics() {
        assert!(constant_time_eq(b"Bearer abc", b"Bearer abc"));
        assert!(!constant_time_eq(b"Bearer abc", b"Bearer abd"));
        assert!(!constant_time_eq(b"Bearer ab", b"Bearer abc"));
        assert!(!constant_time_eq(b"", b"Bearer abc"));
    }
}
