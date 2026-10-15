//! HTTP front end: the REST routes, the MCP route, and bearer-token auth.

use crate::app::{App, AppError, SearchRequest};
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
    if let Some(expected) = &state.expected_auth {
        let supplied = headers
            .get("authorization")
            .map(|v| v.as_bytes())
            .unwrap_or_default();
        if !constant_time_eq(supplied, expected.as_bytes()) {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "invalid or missing bearer token" })),
            )
                .into_response();
        }
    }
    next.run(req).await
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

async fn health(State(s): State<HttpState>) -> Result<Json<Value>, AppError> {
    let stats = s.app.stats().await?;
    Ok(Json(
        json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION"), "index": stats }),
    ))
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    k: Option<u64>,
    mode: Option<String>,
    rerank: Option<bool>,
}

async fn search(
    State(s): State<HttpState>,
    Query(p): Query<SearchParams>,
) -> Result<Json<Value>, AppError> {
    let query = p.q.trim().to_string();
    let hits = s
        .app
        .search(SearchRequest {
            query: p.q,
            k: p.k,
            mode: p.mode,
            rerank: p.rerank,
        })
        .await?;
    Ok(Json(json!({ "query": query, "results": hits })))
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
        assert_eq!(
            (status, body["index"]["files"].clone()),
            (StatusCode::OK, json!(1))
        );
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

    #[test]
    fn constant_time_eq_basics() {
        assert!(constant_time_eq(b"Bearer abc", b"Bearer abc"));
        assert!(!constant_time_eq(b"Bearer abc", b"Bearer abd"));
        assert!(!constant_time_eq(b"Bearer ab", b"Bearer abc"));
        assert!(!constant_time_eq(b"", b"Bearer abc"));
    }
}
