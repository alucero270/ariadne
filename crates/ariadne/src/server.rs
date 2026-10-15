//! HTTP API and MCP (JSON-RPC over HTTP) for Ariadne.

use ariadne_core::{Index, Mode, SearchOptions};
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
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub const MCP_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];
const BODY_LIMIT: usize = 1024 * 1024;
const MAX_TOKEN_HEADER: usize = 4096;
const SEARCH_TEXT_CHARS: usize = 900;
const DOCUMENT_CHARS: usize = 50_000;

#[derive(Clone)]
pub struct AppState {
    pub index: Arc<Mutex<Index>>,
    pub root: Option<PathBuf>,
    pub excludes: Vec<String>,
    pub max_chars: usize,
    pub token: Option<String>,
}

type ApiResult = Result<Json<Value>, (StatusCode, Json<Value>)>;

fn api_error(status: StatusCode, msg: impl ToString) -> (StatusCode, Json<Value>) {
    (status, Json(json!({"error": msg.to_string()})))
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() > MAX_TOKEN_HEADER || b.len() > MAX_TOKEN_HEADER {
        return false;
    }
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= usize::from(*a.get(i).unwrap_or(&0) ^ *b.get(i).unwrap_or(&0));
    }
    diff == 0
}

async fn require_token(
    State(state): State<AppState>,
    headers: HeaderMap,
    req: Request,
    next: Next,
) -> Response {
    if let Some(token) = &state.token {
        let expected = format!("Bearer {token}");
        let supplied = headers
            .get("authorization")
            .map(|v| v.as_bytes())
            .unwrap_or_default();
        if !constant_time_eq(supplied, expected.as_bytes()) {
            return api_error(StatusCode::UNAUTHORIZED, "invalid or missing bearer token")
                .into_response();
        }
    }
    next.run(req).await
}

async fn with_index<T, F>(state: &AppState, f: F) -> Result<T, (StatusCode, Json<Value>)>
where
    T: Send + 'static,
    F: FnOnce(&mut Index) -> anyhow::Result<T> + Send + 'static,
{
    let index = state.index.clone();
    tokio::task::spawn_blocking(move || {
        let mut guard = index
            .lock()
            .map_err(|_| anyhow::anyhow!("index lock poisoned"))?;
        f(&mut guard)
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e))?
    .map_err(|e| api_error(StatusCode::BAD_REQUEST, e))
}

pub fn router(state: AppState) -> Router {
    let protected = Router::new()
        .route("/search", get(search))
        .route("/documents", get(document))
        .route("/reindex", post(reindex))
        .route(
            "/mcp",
            post(mcp).get(|| async { StatusCode::METHOD_NOT_ALLOWED }),
        )
        .layer(middleware::from_fn_with_state(state.clone(), require_token));
    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> ApiResult {
    let stats = with_index(&state, |idx| idx.stats()).await?;
    Ok(Json(
        json!({"status": "ok", "version": env!("CARGO_PKG_VERSION"), "index": stats}),
    ))
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    k: Option<usize>,
    mode: Option<String>,
    rerank: Option<bool>,
}

fn search_options(
    k: Option<usize>,
    mode: Option<&str>,
    rerank: Option<bool>,
) -> anyhow::Result<SearchOptions> {
    let k = k.unwrap_or(8);
    anyhow::ensure!((1..=50).contains(&k), "k must be between 1 and 50");
    Ok(SearchOptions {
        k,
        mode: mode.unwrap_or("hybrid").parse::<Mode>()?,
        rerank: rerank.unwrap_or(true),
        ..Default::default()
    })
}

async fn search(State(state): State<AppState>, Query(p): Query<SearchParams>) -> ApiResult {
    let q = p.q.trim().to_string();
    if q.is_empty() || q.chars().count() > 500 {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "q must be 1-500 characters",
        ));
    }
    let opts = search_options(p.k, p.mode.as_deref(), p.rerank)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, e))?;
    let mode = opts.mode;
    let query = q.clone();
    let hits = with_index(&state, move |idx| idx.search(&query, &opts)).await?;
    Ok(Json(json!({"query": q, "mode": mode, "results": hits})))
}

#[derive(Deserialize)]
struct DocParams {
    path: String,
}

async fn document(State(state): State<AppState>, Query(p): Query<DocParams>) -> ApiResult {
    let path = p.path.clone();
    let text = with_index(&state, move |idx| idx.read_document(&path)).await?;
    Ok(Json(json!({"path": p.path, "text": text})))
}

async fn reindex(State(state): State<AppState>) -> ApiResult {
    let Some(root) = state.root.clone() else {
        return Err(api_error(
            StatusCode::CONFLICT,
            "server was started without a source root",
        ));
    };
    let (excludes, max_chars) = (state.excludes.clone(), state.max_chars);
    let stats = with_index(&state, move |idx| idx.build(&root, &excludes, max_chars)).await?;
    Ok(Json(json!({"status": "ok", "stats": stats})))
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "search",
            "description": "Search the indexed knowledge base (homelab docs, runbooks, decisions). Returns the best-matching sections, each cited as [[path#Heading]]. Use read_document to open a cited file in full.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 500},
                    "k": {"type": "integer", "minimum": 1, "maximum": 20}
                },
                "required": ["query"],
                "additionalProperties": false
            }
        },
        {
            "name": "read_document",
            "description": "Read one indexed document in full by its path, as cited by search (the .md suffix is optional).",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 400}},
                "required": ["path"],
                "additionalProperties": false
            }
        }
    ])
}

fn truncate_chars(s: &str, n: usize) -> String {
    match s.char_indices().nth(n) {
        Some((i, _)) => format!("{}\n[truncated]", &s[..i]),
        None => s.to_string(),
    }
}

async fn call_tool(state: &AppState, name: &str, args: &Value) -> Result<String, String> {
    match name {
        "search" => {
            let query = args["query"]
                .as_str()
                .map(str::trim)
                .filter(|q| !q.is_empty() && q.chars().count() <= 500);
            let Some(query) = query.map(str::to_string) else {
                return Err("query must be a non-empty string of at most 500 characters".into());
            };
            let k = match &args["k"] {
                Value::Null => 5,
                v => v
                    .as_u64()
                    .filter(|k| (1..=20).contains(k))
                    .ok_or("k must be an integer from 1 to 20")? as usize,
            };
            let opts = SearchOptions {
                k,
                ..Default::default()
            };
            let hits = with_index(state, move |idx| idx.search(&query, &opts))
                .await
                .map_err(|(_, e)| e.0["error"].to_string())?;
            if hits.is_empty() {
                return Ok("No matching documents.".into());
            }
            Ok(hits
                .iter()
                .enumerate()
                .map(|(i, h)| {
                    format!(
                        "[{}] [[{}]]\n{}",
                        i + 1,
                        h.reference,
                        truncate_chars(&h.text, SEARCH_TEXT_CHARS)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n"))
        }
        "read_document" => {
            let Some(path) = args["path"].as_str().map(str::to_string) else {
                return Err("path must be a string".into());
            };
            let text = with_index(state, move |idx| idx.read_document(&path))
                .await
                .map_err(|(_, e)| e.0["error"].as_str().unwrap_or("read failed").to_string())?;
            Ok(truncate_chars(&text, DOCUMENT_CHARS))
        }
        _ => unreachable!("unknown tools are rejected before call_tool"),
    }
}

fn rpc_result(id: &Value, result: Value) -> Json<Value> {
    Json(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

fn rpc_error(id: &Value, code: i64, message: &str) -> Json<Value> {
    Json(json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}))
}

async fn mcp(State(state): State<AppState>, body: Option<Json<Value>>) -> Response {
    let Some(Json(req)) = body else {
        return rpc_error(
            &Value::Null,
            -32700,
            "parse error: body must be a JSON-RPC object",
        )
        .into_response();
    };
    if !req.is_object() {
        return rpc_error(
            &Value::Null,
            -32600,
            "batched or non-object requests are not supported",
        )
        .into_response();
    }
    let method = req["method"].as_str().unwrap_or_default();
    let Some(id) = req.get("id").cloned() else {
        return StatusCode::ACCEPTED.into_response();
    };
    let params = &req["params"];
    match method {
        "initialize" => {
            let requested = params["protocolVersion"].as_str().unwrap_or_default();
            let version = if MCP_PROTOCOL_VERSIONS.contains(&requested) {
                requested
            } else {
                MCP_PROTOCOL_VERSIONS[0]
            };
            rpc_result(
                &id,
                json!({
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "ariadne", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": "Search first, then read_document on a cited path when a section is not enough."
                }),
            )
            .into_response()
        }
        "ping" => rpc_result(&id, json!({})).into_response(),
        "tools/list" => rpc_result(&id, json!({"tools": tool_definitions()})).into_response(),
        "tools/call" => {
            let name = params["name"].as_str().unwrap_or_default();
            if !matches!(name, "search" | "read_document") {
                return rpc_error(&id, -32602, &format!("unknown tool: {name}")).into_response();
            }
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            let (text, is_error) = match call_tool(&state, name, &args).await {
                Ok(text) => (text, false),
                Err(msg) => (msg, true),
            };
            rpc_result(
                &id,
                json!({"content": [{"type": "text", "text": text}], "isError": is_error}),
            )
            .into_response()
        }
        _ => rpc_error(&id, -32601, &format!("method not found: {method}")).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use http_body_util::BodyExt;
    use std::fs;
    use tower::ServiceExt;

    struct Fixture {
        _src: tempfile::TempDir,
        _db: tempfile::TempDir,
        app: Router,
    }

    fn fixture(token: Option<&str>) -> Fixture {
        let src = tempfile::tempdir().unwrap();
        fs::create_dir_all(src.path().join("systems/fabrication")).unwrap();
        fs::write(
            src.path().join("systems/fabrication/qidi.md"),
            "# Qidi Plus 4\n\n## Network Placement\n\nMoonraker listens on port 7125.\n",
        )
        .unwrap();
        fs::write(src.path().join("secret.txt"), "not indexed").unwrap();
        let db = tempfile::tempdir().unwrap();
        let excludes: Vec<String> = ariadne_core::source::DEFAULT_EXCLUDES
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut index = Index::open(&db.path().join("i.db"), None, None).unwrap();
        index.build(src.path(), &excludes, 1200).unwrap();
        let state = AppState {
            index: Arc::new(Mutex::new(index)),
            root: Some(src.path().to_path_buf()),
            excludes,
            max_chars: 1200,
            token: token.map(str::to_string),
        };
        Fixture {
            app: router(state),
            _src: src,
            _db: db,
        }
    }

    async fn call(
        app: &Router,
        method: &str,
        uri: &str,
        body: Option<Value>,
        token: Option<&str>,
    ) -> (StatusCode, Value) {
        let mut req = HttpRequest::builder().method(method).uri(uri);
        if let Some(t) = token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        let req = match body {
            Some(b) => req
                .header("content-type", "application/json")
                .body(Body::from(b.to_string())),
            None => req.body(Body::empty()),
        }
        .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    #[tokio::test]
    async fn health_is_open_and_reports_index() {
        let f = fixture(Some("t0ken-t0ken-t0ken"));
        let (status, body) = call(&f.app, "GET", "/health", None, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["index"]["files"], 1);
        assert_eq!(body["index"]["embedder"], "none");
    }

    #[tokio::test]
    async fn token_is_required_when_configured() {
        let f = fixture(Some("t0ken-t0ken-t0ken"));
        assert_eq!(
            call(&f.app, "GET", "/search?q=moonraker", None, None)
                .await
                .0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            call(&f.app, "GET", "/search?q=moonraker", None, Some("wrong"))
                .await
                .0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            call(&f.app, "POST", "/mcp", Some(json!({})), None).await.0,
            StatusCode::UNAUTHORIZED
        );
        let (status, body) = call(
            &f.app,
            "GET",
            "/search?q=moonraker",
            None,
            Some("t0ken-t0ken-t0ken"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["results"][0]["ref"],
            "systems/fabrication/qidi#Network Placement"
        );
    }

    #[tokio::test]
    async fn search_validates_input() {
        let f = fixture(None);
        assert_eq!(
            call(&f.app, "GET", "/search?q=%20", None, None).await.0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            call(&f.app, "GET", "/search?q=x&mode=bogus", None, None)
                .await
                .0,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            call(&f.app, "GET", "/search?q=x&k=0", None, None).await.0,
            StatusCode::BAD_REQUEST
        );
    }

    #[tokio::test]
    async fn documents_are_confined_to_the_index() {
        let f = fixture(None);
        let (status, body) = call(
            &f.app,
            "GET",
            "/documents?path=systems/fabrication/qidi",
            None,
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["text"].as_str().unwrap().contains("Moonraker"));
        for bad in ["secret.txt", "../etc/passwd", "missing"] {
            assert_eq!(
                call(&f.app, "GET", &format!("/documents?path={bad}"), None, None)
                    .await
                    .0,
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[tokio::test]
    async fn reindex_picks_up_changes() {
        let f = fixture(None);
        let (status, body) = call(&f.app, "POST", "/reindex", None, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["stats"]["files_changed"], 0);
    }

    #[tokio::test]
    async fn mcp_handshake_list_and_calls() {
        let f = fixture(None);
        let (_, init) = call(&f.app, "POST", "/mcp", Some(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}})), None).await;
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(init["result"]["serverInfo"]["name"], "ariadne");

        let (_, unknown_version) = call(&f.app, "POST", "/mcp", Some(json!({"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"1999-01-01"}})), None).await;
        assert_eq!(
            unknown_version["result"]["protocolVersion"],
            MCP_PROTOCOL_VERSIONS[0]
        );

        let (status, _) = call(
            &f.app,
            "POST",
            "/mcp",
            Some(json!({"jsonrpc":"2.0","method":"notifications/initialized"})),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);

        let (_, list) = call(
            &f.app,
            "POST",
            "/mcp",
            Some(json!({"jsonrpc":"2.0","id":3,"method":"tools/list"})),
            None,
        )
        .await;
        let names: Vec<_> = list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["search", "read_document"]);

        let (_, found) = call(&f.app, "POST", "/mcp", Some(json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"search","arguments":{"query":"moonraker port"}}})), None).await;
        assert_eq!(found["result"]["isError"], false);
        assert!(found["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("[1] [[systems/fabrication/qidi#Network Placement]]"));

        let (_, read) = call(&f.app, "POST", "/mcp", Some(json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"read_document","arguments":{"path":"systems/fabrication/qidi.md"}}})), None).await;
        assert!(read["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("7125"));

        let (_, denied) = call(&f.app, "POST", "/mcp", Some(json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"read_document","arguments":{"path":"secret.txt"}}})), None).await;
        assert_eq!(denied["result"]["isError"], true);

        let (_, bad_args) = call(&f.app, "POST", "/mcp", Some(json!({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"search","arguments":{"query":"x","k":99}}})), None).await;
        assert_eq!(bad_args["result"]["isError"], true);

        let (_, no_tool) = call(
            &f.app,
            "POST",
            "/mcp",
            Some(json!({"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"rm"}})),
            None,
        )
        .await;
        assert_eq!(no_tool["error"]["code"], -32602);

        let (_, no_method) = call(
            &f.app,
            "POST",
            "/mcp",
            Some(json!({"jsonrpc":"2.0","id":9,"method":"resources/list"})),
            None,
        )
        .await;
        assert_eq!(no_method["error"]["code"], -32601);

        let (_, batch) = call(
            &f.app,
            "POST",
            "/mcp",
            Some(json!([{"jsonrpc":"2.0","id":10,"method":"ping"}])),
            None,
        )
        .await;
        assert_eq!(batch["error"]["code"], -32600);
    }

    #[test]
    fn constant_time_eq_basics() {
        assert!(constant_time_eq(b"Bearer abc", b"Bearer abc"));
        assert!(!constant_time_eq(b"Bearer abc", b"Bearer abd"));
        assert!(!constant_time_eq(b"Bearer ab", b"Bearer abc"));
        assert!(!constant_time_eq(b"", b"Bearer abc"));
    }
}
