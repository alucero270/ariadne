//! MCP front end: JSON-RPC 2.0 over HTTP POST, exposing `search` and `read_document`.

use crate::app::{
    parse_file_status, App, AppError, SearchRequest, DEFAULT_FILE_LIMIT, MAX_FILE_LIMIT, MAX_K,
    MAX_PATH_CHARS, MAX_QUERY_CHARS,
};
use ariadne_core::FileState;
use axum::{http::StatusCode, response::IntoResponse, response::Response, Json};
use serde_json::{json, Value};
use std::sync::Arc;

pub const PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];
const SEARCH_TEXT_CHARS: usize = 900;
const DOCUMENT_CHARS: usize = 50_000;

pub async fn handle(app: &Arc<App>, body: Option<Json<Value>>) -> Response {
    let Some(Json(req)) = body else {
        return error(
            &Value::Null,
            -32700,
            "parse error: body must be a JSON-RPC object",
        );
    };
    if !req.is_object() {
        return error(
            &Value::Null,
            -32600,
            "batched or non-object requests are not supported",
        );
    }
    let Some(id) = req.get("id").cloned() else {
        // Notifications get no response body.
        return StatusCode::ACCEPTED.into_response();
    };
    let params = &req["params"];
    match req["method"].as_str().unwrap_or_default() {
        "initialize" => result(&id, initialize(params)),
        "ping" => result(&id, json!({})),
        "tools/list" => result(&id, json!({ "tools": tools() })),
        "tools/call" => {
            let name = params["name"].as_str().unwrap_or_default();
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            let outcome = match name {
                "search" => search(app, &args).await,
                "read_document" => read_document(app, &args).await,
                "list_documents" => list_documents(app, &args).await,
                _ => return error(&id, -32602, &format!("unknown tool: {name}")),
            };
            let (text, is_error) = match outcome {
                Ok(text) => (text, false),
                Err(e) => (caller_message(e), true),
            };
            result(
                &id,
                json!({ "content": [{ "type": "text", "text": text }], "isError": is_error }),
            )
        }
        method => error(&id, -32601, &format!("method not found: {method}")),
    }
}

fn initialize(params: &Value) -> Value {
    let requested = params["protocolVersion"].as_str().unwrap_or_default();
    let version = if PROTOCOL_VERSIONS.contains(&requested) {
        requested
    } else {
        PROTOCOL_VERSIONS[0]
    };
    json!({
        "protocolVersion": version,
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": { "name": "ariadne", "version": env!("CARGO_PKG_VERSION") },
        "instructions": "Search first, then read_document on a cited path when a section is not enough."
    })
}

fn tools() -> Value {
    json!([
        {
            "name": "search",
            "description": "Search the indexed knowledge base (homelab docs, runbooks, decisions). Returns the best-matching sections, each cited as [[path#Heading]]. Use read_document to open a cited file in full.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "minLength": 1, "maxLength": MAX_QUERY_CHARS },
                    "k": { "type": "integer", "minimum": 1, "maximum": MAX_K }
                },
                "required": ["query"],
                "additionalProperties": false
            }
        },
        {
            "name": "read_document",
            "description": "Read one indexed document in full. Pass the path exactly as search cites it. Markdown files return their source; other formats (EPUB) return their text as Markdown headings. Omit the extension only for a markdown file.",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string", "minLength": 1, "maxLength": MAX_PATH_CHARS } },
                "required": ["path"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_documents",
            "description": "List the files in the index with each one's status, to answer 'what is indexed, and what failed to index and why?'. Filter by status (indexed or skipped) and/or a path prefix; paginate with limit (default 200, capped at 1000) and offset. Skipped files carry the reason they were left out.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "status": { "type": "string", "enum": ["indexed", "skipped"] },
                    "prefix": { "type": "string", "maxLength": MAX_PATH_CHARS },
                    "limit": { "type": "integer", "minimum": 1, "default": DEFAULT_FILE_LIMIT, "maximum": MAX_FILE_LIMIT },
                    "offset": { "type": "integer", "minimum": 0, "default": 0 }
                },
                "additionalProperties": false
            }
        }
    ])
}

async fn search(app: &Arc<App>, args: &Value) -> Result<String, AppError> {
    let query = args["query"]
        .as_str()
        .ok_or_else(|| AppError::BadRequest("query must be a string".into()))?;
    let k = match &args["k"] {
        Value::Null => None,
        v => Some(
            v.as_u64()
                .ok_or_else(|| AppError::BadRequest("k must be a positive integer".into()))?,
        ),
    };
    let hits = app
        .search(SearchRequest {
            query: query.to_string(),
            k,
            ..Default::default()
        })
        .await?;
    if hits.is_empty() {
        return Ok("No matching documents.".into());
    }
    Ok(hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let text = truncate_chars(&h.text, SEARCH_TEXT_CHARS);
            match h.context.is_empty() {
                true => format!("[{}] [[{}]]\n{text}", i + 1, h.reference),
                false => format!(
                    "[{}] [[{}]]\nContext: {}\n{text}",
                    i + 1,
                    h.reference,
                    h.context
                ),
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n"))
}

async fn read_document(app: &Arc<App>, args: &Value) -> Result<String, AppError> {
    let path = args["path"]
        .as_str()
        .ok_or_else(|| AppError::BadRequest("path must be a string".into()))?;
    let text = app.read_document(path.to_string()).await?;
    Ok(truncate_chars(&text, DOCUMENT_CHARS))
}

async fn list_documents(app: &Arc<App>, args: &Value) -> Result<String, AppError> {
    let status = match args["status"].as_str() {
        None => None,
        Some(v) => Some(parse_file_status(v)?),
    };
    let prefix = args["prefix"].as_str().map(str::to_string);
    let limit = match &args["limit"] {
        Value::Null => None,
        v => Some(
            v.as_u64()
                .ok_or_else(|| AppError::BadRequest("limit must be a positive integer".into()))?,
        ),
    };
    let offset =
        match &args["offset"] {
            Value::Null => None,
            v => Some(v.as_u64().ok_or_else(|| {
                AppError::BadRequest("offset must be a non-negative integer".into())
            })?),
        };
    let list = app.files(status, prefix, limit, offset).await?;
    let mut out = format!(
        "{} file(s) match; showing {}:\n",
        list.total,
        list.files.len()
    );
    for f in &list.files {
        let state = match f.status {
            FileState::Indexed => "indexed",
            FileState::Skipped => "skipped",
        };
        match f.reason.as_deref() {
            Some(reason) => out.push_str(&format!("{}  {state}  reason={reason}\n", f.path)),
            None => out.push_str(&format!(
                "{}  {state}  chunks={} chars={}\n",
                f.path, f.chunks, f.chars
            )),
        }
    }
    Ok(out.trim_end().to_string())
}

/// What a tool caller may see: our own messages for their mistakes, nothing internal.
fn caller_message(e: AppError) -> String {
    match e {
        AppError::BadRequest(m) | AppError::NotFound(m) => m,
        internal => {
            eprintln!("mcp tool failed: {internal}");
            "internal error".into()
        }
    }
}

fn truncate_chars(s: &str, n: usize) -> String {
    match s.char_indices().nth(n) {
        Some((i, _)) => format!("{}\n[truncated]", &s[..i]),
        None => s.to_string(),
    }
}

fn result(id: &Value, result: Value) -> Response {
    Json(json!({ "jsonrpc": "2.0", "id": id, "result": result })).into_response()
}

fn error(id: &Value, code: i64, message: &str) -> Response {
    Json(json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }))
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests::fixture;
    use axum::body::to_bytes;

    async fn rpc(app: &Arc<App>, body: Value) -> (StatusCode, Value) {
        let resp = handle(app, Some(Json(body))).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    fn call(id: u64, name: &str, args: Value) -> Value {
        json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":name,"arguments":args}})
    }

    #[tokio::test]
    async fn handshake_negotiates_version_and_accepts_notifications() {
        let f = fixture();
        let (_, init) = rpc(&f.app, json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}})).await;
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(init["result"]["serverInfo"]["name"], "ariadne");
        let (_, old) = rpc(&f.app, json!({"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"1999-01-01"}})).await;
        assert_eq!(old["result"]["protocolVersion"], PROTOCOL_VERSIONS[0]);
        let (status, _) = rpc(
            &f.app,
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let (_, list) = rpc(
            &f.app,
            json!({"jsonrpc":"2.0","id":3,"method":"tools/list"}),
        )
        .await;
        let names: Vec<_> = list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["search", "read_document", "list_documents"]);
    }

    #[tokio::test]
    async fn tools_answer_and_report_caller_mistakes_as_tool_errors() {
        let f = fixture();
        let (_, found) = rpc(&f.app, call(1, "search", json!({"query":"moonraker port"}))).await;
        assert_eq!(found["result"]["isError"], false);
        assert!(found["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("[1] [[systems/fabrication/qidi#Network Placement]]"));

        let (_, read) = rpc(
            &f.app,
            call(
                2,
                "read_document",
                json!({"path":"systems/fabrication/qidi.md"}),
            ),
        )
        .await;
        assert!(read["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("7125"));

        for (name, args) in [
            ("read_document", json!({"path":"secret.txt"})),
            ("search", json!({"query":"x","k":99})),
            ("search", json!({"query":7})),
        ] {
            let (_, r) = rpc(&f.app, call(3, name, args)).await;
            assert_eq!(r["result"]["isError"], true, "{name}");
        }
    }

    #[tokio::test]
    async fn read_document_renders_an_epub_for_the_caller_too() {
        use crate::app::tests::fixture_with_epub;
        let f = fixture_with_epub();
        let (_, read) = rpc(
            &f.app,
            call(1, "read_document", json!({"path":"Library/Some Book.epub"})),
        )
        .await;
        assert_eq!(read["result"]["isError"], false, "{read}");
        let text = read["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("# Replication Basics"), "{text}");
        assert!(text.contains("## Chapter 5 Replication"), "{text}");
        assert!(text.contains("Print a calibration cube first."), "{text}");
        assert!(!text.contains("PK\u{3}\u{4}"), "zip bytes: {text}");

        // Markdown is unchanged, and a path that is not indexed stays a tool error.
        let (_, md) = rpc(
            &f.app,
            call(
                2,
                "read_document",
                json!({ "path": "systems/fabrication/qidi" }),
            ),
        )
        .await;
        assert!(md["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("# Qidi Plus 4"));
        for path in ["Library/Some Book", "secret.txt"] {
            let (_, r) = rpc(&f.app, call(3, "read_document", json!({"path":path}))).await;
            assert_eq!(r["result"]["isError"], true, "{path}");
        }

        // The tool no longer tells the caller that every file is markdown.
        let (_, list) = rpc(
            &f.app,
            json!({"jsonrpc":"2.0","id":4,"method":"tools/list"}),
        )
        .await;
        let description = list["result"]["tools"][1]["description"].as_str().unwrap();
        assert!(description.contains("EPUB"), "{description}");
        assert!(
            !description.contains(".md suffix is optional"),
            "{description}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn list_documents_lists_filters_and_reports_mistakes_as_errors() {
        use crate::app::tests::fixture_with_skipped;
        let f = fixture_with_skipped();

        // The tool is listed alongside search and read_document.
        let (_, list) = rpc(
            &f.app,
            json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
        )
        .await;
        let names: Vec<_> = list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["search", "read_document", "list_documents"]);

        // A call returns the same rows the HTTP /files route does.
        let (_, rows) = rpc(&f.app, call(2, "list_documents", json!({}))).await;
        assert_eq!(rows["result"]["isError"], false, "{rows}");
        let text = rows["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.starts_with("3 file(s) match"), "{text}");
        for p in [
            "systems/fabrication/qidi.md",
            "systems/locked.md",
            "tools/bench.md",
        ] {
            assert!(text.contains(p), "missing {p} in:\n{text}");
        }
        assert!(text.contains("systems/locked.md  skipped"), "{text}");

        // The status filter narrows it, like the HTTP route.
        let (_, skipped) = rpc(
            &f.app,
            call(3, "list_documents", json!({"status":"skipped"})),
        )
        .await;
        let stext = skipped["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(stext.contains("systems/locked.md"), "{stext}");
        assert!(!stext.contains("tools/bench.md"), "{stext}");

        // A bad argument is a tool error, not a crash.
        let (_, bad) = rpc(&f.app, call(4, "list_documents", json!({"status":"bogus"}))).await;
        assert_eq!(bad["result"]["isError"], true, "{bad}");
    }

    #[tokio::test]
    async fn protocol_errors_use_json_rpc_codes() {
        let f = fixture();
        assert_eq!(
            rpc(&f.app, call(1, "rm", json!({}))).await.1["error"]["code"],
            -32602
        );
        assert_eq!(
            rpc(
                &f.app,
                json!({"jsonrpc":"2.0","id":2,"method":"resources/list"})
            )
            .await
            .1["error"]["code"],
            -32601
        );
        assert_eq!(
            rpc(&f.app, json!([{"jsonrpc":"2.0","id":3,"method":"ping"}]))
                .await
                .1["error"]["code"],
            -32600
        );
        let resp = handle(&f.app, None).await;
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap()["error"]["code"],
            -32700
        );
    }
}
