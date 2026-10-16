//! HTTP adapters for the model ports: OpenAI-compatible `/v1/embeddings` and llama.cpp-style
//! `/v1/rerank`, such as llama-swap serves.

use anyhow::{anyhow, Context, Result};
use ariadne_core::{Contextualizer, Embedder, Reranker};
use serde_json::{json, Value};
use std::time::Duration;

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new().timeout(timeout).build()
}

fn post(agent: &ureq::Agent, url: &str, api_key: &str, body: Value) -> Result<Value> {
    let mut req = agent.post(url);
    if !api_key.is_empty() {
        req = req.set("Authorization", &format!("Bearer {api_key}"));
    }
    let resp = req.send_json(body).map_err(|e| match e {
        ureq::Error::Status(code, _) => anyhow!("{url} returned HTTP {code}"),
        other => anyhow!("{url} failed: {other}"),
    })?;
    resp.into_json::<Value>()
        .with_context(|| format!("{url} returned invalid JSON"))
}

pub struct HttpEmbedder {
    url: String,
    model: String,
    api_key: String,
    batch_size: usize,
    max_chars: usize,
    agent: ureq::Agent,
}

impl HttpEmbedder {
    pub fn new(base_url: &str, model: &str, api_key: &str) -> Self {
        Self {
            url: format!("{}/v1/embeddings", base_url.trim_end_matches('/')),
            model: model.to_string(),
            api_key: api_key.to_string(),
            batch_size: 32,
            max_chars: 2000,
            agent: agent(Duration::from_secs(120)),
        }
    }
}

impl Embedder for HttpEmbedder {
    fn name(&self) -> String {
        format!("openai:{}", self.model)
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size) {
            let input: Vec<String> = batch
                .iter()
                .map(|t| t.chars().take(self.max_chars).collect())
                .collect();
            let body = post(
                &self.agent,
                &self.url,
                &self.api_key,
                json!({"model": self.model, "input": input}),
            )?;
            let mut data = body["data"]
                .as_array()
                .ok_or_else(|| anyhow!("embedding response has no data array"))?
                .clone();
            if data.len() != batch.len() {
                return Err(anyhow!(
                    "embedding response has {} vectors for {} inputs",
                    data.len(),
                    batch.len()
                ));
            }
            data.sort_by_key(|d| d["index"].as_u64().unwrap_or(0));
            for d in data {
                let v = d["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow!("embedding entry has no vector"))?
                    .iter()
                    .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                out.push(v);
            }
        }
        Ok(out)
    }
}

pub struct HttpReranker {
    url: String,
    model: String,
    api_key: String,
    agent: ureq::Agent,
}

impl HttpReranker {
    pub fn new(base_url: &str, model: &str, api_key: &str) -> Self {
        Self {
            url: format!("{}/v1/rerank", base_url.trim_end_matches('/')),
            model: model.to_string(),
            api_key: api_key.to_string(),
            agent: agent(Duration::from_secs(60)),
        }
    }
}

impl Reranker for HttpReranker {
    fn name(&self) -> String {
        format!("rerank:{}", self.model)
    }

    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        let body = post(
            &self.agent,
            &self.url,
            &self.api_key,
            json!({"model": self.model, "query": query, "documents": documents}),
        )?;
        let mut scores: Vec<Option<f32>> = vec![None; documents.len()];
        let results = body["results"]
            .as_array()
            .ok_or_else(|| anyhow!("rerank response has no results"))?;
        for r in results {
            let i = r["index"]
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
                .filter(|i| *i < documents.len())
                .ok_or_else(|| anyhow!("rerank result has a missing or out-of-range index"))?;
            let score = r["relevance_score"]
                .as_f64()
                .ok_or_else(|| anyhow!("rerank result {i} has no relevance_score"))?;
            scores[i] = Some(score as f32);
        }
        let scores: Option<Vec<f32>> = scores.into_iter().collect();
        let scores = scores.ok_or_else(|| {
            anyhow!(
                "rerank response scored only some of the {} documents",
                documents.len()
            )
        })?;
        Ok(scores)
    }
}

/// Writes chunk contexts with any OpenAI-compatible `/v1/chat/completions` endpoint, using the
/// prompt from Anthropic's contextual retrieval write-up.
pub struct HttpContextualizer {
    url: String,
    model: String,
    api_key: String,
    workers: usize,
    max_document_chars: usize,
    agent: ureq::Agent,
}

impl HttpContextualizer {
    pub fn new(base_url: &str, model: &str, api_key: &str, workers: usize) -> Self {
        Self {
            url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
            model: model.to_string(),
            api_key: api_key.to_string(),
            workers,
            max_document_chars: 12_000,
            agent: agent(Duration::from_secs(180)),
        }
    }

    fn prompt(&self, document: &str, chunk: &str) -> String {
        let document: String = document.chars().take(self.max_document_chars).collect();
        format!(
            "<document>\n{document}\n</document>\n\
             Here is the chunk we want to situate within the whole document\n\
             <chunk>\n{chunk}\n</chunk>\n\
             Please give a short succinct context to situate this chunk within the overall \
             document for the purposes of improving search retrieval of the chunk. Answer only \
             with the succinct context and nothing else."
        )
    }
}

impl Contextualizer for HttpContextualizer {
    fn name(&self) -> String {
        format!("chat:{}", self.model)
    }

    fn parallelism(&self) -> usize {
        self.workers
    }

    fn describe(&self, document: &str, chunk: &str) -> Result<String> {
        let body = post(
            &self.agent,
            &self.url,
            &self.api_key,
            json!({
                "model": self.model,
                "messages": [{"role": "user", "content": self.prompt(document, chunk)}],
                "max_tokens": 200,
                "temperature": 0
            }),
        )?;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow!("chat response has no message content"))?;
        Ok(strip_think(content))
    }
}

/// Reasoning models may wrap their thinking in <think> tags; keep only the answer.
fn strip_think(text: &str) -> String {
    let mut out = text.to_string();
    while let (Some(start), Some(end)) = (out.find("<think>"), out.find("</think>")) {
        if end < start {
            break;
        }
        out.replace_range(start..end + "</think>".len(), "");
    }
    if let Some(start) = out.find("<think>") {
        out.truncate(start);
    }
    out.trim().to_string()
}

#[cfg(test)]
pub(crate) mod testing {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// (path, authorization header, body) for each request received.
    pub type Seen = Arc<Mutex<Vec<(String, String, String)>>>;

    /// One-request-at-a-time HTTP server returning `respond(path, body)` as JSON.
    pub fn serve<F>(respond: F) -> (String, Seen)
    where
        F: Fn(&str, &serde_json::Value) -> serde_json::Value + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let path = request_line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("")
                    .to_string();
                let (mut len, mut auth) = (0usize, String::new());
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let l = line.trim_end();
                    if l.is_empty() {
                        break;
                    }
                    let lower = l.to_ascii_lowercase();
                    if let Some(v) = lower.strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap();
                    }
                    if lower.starts_with("authorization:") {
                        auth = l["authorization:".len()..].trim().to_string();
                    }
                }
                let mut body = vec![0; len];
                reader.read_exact(&mut body).unwrap();
                let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                seen2
                    .lock()
                    .unwrap()
                    .push((path.clone(), auth, body.to_string()));
                let out = respond(&path, &body).to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    out.len(),
                    out
                );
                stream.write_all(resp.as_bytes()).unwrap();
            }
        });
        (url, seen)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::serve;
    use super::*;

    #[test]
    fn embedder_batches_orders_and_authenticates() {
        let (url, seen) = serve(|_, body| {
            let n = body["input"].as_array().unwrap().len();
            let data: Vec<_> = (0..n)
                .rev()
                .map(|i| json!({"index": i, "embedding": [i as f64 + 1.0, 0.0]}))
                .collect();
            json!({"data": data})
        });
        let mut e = HttpEmbedder::new(&format!("{url}/"), "m", "test-key");
        e.batch_size = 2;
        let vecs = e.embed(&["a".into(), "b".into(), "c".into()]).unwrap();
        // Each batch restarts at index 0; replies arrive reversed and are put back in order.
        assert_eq!(vecs, vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![1.0, 0.0]]);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert!(seen
            .iter()
            .all(|(p, a, _)| p == "/v1/embeddings" && a == "Bearer test-key"));
    }

    #[test]
    fn embedder_rejects_count_mismatch() {
        let (url, _) = serve(|_, _| json!({"data": [{"index": 0, "embedding": [1.0]}]}));
        assert!(HttpEmbedder::new(&url, "m", "")
            .embed(&["a".into(), "b".into()])
            .is_err());
    }

    #[test]
    fn contextualizer_sends_document_and_chunk_and_strips_thinking() {
        let (url, seen) = serve(
            |_, _| json!({"choices": [{"message": {"content": "<think>hmm</think> About the printer."}}]}),
        );
        let c = HttpContextualizer::new(&url, "small", "test-key", 4);
        let context = c.describe("# Qidi\n\nwhole doc", "the chunk").unwrap();
        assert_eq!(context, "About the printer.");
        assert_eq!((c.name(), c.parallelism()), ("chat:small".to_string(), 4));
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].0, "/v1/chat/completions");
        assert_eq!(seen[0].1, "Bearer test-key");
        let body: serde_json::Value = serde_json::from_str(&seen[0].2).unwrap();
        let prompt = body["messages"][0]["content"].as_str().unwrap();
        assert!(prompt.contains("whole doc") && prompt.contains("<chunk>\nthe chunk\n</chunk>"));
    }

    #[test]
    fn contextualizer_rejects_a_reply_without_content() {
        let (url, _) = serve(|_, _| json!({"choices": []}));
        assert!(HttpContextualizer::new(&url, "m", "", 1)
            .describe("d", "c")
            .is_err());
    }

    #[test]
    fn strip_think_handles_closed_and_unclosed_blocks() {
        assert_eq!(strip_think("<think>a</think>Answer"), "Answer");
        assert_eq!(strip_think("Answer<think>never closed"), "Answer");
        assert_eq!(strip_think("plain"), "plain");
    }

    #[test]
    fn reranker_rejects_partial_or_malformed_replies() {
        for reply in [
            json!({"results": [{"index": 0, "relevance_score": 0.5}]}),
            json!({"results": [{"index": 0, "relevance_score": 0.5}, {"index": 5, "relevance_score": 0.1}]}),
            json!({"results": [{"index": 0}, {"index": 1, "relevance_score": 0.1}]}),
        ] {
            let (url, _) = serve(move |_, _| reply.clone());
            assert!(HttpReranker::new(&url, "r", "")
                .rerank("q", &["x".into(), "y".into()])
                .is_err());
        }
    }

    #[test]
    fn reranker_maps_scores_by_index_without_auth_when_unset() {
        let (url, seen) = serve(
            |_, _| json!({"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}),
        );
        let scores = HttpReranker::new(&url, "r", "")
            .rerank("q", &["x".into(), "y".into()])
            .unwrap();
        assert_eq!(scores, vec![0.1, 0.9]);
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].0, "/v1/rerank");
        assert_eq!(seen[0].1, "");
    }
}
