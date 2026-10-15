//! HTTP adapters for the model ports: OpenAI-compatible `/v1/embeddings` and llama.cpp-style
//! `/v1/rerank`, such as llama-swap serves.

use anyhow::{anyhow, Context, Result};
use ariadne_core::ports::normalize;
use ariadne_core::{Embedder, Reranker};
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
                out.push(normalize(v));
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
        let mut scores = vec![0.0f32; documents.len()];
        for r in body["results"]
            .as_array()
            .ok_or_else(|| anyhow!("rerank response has no results"))?
        {
            let i = r["index"]
                .as_u64()
                .ok_or_else(|| anyhow!("rerank result has no index"))? as usize;
            if let Some(slot) = scores.get_mut(i) {
                *slot = r["relevance_score"].as_f64().unwrap_or(0.0) as f32;
            }
        }
        Ok(scores)
    }
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
    fn embedder_batches_orders_normalizes_and_authenticates() {
        let (url, seen) = serve(|_, body| {
            let n = body["input"].as_array().unwrap().len();
            let data: Vec<_> = (0..n)
                .rev()
                .map(|i| json!({"index": i, "embedding": [i as f64 + 1.0, 0.0]}))
                .collect();
            json!({"data": data})
        });
        let mut e = HttpEmbedder::new(&format!("{url}/"), "m", "LOCAL");
        e.batch_size = 2;
        let vecs = e.embed(&["a".into(), "b".into(), "c".into()]).unwrap();
        assert_eq!(vecs.len(), 3);
        assert!(vecs
            .iter()
            .all(|v| ((v[0] * v[0] + v[1] * v[1]) - 1.0).abs() < 1e-6));
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert!(seen
            .iter()
            .all(|(p, a, _)| p == "/v1/embeddings" && a == "Bearer LOCAL"));
    }

    #[test]
    fn embedder_rejects_count_mismatch() {
        let (url, _) = serve(|_, _| json!({"data": [{"index": 0, "embedding": [1.0]}]}));
        assert!(HttpEmbedder::new(&url, "m", "")
            .embed(&["a".into(), "b".into()])
            .is_err());
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
