//! Retrieval evaluation: Recall@k and MRR over a question set with known answer files.

use crate::{Index, Mode, SearchOptions};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
pub struct EvalSet {
    pub questions: Vec<Question>,
}

#[derive(Debug, Deserialize)]
pub struct Question {
    pub q: String,
    pub expect: Vec<String>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct ModeScore {
    pub recall_at_k: f64,
    pub mrr: f64,
}

#[derive(Debug, Serialize)]
pub struct EvalReport {
    pub k: usize,
    pub questions: usize,
    pub rerank: bool,
    pub modes: BTreeMap<String, ModeScore>,
    pub misses: BTreeMap<String, Vec<String>>,
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

impl Index {
    /// Score this index against `set` the way it serves: for each mode, the share of questions
    /// whose top `k` contains an expected document (Recall@k) and the mean reciprocal rank of
    /// the first one (MRR). Dense mode is measured only when an embedder is configured, and the
    /// reranker runs exactly when `rerank` asks for it and one is configured.
    pub fn evaluate(&mut self, set: &EvalSet, k: usize, rerank: bool) -> Result<EvalReport> {
        let mut report = EvalReport {
            k,
            questions: set.questions.len(),
            rerank: rerank && self.has_reranker(),
            modes: BTreeMap::new(),
            misses: BTreeMap::new(),
        };
        for mode in Mode::ALL {
            if mode == Mode::Dense && !self.has_embedder() {
                continue;
            }
            let opts = SearchOptions::new(k, mode, rerank);
            let (mut found, mut rr, mut misses) = (0usize, 0.0f64, Vec::new());
            for q in &set.questions {
                let hits = self.search(&q.q, &opts)?;
                match hits.iter().position(|h| q.expect.contains(&h.path)) {
                    Some(rank) => {
                        found += 1;
                        rr += 1.0 / (rank as f64 + 1.0);
                    }
                    None => misses.push(q.q.clone()),
                }
            }
            let n = set.questions.len().max(1) as f64;
            let score = ModeScore {
                recall_at_k: round3(found as f64 / n),
                mrr: round3(rr / n),
            };
            report.modes.insert(mode.name().into(), score);
            report.misses.insert(mode.name().into(), misses);
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::tests::Fixture;

    #[test]
    fn scores_hits_and_misses() {
        let f = Fixture::new();
        let mut idx = f.open(true);
        idx.build().unwrap();
        let set: EvalSet = serde_json::from_str(
            r#"{"questions":[{"q":"moonraker 7125","expect":["systems/fabrication/qidi.md"]},{"q":"zzz","expect":["nope.md"]}]}"#,
        )
        .unwrap();
        let report = idx.evaluate(&set, 1, false).unwrap();
        assert_eq!(
            report.modes["hybrid"],
            ModeScore {
                recall_at_k: 0.5,
                mrr: 0.5
            }
        );
        assert_eq!(report.misses["hybrid"], vec!["zzz".to_string()]);
        assert!(!report.rerank);
        assert_eq!(report.modes.len(), 3);

        let g = Fixture::new();
        let mut keyword_only = g.open(false);
        keyword_only.build().unwrap();
        let report = keyword_only.evaluate(&set, 1, true).unwrap();
        assert!(!report.modes.contains_key("dense"));
    }
}
