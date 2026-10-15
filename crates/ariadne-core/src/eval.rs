//! Retrieval evaluation: Recall@k and MRR over a question set with known answer files.

use crate::index::{Index, Mode, SearchOptions};
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

pub fn evaluate(
    index: &mut Index,
    set: &EvalSet,
    k: usize,
    rerank: bool,
    has_embedder: bool,
) -> Result<EvalReport> {
    let has_reranker = index.stats()?.reranker.is_some();
    let mut report = EvalReport {
        k,
        questions: set.questions.len(),
        rerank: rerank && has_reranker,
        modes: BTreeMap::new(),
        misses: BTreeMap::new(),
    };
    for (name, mode) in [
        ("bm25", Mode::Bm25),
        ("dense", Mode::Dense),
        ("hybrid", Mode::Hybrid),
    ] {
        if mode == Mode::Dense && !has_embedder {
            continue;
        }
        let opts = SearchOptions {
            k,
            mode,
            rerank,
            ..Default::default()
        };
        let (mut found, mut rr, mut misses) = (0usize, 0.0f64, Vec::new());
        for q in &set.questions {
            let hits = index.search(&q.q, &opts)?;
            match hits.iter().position(|h| q.expect.contains(&h.path)) {
                Some(rank) => {
                    found += 1;
                    rr += 1.0 / (rank as f64 + 1.0);
                }
                None => misses.push(q.q.clone()),
            }
        }
        let n = set.questions.len().max(1) as f64;
        report.modes.insert(
            name.into(),
            ModeScore {
                recall_at_k: round3(found as f64 / n),
                mrr: round3(rr / n),
            },
        );
        report.misses.insert(name.into(), misses);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::tests::{excludes, repo, HashEmbedder};

    #[test]
    fn scores_hits_and_misses() {
        let src = repo();
        let db = tempfile::tempdir().unwrap();
        let mut idx =
            Index::open(&db.path().join("i.db"), Some(Box::new(HashEmbedder)), None).unwrap();
        idx.build(src.path(), &excludes(), 1200).unwrap();
        let set: EvalSet = serde_json::from_str(
            r#"{"questions":[{"q":"moonraker 7125","expect":["systems/fabrication/qidi.md"]},{"q":"zzz","expect":["nope.md"]}]}"#,
        )
        .unwrap();
        let report = evaluate(&mut idx, &set, 1, false, true).unwrap();
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
    }
}
