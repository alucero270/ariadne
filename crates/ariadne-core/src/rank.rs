//! Pure ranking: query sanitizing for FTS5, reciprocal rank fusion, and the per-document cap.

use std::collections::HashMap;

/// Reciprocal rank fusion constant. 60 is the value from the original RRF paper.
const RRF_K: f64 = 60.0;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "should", "so", "that",
    "the", "their", "there", "this", "to", "was", "we", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your",
];

/// Turn free text into an FTS5 query: distinct non-stopword terms, quoted, OR-ed.
/// Quoting makes every term a literal, so user text can never inject FTS5 syntax.
pub(crate) fn fts_query(query: &str) -> String {
    let lower = query.to_lowercase();
    let mut terms: Vec<&str> = Vec::new();
    for term in lower.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if !term.is_empty() && !STOPWORDS.contains(&term) && !terms.contains(&term) {
            terms.push(term);
        }
    }
    terms
        .iter()
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Fuse ranked id lists into one ordering, best first, ties broken by id.
pub(crate) fn fuse(rankings: &[&[i64]]) -> Vec<(i64, f64)> {
    let mut scores: HashMap<i64, f64> = HashMap::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }
    let mut fused: Vec<(i64, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused
}

/// Keep at most `cap` items per key, preserving order.
pub(crate) fn cap_per_key<T>(items: Vec<T>, cap: usize, key: impl Fn(&T) -> &str) -> Vec<T> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    items
        .into_iter()
        .filter(|item| {
            let n = seen.entry(key(item).to_string()).or_insert(0);
            *n += 1;
            *n <= cap
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_query_quotes_distinct_terms_and_drops_stopwords() {
        assert_eq!(
            fts_query("Which ports does the \"printer\" expose?"),
            "\"ports\" OR \"printer\" OR \"expose\""
        );
        assert_eq!(fts_query("the of and"), "");
        assert_eq!(fts_query("llama-swap llama"), "\"llama\" OR \"swap\"");
        // FTS5 operators become plain terms; "or" and "a" are stopwords.
        assert_eq!(fts_query("NEAR(a b) OR c*"), "\"near\" OR \"b\" OR \"c\"");
    }

    #[test]
    fn fusion_rewards_agreement_between_rankings() {
        // By hand: 3 = 1/61 + 1/63 = 0.032266, 2 = 2/62 = 0.032258, 1 = 1/61, 4 = 1/63.
        let fused = fuse(&[&[1, 2, 3], &[3, 2, 4]]);
        let order: Vec<i64> = fused.iter().map(|(id, _)| *id).collect();
        assert_eq!(order, vec![3, 2, 1, 4]);
    }

    #[test]
    fn fusion_breaks_ties_by_id() {
        let order: Vec<i64> = fuse(&[&[7], &[5]]).iter().map(|(id, _)| *id).collect();
        assert_eq!(order, vec![5, 7]);
    }

    #[test]
    fn cap_keeps_order_and_limits_each_key() {
        let items = vec![("a", 1), ("a", 2), ("b", 3), ("a", 4), ("b", 5)];
        assert_eq!(cap_per_key(items, 1, |i| i.0), vec![("a", 1), ("b", 3)]);
    }
}
