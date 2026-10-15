//! Markdown chunking by heading.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub path: String,
    pub title: String,
    pub heading: String,
    pub anchor: String,
    pub ord: usize,
    pub body: String,
}

impl Chunk {
    pub fn context(&self) -> String {
        [self.title.as_str(), self.heading.as_str()]
            .iter()
            .filter(|s| !s.is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join(" > ")
    }

    pub fn embed_text(&self) -> String {
        format!("{}\n\n{}", self.context(), self.body)
    }
}

/// Wiki-style reference: the path without `.md`, plus `#anchor` when there is one.
pub fn reference(path: &str, anchor: &str) -> String {
    let stem = path.strip_suffix(".md").unwrap_or(path);
    if anchor.is_empty() {
        stem.to_string()
    } else {
        format!("{stem}#{anchor}")
    }
}

pub fn strip_frontmatter(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("---") else {
        return text;
    };
    let Some(first_nl) = rest.find('\n') else {
        return text;
    };
    if !rest[..first_nl].trim().is_empty() {
        return text;
    }
    let mut offset = 3 + first_nl + 1;
    for line in text[offset..].split_inclusive('\n') {
        offset += line.len();
        if line.trim_end_matches(['\n', '\r']).trim_end() == "---" {
            return &text[offset..];
        }
    }
    text
}

fn heading(line: &str) -> Option<(usize, String)> {
    let level = line.chars().take_while(|c| *c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &line[level..];
    if !rest.starts_with([' ', '\t']) {
        return None;
    }
    let text = rest.trim().trim_end_matches('#').trim_end();
    if text.is_empty() {
        None
    } else {
        Some((level, text.to_string()))
    }
}

fn is_fence(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn take_chars(s: &str, n: usize) -> (&str, &str) {
    match s.char_indices().nth(n) {
        Some((i, _)) => (&s[..i], &s[i..]),
        None => (s, ""),
    }
}

fn split_long(body: &str, max_chars: usize) -> Vec<String> {
    if char_len(body) <= max_chars {
        return vec![body.to_string()];
    }
    let mut pieces = Vec::new();
    let mut current = String::new();
    for para in split_paragraphs(body) {
        let mut para = para.as_str();
        while char_len(para) > max_chars {
            if !current.is_empty() {
                pieces.push(std::mem::take(&mut current));
            }
            let (head, tail) = take_chars(para, max_chars);
            pieces.push(head.to_string());
            para = tail;
        }
        if para.is_empty() {
            continue;
        }
        if !current.is_empty() && char_len(&current) + char_len(para) + 2 > max_chars {
            pieces.push(std::mem::replace(&mut current, para.to_string()));
        } else if current.is_empty() {
            current = para.to_string();
        } else {
            current.push_str("\n\n");
            current.push_str(para);
        }
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}

fn split_paragraphs(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in body.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                out.push(current.join("\n").trim().to_string());
                current.clear();
            }
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() {
        out.push(current.join("\n").trim().to_string());
    }
    out.retain(|p| !p.is_empty());
    out
}

pub fn chunk_markdown(path: &str, text: &str, max_chars: usize) -> Vec<Chunk> {
    let normalized = text.replace("\r\n", "\n");
    let text = strip_frontmatter(&normalized);
    let mut title = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut title_from_h1 = false;
    let mut trail: Vec<(usize, String)> = Vec::new();
    let mut body: Vec<&str> = Vec::new();
    let mut sections: Vec<(Vec<String>, String)> = Vec::new();
    let mut in_fence = false;

    let flush = |trail: &[(usize, String)], body: &mut Vec<&str>, sections: &mut Vec<_>| {
        let headings = trail.iter().map(|(_, h)| h.clone()).collect::<Vec<_>>();
        sections.push((headings, body.join("\n")));
        body.clear();
    };

    for line in text.split('\n') {
        if is_fence(line) {
            in_fence = !in_fence;
            body.push(line);
            continue;
        }
        let parsed = if in_fence { None } else { heading(line) };
        let Some((level, text)) = parsed else {
            body.push(line);
            continue;
        };
        flush(&trail, &mut body, &mut sections);
        if level == 1 && !title_from_h1 {
            title = text;
            title_from_h1 = true;
            trail.clear();
            continue;
        }
        trail.retain(|(lv, _)| *lv < level);
        trail.push((level, text));
    }
    flush(&trail, &mut body, &mut sections);

    let mut chunks = Vec::new();
    for (headings, content) in sections {
        let content = content.trim();
        if content.is_empty() {
            continue;
        }
        let anchor = headings.last().cloned().unwrap_or_default();
        let heading = headings.join(" > ");
        for piece in split_long(content, max_chars) {
            chunks.push(Chunk {
                path: path.to_string(),
                title: title.clone(),
                heading: heading.clone(),
                anchor: anchor.clone(),
                ord: chunks.len(),
                body: piece,
            });
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "---\ntype: device\n---\n\n# Qidi Plus 4\n\nIntro paragraph.\n\n## Network Placement\n\nIOT VLAN 30. Moonraker on 7125.\n\n```bash\n# not a heading\ncurl http://printer:7125\n```\n\n### Firewall\n\nOnly USER VLAN 20.\n";

    #[test]
    fn frontmatter_is_stripped_only_when_closed() {
        assert_eq!(strip_frontmatter("---\na: 1\n---\nbody"), "body");
        assert_eq!(strip_frontmatter("no\n---\n"), "no\n---\n");
        assert_eq!(strip_frontmatter("---\nunclosed"), "---\nunclosed");
    }

    #[test]
    fn headings_titles_and_fences() {
        let chunks = chunk_markdown("systems/fab/qidi.md", DOC, 1200);
        let headings: Vec<_> = chunks.iter().map(|c| c.heading.as_str()).collect();
        assert_eq!(
            headings,
            ["", "Network Placement", "Network Placement > Firewall"]
        );
        assert!(chunks.iter().all(|c| c.title == "Qidi Plus 4"));
        assert!(chunks[1].body.contains("# not a heading"));
        assert!(!chunks.iter().any(|c| c.body.contains("type: device")));
        assert_eq!(
            reference(&chunks[1].path, &chunks[1].anchor),
            "systems/fab/qidi#Network Placement"
        );
        assert_eq!(
            reference(&chunks[0].path, &chunks[0].anchor),
            "systems/fab/qidi"
        );
        assert_eq!(
            chunks[2].embed_text(),
            "Qidi Plus 4 > Network Placement > Firewall\n\nOnly USER VLAN 20."
        );
    }

    #[test]
    fn untitled_file_uses_stem_and_ignores_hashtags() {
        let chunks = chunk_markdown("notes/todo.md", "#tag not heading\n\ntext", 1200);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].title, "todo");
        assert_eq!(chunks[0].heading, "");
    }

    #[test]
    fn long_sections_split_at_paragraphs_and_stay_bounded() {
        let body: Vec<String> = (0..10)
            .map(|i| format!("paragraph {i} {}", "word ".repeat(50)))
            .collect();
        let text = format!("# T\n\n## S\n\n{}", body.join("\n\n"));
        let chunks = chunk_markdown("a.md", &text, 600);
        assert!(chunks.len() > 1);
        assert!(chunks
            .iter()
            .all(|c| c.body.chars().count() <= 600 && c.heading == "S"));
    }

    #[test]
    fn oversized_paragraph_is_hard_split_on_char_boundaries() {
        let text = format!("# T\n\n{}", "é".repeat(1000));
        let chunks = chunk_markdown("a.md", &text, 300);
        assert_eq!(chunks.len(), 4);
        assert!(chunks.iter().all(|c| c.body.chars().count() <= 300));
    }

    #[test]
    fn crlf_is_normalized() {
        let chunks = chunk_markdown("a.md", "# T\r\n\r\n## S\r\n\r\nline\r\n", 1200);
        assert_eq!(chunks[0].body, "line");
    }
}
