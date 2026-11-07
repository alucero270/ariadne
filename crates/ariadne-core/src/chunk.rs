//! Markdown chunking by heading.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub path: String,
    pub title: String,
    pub heading: String,
    pub anchor: String,
    pub ord: usize,
    pub body: String,
    /// LLM-written text situating this chunk in its document; empty without a contextualizer.
    pub context: String,
}

impl Chunk {
    pub fn trail(&self) -> String {
        [self.title.as_str(), self.heading.as_str()]
            .iter()
            .filter(|s| !s.is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join(" > ")
    }

    /// The text embedded for dense search, and given to the contextualizer as the chunk.
    pub fn embed_text(&self) -> String {
        if self.context.is_empty() {
            format!("{}\n\n{}", self.trail(), self.body)
        } else {
            format!("{}\n\n{}\n\n{}", self.context, self.trail(), self.body)
        }
    }
}

/// A parsed document: a title and the sections under it, before any chunking. Other formats
/// produce one directly and hand it to [`chunk_document`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub title: String,
    pub sections: Vec<Section>,
}

/// One section of a [`Document`]: the heading trail it sits under, and its raw text. The body
/// is kept as written; [`chunk_document`] trims it and skips it when empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub headings: Vec<String>,
    pub body: String,
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

/// Turns Markdown into a [`Document`]: frontmatter stripping, fences, the H1-as-title rule and
/// the heading trail. Section bodies are kept as written, including empty ones; the chunker
/// decides what to skip.
pub fn markdown_document(path: &str, text: &str) -> Document {
    let normalized = text.replace("\r\n", "\n");
    let text = strip_frontmatter(&normalized);
    let mut title = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut title_from_h1 = false;
    let mut trail: Vec<(usize, String)> = Vec::new();
    let mut body: Vec<&str> = Vec::new();
    let mut sections: Vec<Section> = Vec::new();
    let mut in_fence = false;

    let flush = |trail: &[(usize, String)], body: &mut Vec<&str>, sections: &mut Vec<Section>| {
        sections.push(Section {
            headings: trail.iter().map(|(_, h)| h.clone()).collect(),
            body: body.join("\n"),
        });
        body.clear();
    };

    for line in text.split('\n') {
        if is_fence(line) {
            in_fence = !in_fence;
            body.push(line);
            continue;
        }
        let parsed = if in_fence { None } else { heading(line) };
        let Some((level, heading_text)) = parsed else {
            body.push(line);
            continue;
        };
        flush(&trail, &mut body, &mut sections);
        if level == 1 && !title_from_h1 {
            title = heading_text;
            title_from_h1 = true;
            trail.clear();
            continue;
        }
        trail.retain(|(lv, _)| *lv < level);
        trail.push((level, heading_text));
    }
    flush(&trail, &mut body, &mut sections);

    Document { title, sections }
}

/// Chunks a [`Document`]: empty sections are skipped, the last heading is the anchor, the whole
/// trail is the heading, and long sections split at paragraphs. Chunks are numbered in order.
pub fn chunk_document(path: &str, document: &Document, max_chars: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    for section in &document.sections {
        let content = section.body.trim();
        if content.is_empty() {
            continue;
        }
        let anchor = section.headings.last().cloned().unwrap_or_default();
        let heading = section.headings.join(" > ");
        for piece in split_long(content, max_chars) {
            chunks.push(Chunk {
                path: path.to_string(),
                title: document.title.clone(),
                heading: heading.clone(),
                anchor: anchor.clone(),
                ord: chunks.len(),
                body: piece,
                context: String::new(),
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

    fn md_chunks(path: &str, text: &str, max_chars: usize) -> Vec<Chunk> {
        chunk_document(path, &markdown_document(path, text), max_chars)
    }

    #[test]
    fn headings_titles_and_fences() {
        let chunks = md_chunks("systems/fab/qidi.md", DOC, 1200);
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
        let chunks = md_chunks("notes/todo.md", "#tag not heading\n\ntext", 1200);
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
        let chunks = md_chunks("a.md", &text, 600);
        assert!(chunks.len() > 1);
        assert!(chunks
            .iter()
            .all(|c| c.body.chars().count() <= 600 && c.heading == "S"));
    }

    #[test]
    fn oversized_paragraph_is_hard_split_on_char_boundaries() {
        let text = format!("# T\n\n{}", "é".repeat(1000));
        let chunks = md_chunks("a.md", &text, 300);
        assert_eq!(chunks.len(), 4);
        assert!(chunks.iter().all(|c| c.body.chars().count() <= 300));
    }

    #[test]
    fn crlf_is_normalized() {
        let chunks = md_chunks("a.md", "# T\r\n\r\n## S\r\n\r\nline\r\n", 1200);
        assert_eq!(chunks[0].body, "line");
    }

    #[test]
    fn markdown_document_keeps_title_fences_and_the_heading_trail() {
        let doc = markdown_document("systems/fab/qidi.md", DOC);
        assert_eq!(doc.title, "Qidi Plus 4");

        // A fenced block does not start a new section; its lines stay in the body of the
        // section it sits in, and the `#` line inside it is never a heading.
        let trails: Vec<_> = doc
            .sections
            .iter()
            .map(|s| s.headings.join(" > "))
            .collect();
        assert_eq!(
            trails,
            ["", "", "Network Placement", "Network Placement > Firewall"]
        );
        assert!(doc
            .sections
            .iter()
            .all(|s| !s.body.contains("type: device")));
        assert_eq!(doc.sections[1].body.trim(), "Intro paragraph.");
        assert!(doc.sections[2]
            .body
            .contains("IOT VLAN 30. Moonraker on 7125."));
        assert!(doc.sections[2].body.contains("# not a heading"));
        assert_eq!(doc.sections[3].body.trim(), "Only USER VLAN 20.");
    }

    #[test]
    fn markdown_document_titles_only_the_first_h1() {
        let doc = markdown_document("a.md", "# First\n\n# Second\n\ntext");
        assert_eq!(doc.title, "First");
        let last = doc.sections.last().unwrap();
        assert_eq!(last.headings, ["Second"]);
        assert_eq!(last.body.trim(), "text");
    }

    #[test]
    fn markdown_document_keeps_an_empty_section_for_a_heading_with_no_body() {
        let doc = markdown_document("only-heading.md", "# Only Heading\n");
        assert_eq!(doc.title, "Only Heading");
        assert!(!doc.sections.is_empty());
        assert!(doc.sections.iter().all(|s| s.body.trim().is_empty()));
    }

    #[test]
    fn chunk_document_chunks_a_handwritten_document_from_another_format() {
        let doc = Document {
            title: "Field Notes".to_string(),
            sections: vec![
                Section {
                    headings: vec![],
                    body: "Written on the road.".to_string(),
                },
                Section {
                    headings: vec!["Bridges".to_string(), "Spanning Tree".to_string()],
                    body: "\n  Loops are the enemy.  \n\n\n".to_string(),
                },
            ],
        };
        let chunks = chunk_document("library/field-notes.txt", &doc, 1200);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading, "");
        assert_eq!(chunks[0].anchor, "");
        assert_eq!(chunks[1].heading, "Bridges > Spanning Tree");
        assert_eq!(chunks[1].anchor, "Spanning Tree");
        assert_eq!(chunks[1].body, "Loops are the enemy.");
        assert!(chunks.iter().all(|c| c.title == "Field Notes"));
        assert_eq!(
            reference(&chunks[1].path, &chunks[1].anchor),
            "library/field-notes.txt#Spanning Tree"
        );
    }

    #[test]
    fn reference_keeps_the_extension_of_an_epub_citation() {
        // `reference` strips only `.md`: an epub citation keeps its extension.
        assert_eq!(
            reference("Library/Some Book.epub", "Chapter 5 Replication"),
            "Library/Some Book.epub#Chapter 5 Replication"
        );
    }

    #[test]
    fn chunk_document_skips_a_document_whose_only_section_is_empty() {
        let doc = Document {
            title: "Mostly Blank".to_string(),
            sections: vec![Section {
                headings: vec!["Empty".to_string()],
                body: "   \n\t\n".to_string(),
            }],
        };
        assert!(chunk_document("blank.md", &doc, 1200).is_empty());

        let from_markdown = markdown_document("blank.md", "# Empty\n\n   \n\t\n");
        assert!(chunk_document("blank.md", &from_markdown, 1200).is_empty());
    }

    #[test]
    fn chunk_document_orders_and_splits_its_chunks() {
        let doc = Document {
            title: "Long".to_string(),
            sections: vec![Section {
                headings: vec!["One".to_string()],
                body: "a".repeat(700),
            }],
        };
        let chunks = chunk_document("long.md", &doc, 300);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.body.chars().count() <= 300));
        let ords: Vec<_> = chunks.iter().map(|c| c.ord).collect();
        assert_eq!(ords, [0, 1, 2]);
    }

    /// A verbatim copy of `chunk_markdown` as it stood before the split into
    /// `markdown_document` and `chunk_document`. Only the equivalence test below uses it.
    fn chunk_markdown_before(path: &str, text: &str, max_chars: usize) -> Vec<Chunk> {
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
                    context: String::new(),
                });
            }
        }
        chunks
    }

    /// Every Markdown fixture used anywhere in this crate's tests, plus the corners the
    /// fixtures do not cover. All made-up text.
    fn markdown_fixtures() -> Vec<(&'static str, String, usize)> {
        let long_body: Vec<String> = (0..10)
            .map(|i| format!("paragraph {i} {}", "word ".repeat(50)))
            .collect();
        let printer = "# Qidi Plus 4\n\nIntro paragraph about the printer.\n\n## Network Placement\n\nIOT VLAN 30 at 192.168.30.10. Moonraker listens on port 7125.\n\n### Firewall\n\nOnly USER VLAN 20 may reach it.\n";
        let voice = "# Voice Agent\n\n## Latency\n\nWhisper large-v3-turbo transcribes speech; Qwen3-TTS speaks the reply.\n";
        let big = format!(
            "# Big\n\n## Alpha\n\nzebra marker\n\n## Beta\n\n{}\n",
            "word ".repeat(crate::index::EXPAND_FILE_CHARS / 4)
        );
        let mut fixtures: Vec<(&'static str, String, usize)> = vec![
            ("systems/fab/qidi.md", DOC.to_string(), 1200),
            (
                "notes/todo.md",
                "#tag not heading\n\ntext".to_string(),
                1200,
            ),
            (
                "a.md",
                format!("# T\n\n## S\n\n{}", long_body.join("\n\n")),
                600,
            ),
            ("a.md", format!("# T\n\n{}", "é".repeat(1000)), 300),
            ("a.md", "# T\r\n\r\n## S\r\n\r\nline\r\n".to_string(), 1200),
            ("printer.md", printer.to_string(), 1200),
            ("voice.md", voice.to_string(), 1200),
            (
                "voice.md",
                format!("{voice}\n## Barge-in\n\nInterruptions are supported.\n"),
                1200,
            ),
            ("big.md", big, 1200),
            ("new.md", "# New\n\nfresh text\n".to_string(), 1200),
            (
                "bread.md",
                "# Bread\n\n## Sourdough\n\nFeed the sourdough starter daily.\n".to_string(),
                1200,
            ),
            ("README.md", "# r".to_string(), 1200),
            ("docs/sub/a.md", "# a".to_string(), 1200),
            (".git/c.md", "# c".to_string(), 1200),
            ("x-artifacts/d.md", "# d".to_string(), 1200),
            ("outside.md", "# secret".to_string(), 1200),
        ];
        fixtures.extend_from_slice(&corners());
        fixtures
    }

    /// Corners the existing fixtures do not reach: empty input, unclosed frontmatter, a
    /// tilde fence, a heading with no body, and a file named only by an extension.
    fn corners() -> Vec<(&'static str, String, usize)> {
        vec![
            ("empty.md", String::new(), 1200),
            ("unclosed.md", "---\nunclosed".to_string(), 1200),
            (
                "tilde.md",
                "# T\n\n~~~\n# not a heading\n~~~\n\n## S\n\ntext\n".to_string(),
                1200,
            ),
            ("only-heading.md", "# Only Heading\n".to_string(), 1200),
            ("h6.md", "###### Deep\n\ntext\n".to_string(), 1200),
            (".md", "# No stem\n\ntext\n".to_string(), 1200),
            ("trail.md", "## A\n\n## B\n\ntext\n".to_string(), 1200),
            (
                "closed-hashes.md",
                "# Title #\n\n## Also ##\n\ntext\n".to_string(),
                1200,
            ),
        ]
    }

    #[test]
    fn chunk_markdown_is_identical_to_the_pre_split_implementation() {
        let fixtures = markdown_fixtures();
        assert!(fixtures.len() >= 20, "expected the crate's fixtures");
        for (path, text, max_chars) in &fixtures {
            let before = chunk_markdown_before(path, text, *max_chars);
            let after = md_chunks(path, text, *max_chars);
            assert_eq!(before, after, "chunks changed for {path}");
            assert!(!before.iter().any(|c| c.body.is_empty()));
        }
    }
}
