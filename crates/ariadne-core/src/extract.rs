//! Format support: which file extensions a source reads, and the extractor that turns a file's
//! bytes into a chunkable [`Document`]. A source chooses its formats; the build dispatches each
//! file to the extractor registered for its extension. Markdown and EPUB are the registered
//! extractors; PDF (#155) will register itself here.

use crate::chunk::{markdown_document, Document};
use anyhow::{bail, Result};
use std::path::Path;

/// One format an index can read: its id, the extensions it covers, the largest file it can
/// read, how an oversized file is handled, and the extractor that turns a matching file into a
/// [`Document`]. An extractor error becomes a skip of that file (#150).
#[derive(Clone, Copy, Debug)]
pub struct Format {
    pub id: &'static str,
    pub extensions: &'static [&'static str],
    /// The largest file this format reads, in bytes.
    pub max_bytes: u64,
    /// Whether an oversized file reaches the build, which then skips it with a reason (#150),
    /// or is dropped by the walk as though it were not there (the markdown behaviour).
    pub report_oversize: bool,
    pub extract: fn(&str, &[u8]) -> Result<Document>,
}

/// The markdown extractor: the bytes are UTF-8 text, parsed into a [`Document`] by the
/// heading-based markdown parser.
pub fn markdown(_path: &str, bytes: &[u8]) -> Result<Document> {
    let text = String::from_utf8_lossy(bytes);
    Ok(markdown_document(_path, &text))
}

/// Every format this build knows about. New formats (PDF) are registered here.
pub fn formats() -> Vec<Format> {
    vec![
        Format {
            id: "md",
            extensions: &["md"],
            max_bytes: crate::source::MAX_FILE_BYTES,
            report_oversize: false,
            extract: markdown,
        },
        Format {
            id: "epub",
            extensions: &["epub"],
            max_bytes: 200 * 1024 * 1024,
            report_oversize: true,
            extract: crate::epub::epub,
        },
    ]
}

/// Render a [`Document`] back into Markdown: `# Title`, then each section in reading order,
/// under only the trail headings that are new since the previous section, then its trimmed
/// body. `read_document` uses this so a read of a non-Markdown file returns the book's text
/// rather than the container's bytes.
pub(crate) fn document_markdown(document: &Document) -> String {
    let mut blocks = Vec::new();
    if !document.title.trim().is_empty() {
        blocks.push(format!("# {}", document.title));
    }
    let mut trail: Vec<String> = Vec::new();
    for section in &document.sections {
        let shared = trail
            .iter()
            .zip(section.headings.iter())
            .take_while(|(was, now)| was == now)
            .count();
        // A heading's depth in the trail sets its level: depth 1 is `##`, depth 2 is `###`.
        for (depth, heading) in section.headings.iter().enumerate().skip(shared) {
            blocks.push(format!("{} {heading}", "#".repeat((depth + 2).min(6))));
        }
        // An empty section adds no text, but its headings still move the trail, so the next
        // section nests under them rather than repeating them.
        trail = section.headings.clone();
        let body = section.body.trim();
        if !body.is_empty() {
            blocks.push(body.to_string());
        }
    }
    let text = collapse_blank_lines(&blocks.join("\n\n"));
    match text.is_empty() {
        true => String::new(),
        false => format!("{text}\n"),
    }
}

/// Collapse runs of three or more newlines to a single blank line.
fn collapse_blank_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut newlines = 0usize;
    for c in text.chars() {
        if c == '\n' {
            newlines += 1;
            continue;
        }
        if newlines > 0 {
            out.push_str(&"\n".repeat(newlines.min(2)));
            newlines = 0;
        }
        out.push(c);
    }
    if newlines > 0 {
        out.push_str(&"\n".repeat(newlines.min(2)));
    }
    out
}

/// The format covering a path's extension, if any enabled format covers it.
pub(crate) fn covering_format(formats: &[Format], path: &str) -> Option<Format> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    formats
        .iter()
        .find(|f| f.extensions.iter().any(|e| e.eq_ignore_ascii_case(&ext)))
        .copied()
}

/// Turn one file's bytes into a [`Document`], dispatching on its extension. A file whose
/// extension is covered by none of `formats` is an error: it belongs to no enabled format.
pub fn extract(path: &str, bytes: &[u8], formats: &[Format]) -> Result<Document> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    for format in formats {
        if format
            .extensions
            .iter()
            .any(|e| e.to_ascii_lowercase() == ext)
        {
            return (format.extract)(path, bytes);
        }
    }
    bail!("no format covers {}", path)
}

/// Parse the `--formats` / `ARIADNE_FORMATS` value (a comma-separated list) into the matching
/// formats, defaulting to markdown. An unknown value is a startup error that lists the supported
/// values.
pub fn enabled_formats(value: &str) -> Result<Vec<Format>> {
    let all = formats();
    let tokens: Vec<&str> = value
        .split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect();
    let picked: Vec<&str> = if tokens.is_empty() {
        vec!["md"]
    } else {
        tokens
    };
    let mut out = Vec::new();
    for id in picked {
        match all.iter().find(|f| f.id == id) {
            Some(f) => out.push(*f),
            None => {
                let supported: Vec<&str> = all.iter().map(|f| f.id).collect();
                bail!(
                    "unknown format {id}; supported formats are: {}",
                    supported.join(", ")
                )
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{chunk_document, Section};

    /// A test-only extractor that proves the dispatch: one section, body the file's text.
    fn fake(_path: &str, bytes: &[u8]) -> Result<Document> {
        Ok(Document {
            title: "fake".to_string(),
            sections: vec![Section {
                headings: vec!["fake".to_string()],
                body: String::from_utf8_lossy(bytes).into_owned(),
            }],
        })
    }

    fn with_fake() -> Vec<Format> {
        let mut formats = formats();
        formats.push(Format {
            id: "fake",
            extensions: &["fke"],
            max_bytes: crate::source::MAX_FILE_BYTES,
            report_oversize: false,
            extract: fake,
        });
        formats
    }

    #[test]
    fn default_is_markdown() {
        let default = enabled_formats("").unwrap();
        assert_eq!(default.iter().map(|f| f.id).collect::<Vec<_>>(), vec!["md"]);
        assert_eq!(enabled_formats("md").unwrap().len(), 1);
    }

    #[test]
    fn epub_is_a_registered_format() {
        let formats = enabled_formats("epub").unwrap();
        assert_eq!(
            formats.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec!["epub"]
        );
    }

    #[test]
    fn an_unknown_format_fails_the_parse_with_the_supported_list() {
        let err = enabled_formats("pdf").unwrap_err().to_string();
        assert!(err.contains("unknown format pdf"), "{err}");
        assert!(
            err.contains("md"),
            "supported values should be listed: {err}"
        );
    }

    fn doc(title: &str, sections: &[(&[&str], &str)]) -> Document {
        Document {
            title: title.to_string(),
            sections: sections
                .iter()
                .map(|(headings, body)| Section {
                    headings: headings.iter().map(|h| (*h).to_string()).collect(),
                    body: (*body).to_string(),
                })
                .collect(),
        }
    }

    #[test]
    fn document_markdown_renders_the_title_then_trails_and_bodies_in_order() {
        let text = document_markdown(&doc(
            "Replication Basics",
            &[
                (
                    &["Chapter 5 Replication"],
                    "\n Print a calibration cube first. \n",
                ),
                (
                    &["Chapter 5 Replication", "Belt tension"],
                    "Tight but not too tight.",
                ),
                (&["Chapter 1 Setup"], "Plug in the press."),
                (&["Chapter 1 Setup", "Power"], "   "),
            ],
        ));
        assert_eq!(
            text,
            "# Replication Basics\n\n## Chapter 5 Replication\n\nPrint a calibration cube first.\n\n### Belt tension\n\nTight but not too tight.\n\n## Chapter 1 Setup\n\nPlug in the press.\n\n### Power\n"
        );
    }

    #[test]
    fn document_markdown_emits_a_repeated_heading_once_and_keeps_the_trail() {
        let text = document_markdown(&doc(
            "Notes",
            &[
                (&["Fabrication", "Belt tension"], "one"),
                (&["Fabrication", "Belt tension"], "two"),
                // The empty section contributes no text but updates the trail, so `Tension`
                // nests under `Bed leveling` instead of being re-labelled.
                (&["Fabrication", "Bed leveling"], "   "),
                (&["Fabrication", "Bed leveling", "Tension"], "three"),
            ],
        ));
        assert_eq!(
            text,
            "# Notes\n\n## Fabrication\n\n### Belt tension\n\none\n\ntwo\n\n### Bed leveling\n\n#### Tension\n\nthree\n"
        );
        assert_eq!(text.matches("## Fabrication").count(), 1);
    }

    #[test]
    fn document_markdown_of_an_empty_document_is_empty() {
        assert_eq!(document_markdown(&doc("", &[(&[], "   ")])), "");
    }

    #[test]
    fn deep_trails_stop_at_six_hashes() {
        let deep: Vec<&str> = (0..8).map(|_| "h").collect();
        let text = document_markdown(&doc("T", &[(&deep, "body")]));
        assert!(text.contains("###### h"), "{text}");
        assert!(!text.contains("#######"), "{text}");
    }

    #[test]
    fn dispatch_sends_each_file_to_its_format() {
        let formats = with_fake();
        // A markdown file goes to the markdown extractor and yields real chunks.
        let md = extract("a.md", b"# Title\n\nbody", &formats).unwrap();
        assert!(!chunk_document("a.md", &md, 1200).is_empty());
        // A `.fke` file goes to the test extractor: one chunk titled "fake".
        let fake_doc = extract("x.fke", b"hello", &formats).unwrap();
        let chunks = chunk_document("x.fke", &fake_doc, 1200);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].title, "fake");
        assert_eq!(chunks[0].body, "hello");
        // A `.txt` file matches no enabled format, so extraction fails cleanly.
        assert!(extract("note.txt", b"x", &formats).is_err());
    }
}
