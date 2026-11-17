//! The EPUB extractor (#153). A book is a zip container: `META-INF/container.xml` locates the
//! package document (the OPF), the OPF's `spine` gives the reading order through `manifest`
//! hrefs, and each spine document becomes sections of one [`Document`] titled by the book, so
//! citations read `Library/Some Book.epub#Chapter 5 Replication`. Chapter titles come from the
//! EPUB 3 `nav` document when present, then the EPUB 2 NCX, then the first `h1` or `h2` in the
//! chapter, then its `title`, then the file stem.
//!
//! Limits (#150 turns a failure into a skip with a reason): over [`MAX_ENTRIES`] zip entries, an
//! archive that would expand past [`MAX_EXPANDED`], or encrypted content documents are refused.
//! A chapter that fails to parse is dropped; the book fails only when no chapter yields text.

use crate::chunk::{Document, Section};
use anyhow::{bail, Context, Result};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read};
use std::path::Path;

/// Largest number of entries the archive may contain.
const MAX_ENTRIES: usize = 10_000;
/// Largest total uncompressed size of every entry, in bytes.
const MAX_EXPANDED: u64 = 500 * 1024 * 1024;

/// One file entry: an id, where it lives, what it is, and its EPUB 3 properties.
#[derive(Debug, Default)]
struct Item {
    id: String,
    href: String,
    media_type: String,
    properties: Vec<String>,
}

/// The parts of the OPF this extractor reads.
#[derive(Debug, Default)]
struct Opf {
    title: Option<String>,
    items: Vec<Item>,
    /// Spine itemrefs, in reading order.
    spine: Vec<String>,
    /// The `toc` attribute of `<spine>`: the NCX item id (EPUB 2).
    toc: Option<String>,
}

/// A heading or a paragraph of chapter text, in document order.
#[derive(Debug, PartialEq)]
enum Block {
    Heading(u8, String),
    Paragraph(String),
}

/// One chapter's parsed content.
#[derive(Debug, Default)]
struct Chapter {
    /// The `<title>` element, used only when no better heading exists.
    title: Option<String>,
    blocks: Vec<Block>,
}

/// Extract an EPUB into a [`Document`]. Any failure is an error naming why, which `build`
/// records as a skip (#150).
pub fn epub(path: &str, bytes: &[u8]) -> Result<Document> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).context("not a readable zip archive")?;
    if zip.len() > MAX_ENTRIES {
        bail!("over 10,000 zip entries (found {})", zip.len());
    }
    let mut expanded: u64 = 0;
    for i in 0..zip.len() {
        let entry = zip.by_index(i)?;
        expanded = expanded.saturating_add(entry.size());
    }
    if expanded > MAX_EXPANDED {
        bail!(
            "book expands past the {} MB limit",
            MAX_EXPANDED / (1024 * 1024)
        );
    }

    let container = read_entry(&mut zip, "META-INF/container.xml")
        .context("no META-INF/container.xml: not an EPUB")?;
    let opf_path = opf_path(&container)?;
    let opf_bytes =
        read_entry(&mut zip, &opf_path).with_context(|| format!("no OPF at {opf_path}"))?;
    let opf = parse_opf(&opf_bytes)?;
    let opf_dir = parent_dir(&opf_path);

    // The spine is the reading order, which need not be the file order.
    let spine: Vec<String> = opf
        .spine
        .iter()
        .filter_map(|id| opf.items.iter().find(|item| &item.id == id))
        .map(|item| resolve(&opf_dir, &item.href))
        .collect();
    if spine.is_empty() {
        bail!("the OPF spine references no manifest items");
    }

    drm_check(&mut zip, &opf, &opf_dir, &opf_path)?;

    // Chapter titles: the EPUB 3 nav document when present, else the EPUB 2 NCX. Chapters the
    // table of contents does not name fall back to their own headings.
    let nav = opf
        .items
        .iter()
        .find(|item| item.properties.iter().any(|p| p == "nav"));
    let titles = match nav {
        Some(item) => {
            let nav_path = resolve(&opf_dir, &item.href);
            read_entry(&mut zip, &nav_path)
                .map(|bytes| nav_titles(&bytes, &parent_dir(&nav_path)))
                .unwrap_or_default()
        }
        None => ncx_titles(&mut zip, &opf, &opf_dir),
    };

    let book_title = opf
        .title
        .filter(|t| !t.trim().is_empty())
        .unwrap_or_else(|| file_stem(path));

    // Each spine document contributes its sections, in spine order. A chapter that fails to
    // parse is dropped and counted; the book fails only if nothing yields text.
    let mut sections = Vec::new();
    let mut dropped = 0usize;
    let mut produced = false;
    for doc in &spine {
        let parsed = read_entry(&mut zip, doc)
            .and_then(|bytes| parse_xhtml(&bytes).with_context(|| format!("parsing {doc}")));
        match parsed {
            Ok(chapter) => {
                let chapter_sections = chapter_sections(chapter, titles.get(doc), doc);
                if chapter_sections.iter().any(|s| !s.body.trim().is_empty()) {
                    produced = true;
                    sections.extend(chapter_sections);
                }
            }
            Err(e) => {
                eprintln!("warning: dropping chapter {doc}: {e}");
                dropped += 1;
            }
        }
    }
    if !produced {
        bail!("no chapters produced text ({dropped} dropped)");
    }
    Ok(Document {
        title: book_title,
        sections,
    })
}

/// Read one archive entry with a cap: the declared size plus slack, never unbounded.
fn read_entry<R: Read + std::io::Seek>(
    zip: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<Vec<u8>> {
    let entry = zip.by_name(name)?;
    let cap = entry.size().saturating_add(1024).min(MAX_EXPANDED);
    let mut buf = Vec::new();
    entry.take(cap).read_to_end(&mut buf)?;
    Ok(buf)
}

/// `META-INF/container.xml` -> the OPF path from the first `<rootfile full-path="...">`.
fn opf_path(bytes: &[u8]) -> Result<String> {
    let text = std::str::from_utf8(bytes).context("container.xml is not UTF-8")?;
    let mut reader = Reader::from_str(text);
    loop {
        match reader.read_event().context("reading container.xml")? {
            Event::Eof => break,
            Event::Start(e) | Event::Empty(e) if e.local_name().as_ref() == b"rootfile" => {
                if let Some(value) = attr(&e, b"full-path") {
                    if !value.trim().is_empty() {
                        return Ok(normalize(&value));
                    }
                }
            }
            _ => {}
        }
    }
    bail!("container.xml has no rootfile full-path")
}

/// Parse the OPF's metadata title, manifest items, and spine order.
fn parse_opf(bytes: &[u8]) -> Result<Opf> {
    let text = std::str::from_utf8(bytes).context("the OPF is not UTF-8")?;
    let mut reader = Reader::from_str(text);
    let mut opf = Opf::default();
    let mut in_metadata = false;
    let mut in_title = false;
    let mut title_text = String::new();
    loop {
        match reader.read_event().context("reading the OPF")? {
            Event::Eof => break,
            Event::Start(e) => match e.local_name().as_ref() {
                b"metadata" => in_metadata = true,
                b"spine" => opf.toc = attr(&e, b"toc"),
                b"item" if !in_metadata => opf.items.push(parse_item(&e)?),
                b"itemref" => {
                    if let Some(id) = attr(&e, b"idref") {
                        opf.spine.push(id);
                    }
                }
                b"title" if in_metadata && opf.title.is_none() => {
                    in_title = true;
                    title_text.clear();
                }
                _ => {}
            },
            Event::Empty(e) => match e.local_name().as_ref() {
                b"item" if !in_metadata => opf.items.push(parse_item(&e)?),
                b"itemref" => {
                    if let Some(id) = attr(&e, b"idref") {
                        opf.spine.push(id);
                    }
                }
                _ => {}
            },
            Event::Text(e) if in_title => title_text.push_str(&String::from_utf8_lossy(e.as_ref())),
            Event::End(e) => match e.local_name().as_ref() {
                b"metadata" => in_metadata = false,
                b"title" if in_title => {
                    in_title = false;
                    let title = unescape(&collapse(&title_text));
                    if !title.is_empty() {
                        opf.title = Some(title);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    if opf.spine.is_empty() {
        bail!("the OPF has no itemrefs");
    }
    Ok(opf)
}

fn parse_item(e: &quick_xml::events::BytesStart<'_>) -> Result<Item> {
    Ok(Item {
        id: attr(e, b"id").unwrap_or_default(),
        href: attr(e, b"href").unwrap_or_default(),
        media_type: attr(e, b"media-type").unwrap_or_default(),
        properties: attr(e, b"properties")
            .map(|p| p.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default(),
    })
}

/// DRM (#153): if `META-INF/encryption.xml` encrypts a content document, the book is
/// unusable. An archive that only encrypts metadata (the OPF, the NCX, the nav, fonts) is not
/// flagged.
fn drm_check<R: Read + std::io::Seek>(
    zip: &mut zip::ZipArchive<R>,
    opf: &Opf,
    opf_dir: &str,
    opf_path: &str,
) -> Result<()> {
    let xml = match read_entry(zip, "META-INF/encryption.xml") {
        Ok(xml) => xml,
        Err(_) => return Ok(()),
    };
    let text = String::from_utf8_lossy(&xml);
    let mut reader = Reader::from_str(&text);
    let mut targets = Vec::new();
    let mut in_encrypted = false;
    loop {
        match reader.read_event().context("reading encryption.xml")? {
            Event::Eof => break,
            Event::Start(e) if e.local_name().as_ref() == b"EncryptedData" => in_encrypted = true,
            Event::End(e) if e.local_name().as_ref() == b"EncryptedData" => in_encrypted = false,
            Event::Start(e) | Event::Empty(e) if in_encrypted => {
                for key in [b"URI".as_slice(), b"href".as_slice()] {
                    if let Some(value) = attr(&e, key) {
                        targets.push(normalize(&value));
                    }
                }
            }
            _ => {}
        }
    }
    if targets.is_empty() {
        return Ok(());
    }

    // The content documents: manifest items that are real reading content, not the OPF, the
    // table-of-contents documents, or container metadata.
    let toc_id = opf.toc.clone().unwrap_or_default();
    let content: HashSet<String> = opf
        .items
        .iter()
        .filter(|item| item.id != toc_id && !item.properties.iter().any(|p| p == "nav"))
        .map(|item| resolve(opf_dir, &item.href))
        .filter(|p| p != opf_path && !p.starts_with("META-INF/") && p != "mimetype")
        .collect();
    // URIs resolve relative to the container root; resolve from META-INF too, in case the
    // author wrote them relative to the encryption file.
    for target in targets {
        for base in ["", "META-INF/"] {
            if content.contains(&resolve(base, &target)) {
                bail!("DRM-protected");
            }
        }
    }
    Ok(())
}

/// The EPUB 2 NCX: the item the spine's `toc` names, or any `.ncx` item. Maps each chapter
/// path to its `navPoint` text.
fn ncx_titles<R: Read + std::io::Seek>(
    zip: &mut zip::ZipArchive<R>,
    opf: &Opf,
    opf_dir: &str,
) -> HashMap<String, String> {
    let mut candidates: Vec<String> = opf
        .items
        .iter()
        .filter(|item| {
            Some(item.id.as_str()) == opf.toc.as_deref()
                || item.media_type == "application/x-dtbncx+xml"
                || item.href.ends_with(".ncx")
        })
        .map(|item| resolve(opf_dir, &item.href))
        .collect();
    candidates.sort();
    candidates.dedup();
    for candidate in candidates {
        if let Ok(bytes) = read_entry(zip, &candidate) {
            let titles = ncx_navmap(&bytes, &parent_dir(&candidate));
            if !titles.is_empty() {
                return titles;
            }
        }
    }
    HashMap::new()
}

fn ncx_navmap(bytes: &[u8], base_dir: &str) -> HashMap<String, String> {
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => return HashMap::new(),
    };
    let mut reader = Reader::from_str(text);
    reader.config_mut().check_end_names = false;
    let mut out = HashMap::new();
    // navPoints nest; each open one collects its own (src, label).
    let mut points: Vec<(String, String)> = Vec::new();
    let mut in_text = false;
    loop {
        let event = match reader.read_event() {
            Ok(event) => event,
            Err(_) => return HashMap::new(),
        };
        match event {
            Event::Eof => break,
            Event::Start(e) => match e.local_name().as_ref() {
                b"navPoint" => points.push((String::new(), String::new())),
                b"text" if !points.is_empty() => in_text = true,
                b"content" if !points.is_empty() => {
                    if let Some(src) = attr(&e, b"src") {
                        if let Some(last) = points.last_mut() {
                            last.0 = src;
                        }
                    }
                }
                _ => {}
            },
            Event::Empty(e) if !points.is_empty() && e.local_name().as_ref() == b"content" => {
                if let Some(src) = attr(&e, b"src") {
                    if let Some(last) = points.last_mut() {
                        last.0 = src;
                    }
                }
            }
            Event::Text(e) if in_text => {
                if let Some(last) = points.last_mut() {
                    last.1.push_str(&String::from_utf8_lossy(e.as_ref()));
                }
            }
            Event::End(e) if e.local_name().as_ref() == b"text" => in_text = false,
            Event::End(e) if e.local_name().as_ref() == b"navPoint" => {
                if let Some((src, label)) = points.pop() {
                    let label = unescape(&collapse(&label));
                    if !src.is_empty() && !label.is_empty() {
                        out.entry(resolve(base_dir, &src)).or_insert(label);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// The EPUB 3 nav document: every `<nav> ... <a href>` pair.
fn nav_titles(bytes: &[u8], base_dir: &str) -> HashMap<String, String> {
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => return HashMap::new(),
    };
    let mut reader = Reader::from_str(text);
    reader.config_mut().check_end_names = false;
    let mut out = HashMap::new();
    let mut nav_depth = 0usize;
    let mut href: Option<String> = None;
    let mut label = String::new();
    loop {
        let event = match reader.read_event() {
            Ok(event) => event,
            Err(_) => return out,
        };
        match event {
            Event::Eof => break,
            Event::Start(e) if e.local_name().as_ref() == b"nav" => nav_depth += 1,
            Event::Start(e) if nav_depth > 0 && e.local_name().as_ref() == b"a" => {
                href = attr(&e, b"href");
                label.clear();
            }
            Event::Text(e) if href.is_some() => {
                label.push_str(&String::from_utf8_lossy(e.as_ref()));
            }
            Event::End(e) if e.local_name().as_ref() == b"nav" => nav_depth -= 1,
            Event::End(e) if nav_depth > 0 && e.local_name().as_ref() == b"a" => {
                if let Some(href) = href.take() {
                    let label = unescape(&collapse(&label));
                    if !href.is_empty() && !label.is_empty() {
                        out.entry(resolve(base_dir, &href)).or_insert(label);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Chapter text from XHTML: `script` and `style` subtrees dropped, entities decoded,
/// paragraph breaks kept, `h1` to `h6` recognized as headings.
fn parse_xhtml(bytes: &[u8]) -> Result<Chapter> {
    let text = std::str::from_utf8(bytes).context("chapter is not UTF-8")?;
    let mut reader = Reader::from_str(text);
    reader.config_mut().check_end_names = false;
    let mut chapter = Chapter::default();
    let mut stack: Vec<Vec<u8>> = Vec::new();
    let mut skip_depth: Option<usize> = None;
    let mut heading: Option<(u8, String)> = None;
    let mut para = String::new();
    let mut title_raw = String::new();
    let mut in_title = false;
    loop {
        match reader.read_event().context("malformed chapter")? {
            Event::Eof => break,
            Event::Start(e) => {
                let name = e.local_name().as_ref().to_vec();
                stack.push(name.clone());
                if name == b"script" || name == b"style" {
                    skip_depth.get_or_insert(stack.len());
                    continue;
                }
                if skip_depth.is_some() {
                    continue;
                }
                if is_block(&name) {
                    flush_para(&mut para, &mut chapter.blocks);
                }
                if let Some(level) = heading_level(&name) {
                    heading = Some((level, String::new()));
                } else if name == b"title" {
                    in_title = true;
                    title_raw.clear();
                }
            }
            Event::Empty(e) => {
                if skip_depth.is_some() {
                    continue;
                }
                let name = e.local_name().as_ref().to_vec();
                if name == b"br" {
                    if heading.is_some() {
                        if let Some((_, text)) = heading.as_mut() {
                            text.push('\n');
                        }
                    } else {
                        para.push('\n');
                    }
                } else if is_block(&name) {
                    flush_para(&mut para, &mut chapter.blocks);
                    if let Some(level) = heading_level(&name) {
                        // An empty heading element: keep its text only, which is none.
                        heading = Some((level, String::new()));
                        flush_heading(&mut heading, &mut chapter.blocks);
                    }
                }
            }
            Event::Text(e) => {
                if skip_depth.is_some() {
                    continue;
                }
                let raw = String::from_utf8_lossy(e.as_ref()).into_owned();
                if in_title {
                    title_raw.push_str(&raw);
                } else if let Some((_, text)) = heading.as_mut() {
                    text.push_str(&raw);
                } else {
                    para.push_str(&raw);
                }
            }
            Event::End(e) => {
                let name = e.local_name().as_ref().to_vec();
                if stack.len() == skip_depth.unwrap_or(usize::MAX) {
                    skip_depth = None;
                }
                if let Some(pos) = stack.iter().rposition(|n| *n == name) {
                    stack.truncate(pos);
                }
                if skip_depth.is_some() {
                    continue;
                }
                if name == b"title" && in_title {
                    in_title = false;
                    let title = unescape(&collapse(&title_raw));
                    if !title.is_empty() {
                        chapter.title = Some(title);
                    }
                    continue;
                }
                if let Some(expected) = heading_level(&name) {
                    if heading
                        .as_ref()
                        .is_some_and(|(level, _)| *level == expected)
                    {
                        flush_heading(&mut heading, &mut chapter.blocks);
                        continue;
                    }
                }
                if is_block(&name) {
                    flush_para(&mut para, &mut chapter.blocks);
                }
            }
            _ => {}
        }
    }
    flush_heading(&mut heading, &mut chapter.blocks);
    flush_para(&mut para, &mut chapter.blocks);
    Ok(chapter)
}

fn heading_level(name: &[u8]) -> Option<u8> {
    match name {
        b"h1" => Some(1),
        b"h2" => Some(2),
        b"h3" => Some(3),
        b"h4" => Some(4),
        b"h5" => Some(5),
        b"h6" => Some(6),
        _ => None,
    }
}

fn is_block(name: &[u8]) -> bool {
    matches!(
        name,
        b"p" | b"div"
            | b"li"
            | b"ul"
            | b"ol"
            | b"blockquote"
            | b"section"
            | b"article"
            | b"header"
            | b"footer"
            | b"figure"
            | b"figcaption"
            | b"table"
            | b"tr"
            | b"td"
            | b"pre"
            | b"h1"
            | b"h2"
            | b"h3"
            | b"h4"
            | b"h5"
            | b"h6"
    )
}

fn flush_heading(heading: &mut Option<(u8, String)>, blocks: &mut Vec<Block>) {
    if let Some((level, text)) = heading.take() {
        let text = unescape(&collapse(&text));
        if !text.is_empty() {
            blocks.push(Block::Heading(level, text));
        }
    }
}

fn flush_para(para: &mut String, blocks: &mut Vec<Block>) {
    let text = unescape(&collapse(para));
    para.clear();
    if !text.is_empty() {
        blocks.push(Block::Paragraph(text));
    }
}

/// Turn one chapter's blocks into sections: the chapter title is the top heading, and `h1`
/// to `h3` inside it nest below.
fn chapter_sections(chapter: Chapter, mapped: Option<&String>, doc_path: &str) -> Vec<Section> {
    let title = mapped
        .filter(|t| !t.trim().is_empty())
        .cloned()
        .or_else(|| {
            chapter.blocks.iter().find_map(|block| match block {
                Block::Heading(level, text) if *level <= 2 => Some(text.clone()),
                _ => None,
            })
        })
        .or_else(|| chapter.title.clone())
        .unwrap_or_else(|| file_stem(doc_path));

    let mut sections = Vec::new();
    let mut trail: Vec<(u8, String)> = Vec::new();
    let mut body: Vec<String> = Vec::new();
    let mut seen_heading = false;
    for block in chapter.blocks {
        match block {
            Block::Heading(level, text) if level <= 3 => {
                // The chapter's own top heading names the section; it is not also a level
                // below itself.
                if !seen_heading && text == title {
                    seen_heading = true;
                    continue;
                }
                seen_heading = true;
                push_section(&mut sections, &title, &trail, &mut body);
                trail.retain(|(l, _)| *l < level);
                trail.push((level, text));
            }
            // h4 and deeper are not deep enough to chunk by: they stay body text.
            Block::Heading(_, text) | Block::Paragraph(text) => body.push(text),
        }
    }
    push_section(&mut sections, &title, &trail, &mut body);
    sections
}

fn push_section(
    sections: &mut Vec<Section>,
    title: &str,
    trail: &[(u8, String)],
    body: &mut Vec<String>,
) {
    if !body.is_empty() {
        sections.push(Section {
            headings: std::iter::once(title.to_string())
                .chain(trail.iter().map(|(_, heading)| heading.clone()))
                .collect(),
            body: join_paragraphs(body),
        });
    }
    body.clear();
}

fn join_paragraphs(parts: &[String]) -> String {
    let joined = parts.join("\n\n");
    let mut out = String::with_capacity(joined.len());
    let mut newlines = 0usize;
    for c in joined.chars() {
        if c == '\n' {
            newlines += 1;
            continue;
        }
        if newlines > 0 {
            for _ in 0..newlines.min(2) {
                out.push('\n');
            }
            newlines = 0;
        }
        out.push(c);
    }
    for _ in 0..newlines.min(2) {
        out.push('\n');
    }
    out
}

/// A one-attribute lookup on an open tag.
fn attr(e: &quick_xml::events::BytesStart<'_>, key: &[u8]) -> Option<String> {
    for a in e.attributes().with_checks(false).flatten() {
        if a.key.local_name().as_ref() == key {
            return Some(String::from_utf8_lossy(a.value.as_ref()).into_owned());
        }
    }
    None
}

/// Collapse runs of whitespace within lines, keep line breaks, trim the ends.
fn collapse(text: &str) -> String {
    text.split('\n')
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Decode XML and the common HTML named entities, and numeric character references. The
/// XHTML of real books carries HTML entities (`&nbsp;`) that plain XML would reject.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        match after.find(';') {
            Some(end) if end <= 10 => {
                if let Some(c) = entity(&after[..end]) {
                    out.push(c);
                    rest = &after[end + 1..];
                    continue;
                }
            }
            _ => {}
        }
        out.push('&');
        rest = after;
    }
    out.push_str(rest);
    out
}

fn entity(name: &str) -> Option<char> {
    let decoded = match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        "mdash" => '\u{2014}',
        "ndash" => '\u{2013}',
        "hellip" => '\u{2026}',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "ldquo" => '\u{201c}',
        "rdquo" => '\u{201d}',
        "copy" => '\u{00a9}',
        _ => {
            let digits = name.strip_prefix("#x").or(name.strip_prefix("#X"));
            match digits {
                Some(hex) => u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)?,
                None => name
                    .strip_prefix('#')
                    .and_then(|d| d.parse::<u32>().ok())
                    .and_then(char::from_u32)?,
            }
        }
    };
    Some(decoded)
}

/// Percent-decode an href, strip its fragment and query, and resolve it against the base
/// directory; then normalize `.` and `..` segments.
fn resolve(base_dir: &str, href: &str) -> String {
    let href = href.split(['#', '?']).next().unwrap_or("");
    let decoded = percent_decode(href);
    if decoded.starts_with('/') {
        normalize(&decoded)
    } else {
        normalize(&format!("{base_dir}{decoded}"))
    }
}

fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    parts.join("/")
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) =
                u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""), 16)
            {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn parent_dir(path: &str) -> String {
    match path.rsplit_once('/') {
        Some((dir, _)) => format!("{dir}/"),
        None => String::new(),
    }
}

fn file_stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Test fixtures: small made-up EPUB archives built with the zip writer. Nothing here is
/// third-party content; every book is fictional and every path is invented.
#[cfg(test)]
pub(crate) mod fixtures {
    use std::io::{Cursor, Write};
    use std::path::Path;
    use zip::write::SimpleFileOptions;
    use zip::{CompressionMethod, ZipWriter};

    pub const CONTAINER: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"#;

    /// The fake book "Replication Basics". The spine lists chapter 5 before chapter 1, so the
    /// reading order differs from the order the files were written into the archive.
    pub fn opf3(items: &str, spine: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">urn:uuid:00000000-0000-0000-0000-000000000000</dc:identifier>
    <dc:title>Replication Basics</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
{items}  </manifest>
  <spine>
{spine}  </spine>
</package>
"#
        )
    }

    pub const C1: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>One</title></head>
<body>
<h1>Chapter 1 Setup</h1>
<p>Plug in the press &amp; level the bed.</p>
<p>Second paragraph about the bed.</p>
<script>var secret = "not text";</script>
<style>h1 { color: red }</style>
<h2>Power</h2>
<p>230V only &#8212; check the switch.</p>
</body>
</html>
"#;

    pub const C5: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Five</title></head>
<body>
<h1>Chapter 5 Replication</h1>
<p>Print a calibration cube first.</p>
<h3>Belt tension</h3>
<p>Tight but not too tight.</p>
</body>
</html>
"#;

    pub const NAV: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Contents</title></head>
<body>
<nav epub:type="toc">
  <ol>
    <li><a href="text/c1.xhtml">Chapter 1 Setup</a></li>
    <li><a href="text/c5.xhtml#replication">Chapter 5 Replication</a></li>
  </ol>
</nav>
</body>
</html>
"#;

    pub const NCX: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head><meta name="dtb:uid" content="urn:uuid:made-up"/></head>
<docTitle><text>Replication Basics</text></docTitle>
<navMap>
  <navPoint id="n1" playOrder="1">
    <navLabel><text>Chapter 1 Setup</text></navLabel>
    <content src="text/c1.xhtml"/>
  </navPoint>
  <navPoint id="n2" playOrder="2">
    <navLabel><text>Chapter 5 Replication</text></navLabel>
    <content src="text/c5.xhtml"/>
  </navPoint>
</navMap>
</ncx>
"#;

    /// Build a zip archive from (name, bytes) pairs; `mimetype` is stored, the rest deflated.
    pub fn archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
        for (name, data) in entries {
            let options = if *name == "mimetype" {
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored)
            } else {
                SimpleFileOptions::default().compression_method(CompressionMethod::Deflated)
            };
            zip.start_file(*name, options).unwrap();
            zip.write_all(data).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    fn entry<'a>(name: &'a str, data: &'a str) -> (&'a str, &'a [u8]) {
        (name, data.as_bytes())
    }

    const ITEMS: &str = concat!(
        "    <item id=\"nav\" href=\"nav.xhtml\" media-type=\"application/xhtml+xml\" properties=\"nav\"/>\n",
        "    <item id=\"c1\" href=\"text/c1.xhtml\" media-type=\"application/xhtml+xml\"/>\n",
        "    <item id=\"c5\" href=\"text/c5.xhtml\" media-type=\"application/xhtml+xml\"/>\n",
    );

    /// EPUB 3 with a nav document; the spine lists c5 before c1.
    pub fn epub3_nav() -> Vec<u8> {
        let opf = opf3(
            ITEMS,
            "    <itemref idref=\"c5\"/>\n    <itemref idref=\"c1\"/>\n",
        );
        archive(&[
            ("mimetype", b"application/epub+zip"),
            entry("META-INF/container.xml", CONTAINER),
            entry("OEBPS/content.opf", &opf),
            entry("OEBPS/text/c1.xhtml", C1),
            entry("OEBPS/text/c5.xhtml", C5),
            entry("OEBPS/nav.xhtml", NAV),
        ])
    }

    /// EPUB 2: no nav property, the spine's `toc` names the NCX.
    pub fn epub2_ncx() -> Vec<u8> {
        let items = concat!(
            "    <item id=\"ncx\" href=\"toc.ncx\" media-type=\"application/x-dtbncx+xml\"/>\n",
            "    <item id=\"c1\" href=\"text/c1.xhtml\" media-type=\"application/xhtml+xml\"/>\n",
            "    <item id=\"c5\" href=\"text/c5.xhtml\" media-type=\"application/xhtml+xml\"/>\n",
        );
        let opf = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">urn:uuid:made-up</dc:identifier>
    <dc:title>Replication Basics</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
{items}  </manifest>
  <spine toc="ncx">
    <itemref idref="c5"/>
    <itemref idref="c1"/>
  </spine>
</package>
"#
        );
        archive(&[
            ("mimetype", b"application/epub+zip"),
            entry("META-INF/container.xml", CONTAINER),
            entry("OEBPS/content.opf", &opf),
            entry("OEBPS/text/c1.xhtml", C1),
            entry("OEBPS/text/c5.xhtml", C5),
            entry("OEBPS/toc.ncx", NCX),
        ])
    }

    /// No nav and no NCX: titles must come from the chapters' own headings. Chapter 3 has no
    /// h1, only an h2, to prove the fallback reaches h2.
    pub fn no_toc() -> Vec<u8> {
        let c3 = r#"<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Three</title></head>
<body>
<h2>Chapter 3 Bed Leveling</h2>
<p>Squash the first layer until it bites.</p>
</body>
</html>
"#;
        let items = concat!(
            "    <item id=\"c3\" href=\"text/c3.xhtml\" media-type=\"application/xhtml+xml\"/>\n",
            "    <item id=\"c5\" href=\"text/c5.xhtml\" media-type=\"application/xhtml+xml\"/>\n",
        );
        let opf = opf3(
            items,
            "    <itemref idref=\"c3\"/>\n    <itemref idref=\"c5\"/>\n",
        );
        archive(&[
            ("mimetype", b"application/epub+zip"),
            entry("META-INF/container.xml", CONTAINER),
            entry("OEBPS/content.opf", &opf),
            entry("OEBPS/text/c3.xhtml", c3),
            entry("OEBPS/text/c5.xhtml", C5),
        ])
    }

    /// encryption.xml lists an encrypted chapter: the book is DRM-protected.
    pub fn drm_book() -> Vec<u8> {
        let enc = r#"<?xml version="1.0" encoding="UTF-8"?>
<encryption xmlns="https://www.w3.org/2001/04/xmlenc#">
  <EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#">
    <EncryptionMethod Algorithm="http://www.w3.org/2001/04/xmlenc#aes256-cbc"/>
    <CipherData>
      <CipherReference URI="OEBPS/text/c1.xhtml"/>
    </CipherData>
  </EncryptedData>
</encryption>
"#;
        let opf = opf3(
            ITEMS,
            "    <itemref idref=\"c5\"/>\n    <itemref idref=\"c1\"/>\n",
        );
        archive(&[
            ("mimetype", b"application/epub+zip"),
            entry("META-INF/container.xml", CONTAINER),
            entry("META-INF/encryption.xml", enc),
            entry("OEBPS/content.opf", &opf),
            entry("OEBPS/text/c1.xhtml", C1),
            entry("OEBPS/text/c5.xhtml", C5),
            entry("OEBPS/nav.xhtml", NAV),
        ])
    }

    /// encryption.xml covers only the OPF: metadata encryption is not DRM-protected.
    pub fn metadata_encrypted_book() -> Vec<u8> {
        let enc = r#"<?xml version="1.0" encoding="UTF-8"?>
<encryption xmlns="https://www.w3.org/2001/04/xmlenc#">
  <EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#">
    <EncryptionMethod Algorithm="http://www.w3.org/2001/04/xmlenc#aes256-cbc"/>
    <CipherData>
      <CipherReference URI="OEBPS/content.opf"/>
    </CipherData>
  </EncryptedData>
</encryption>
"#;
        let opf = opf3(
            ITEMS,
            "    <itemref idref=\"c5\"/>\n    <itemref idref=\"c1\"/>\n",
        );
        archive(&[
            ("mimetype", b"application/epub+zip"),
            entry("META-INF/container.xml", CONTAINER),
            entry("META-INF/encryption.xml", enc),
            entry("OEBPS/content.opf", &opf),
            entry("OEBPS/text/c1.xhtml", C1),
            entry("OEBPS/text/c5.xhtml", C5),
            entry("OEBPS/nav.xhtml", NAV),
        ])
    }

    /// Chapter 1 is garbage (an unmatched end tag: not well-formed XML); chapter 5 is fine.
    pub fn malformed_chapter() -> Vec<u8> {
        let opf = opf3(
            ITEMS,
            "    <itemref idref=\"c5\"/>\n    <itemref idref=\"c1\"/>\n",
        );
        archive(&[
            ("mimetype", b"application/epub+zip"),
            entry("META-INF/container.xml", CONTAINER),
            entry("OEBPS/content.opf", &opf),
            ("OEBPS/text/c1.xhtml", b"</p></p> this is not markup"),
            entry("OEBPS/text/c5.xhtml", C5),
            entry("OEBPS/nav.xhtml", NAV),
        ])
    }

    /// Both chapters are garbage: nothing produces text.
    pub fn all_chapters_malformed() -> Vec<u8> {
        let opf = opf3(
            ITEMS,
            "    <itemref idref=\"c5\"/>\n    <itemref idref=\"c1\"/>\n",
        );
        archive(&[
            ("mimetype", b"application/epub+zip"),
            entry("META-INF/container.xml", CONTAINER),
            entry("OEBPS/content.opf", &opf),
            ("OEBPS/text/c1.xhtml", b"</p></p> not markup"),
            ("OEBPS/text/c5.xhtml", b"</p></p> not markup"),
            entry("OEBPS/nav.xhtml", NAV),
        ])
    }

    /// A well-formed spine but one chapter is a non-XHTML binary: that chapter is dropped
    /// without panicking, and it never counts toward the produced text.
    pub fn binary_chapter() -> Vec<u8> {
        let opf = opf3(
            ITEMS,
            "    <itemref idref=\"c1\"/>\n    <itemref idref=\"c5\"/>\n",
        );
        archive(&[
            ("mimetype", b"application/epub+zip"),
            entry("META-INF/container.xml", CONTAINER),
            entry("OEBPS/content.opf", &opf),
            entry("OEBPS/text/c1.xhtml", C1),
            ("OEBPS/text/c5.xhtml", &[0xff, 0xfe, 0x00, 0x01, 0x00][..]),
            entry("OEBPS/nav.xhtml", NAV),
        ])
    }

    /// An archive with `count` entries, well past the limit, and nothing else of an EPUB.
    pub fn too_many_entries(count: usize) -> Vec<u8> {
        let mut entries: Vec<(String, Vec<u8>)> = vec![
            ("mimetype".to_string(), b"application/epub+zip".to_vec()),
            (
                "META-INF/container.xml".to_string(),
                CONTAINER.as_bytes().to_vec(),
            ),
        ];
        for i in 0..count {
            entries.push((format!("f{i}.txt"), vec![b'x'; 1]));
        }
        let borrowed: Vec<(&str, &[u8])> = entries
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        archive(&borrowed)
    }

    /// Write one of these archives into a source tree under `rel` (creating folders).
    pub fn write_epub(root: &Path, rel: &str, bytes: &[u8]) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use crate::chunk::{chunk_document, reference};

    fn trails(doc: &Document) -> Vec<String> {
        doc.sections
            .iter()
            .map(|s| s.headings.join(" > "))
            .collect()
    }

    #[test]
    fn epub3_nav_titles_and_spine_order_beat_file_order() {
        let doc = epub("Library/Replication Basics.epub", &epub3_nav()).unwrap();
        assert_eq!(doc.title, "Replication Basics");
        // The spine lists c5 before c1; the archive and manifest list c1 first.
        assert_eq!(
            trails(&doc),
            [
                "Chapter 5 Replication",
                "Chapter 5 Replication > Belt tension",
                "Chapter 1 Setup",
                "Chapter 1 Setup > Power",
            ]
        );
        assert!(doc.sections[0]
            .body
            .contains("Print a calibration cube first."));
        // The nav href carried a fragment; it resolved to the chapter anyway.
        let chunks = chunk_document("Library/Replication Basics.epub", &doc, 1200);
        assert_eq!(
            reference(&chunks[0].path, &chunks[0].anchor),
            "Library/Replication Basics.epub#Chapter 5 Replication"
        );
        assert_eq!(chunks[2].heading, "Chapter 1 Setup");
        assert_eq!(chunks[2].title, "Replication Basics");
    }

    #[test]
    fn epub2_ncx_titles() {
        let doc = epub("book.epub", &epub2_ncx()).unwrap();
        assert_eq!(doc.title, "Replication Basics");
        assert_eq!(
            trails(&doc),
            [
                "Chapter 5 Replication",
                "Chapter 5 Replication > Belt tension",
                "Chapter 1 Setup",
                "Chapter 1 Setup > Power",
            ]
        );
    }

    #[test]
    fn no_table_of_contents_falls_back_to_the_first_h1_or_h2() {
        let doc = epub("book.epub", &no_toc()).unwrap();
        // Chapter 3 opens with an h2; chapter 5 opens with an h1.
        assert_eq!(
            trails(&doc),
            [
                "Chapter 3 Bed Leveling",
                "Chapter 5 Replication",
                "Chapter 5 Replication > Belt tension",
            ]
        );
        assert!(doc.sections[0]
            .body
            .contains("Squash the first layer until it bites."));
    }

    #[test]
    fn a_drm_flagged_book_is_skipped() {
        let err = epub("book.epub", &drm_book()).unwrap_err().to_string();
        assert!(err.contains("DRM-protected"), "{err}");
    }

    #[test]
    fn encrypting_only_the_opf_is_not_drm() {
        let doc = epub("book.epub", &metadata_encrypted_book()).unwrap();
        assert!(!doc.sections.is_empty());
    }

    #[test]
    fn an_archive_past_the_entry_limit_is_refused() {
        let err = epub("book.epub", &too_many_entries(10_001))
            .unwrap_err()
            .to_string();
        assert!(err.contains("over 10,000 zip entries"), "{err}");
    }

    #[test]
    fn a_malformed_chapter_is_dropped_and_the_book_still_indexes() {
        let doc = epub("book.epub", &malformed_chapter()).unwrap();
        assert_eq!(
            trails(&doc),
            [
                "Chapter 5 Replication",
                "Chapter 5 Replication > Belt tension",
            ]
        );
        assert!(!doc
            .sections
            .iter()
            .any(|s| s.body.contains("level the bed")));
    }

    #[test]
    fn a_book_with_no_readable_chapter_reports_the_drop_count() {
        let err = epub("book.epub", &all_chapters_malformed())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no chapters produced text"), "{err}");
        assert!(err.contains("2 dropped"), "{err}");
    }

    #[test]
    fn a_binary_chapter_is_dropped_without_panicking() {
        let doc = epub("book.epub", &binary_chapter()).unwrap();
        assert!(trails(&doc).contains(&"Chapter 1 Setup".to_string()));
        assert!(!trails(&doc).iter().any(|t| t.contains("Belt tension")));
    }

    #[test]
    fn entities_are_decoded_and_script_style_are_dropped_and_breaks_kept() {
        let doc = epub("book.epub", &epub3_nav()).unwrap();
        let setup = doc
            .sections
            .iter()
            .find(|s| s.headings == ["Chapter 1 Setup"])
            .unwrap();
        assert_eq!(
            setup.body,
            "Plug in the press & level the bed.\n\nSecond paragraph about the bed."
        );
        let power = doc
            .sections
            .iter()
            .find(|s| s.headings == ["Chapter 1 Setup", "Power"])
            .unwrap();
        assert_eq!(power.body, "230V only \u{2014} check the switch.");
        assert!(!doc.sections.iter().any(|s| s.body.contains("var secret")));
        assert!(!doc.sections.iter().any(|s| s.body.contains("color: red")));
    }

    #[test]
    fn not_a_zip_is_an_error() {
        let err = epub("book.epub", b"just bytes, not a zip archive").unwrap_err();
        assert!(
            format!("{err:#}").contains("not a readable zip archive"),
            "{err:#}"
        );
    }

    #[test]
    fn an_unanchored_href_resolves_past_the_opf_directory() {
        // hrefs may climb out of, or point sideways around, the OPF folder.
        assert_eq!(
            resolve("OEBPS/", "../OEBPS/text/c1.xhtml"),
            "OEBPS/text/c1.xhtml"
        );
        assert_eq!(resolve("OEBPS/", "/OEBPS/nav.xhtml"), "OEBPS/nav.xhtml");
        assert_eq!(
            resolve("OEBPS/", "text/a%20b.xhtml"),
            "OEBPS/text/a b.xhtml"
        );
        assert_eq!(resolve("OEBPS/", "c5.xhtml#replication"), "OEBPS/c5.xhtml");
    }
}
