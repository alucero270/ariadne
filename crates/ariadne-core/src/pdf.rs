//! The PDF extractor (#155). The parsing itself is borrowed, not written: the bytes go to
//! poppler's `pdftotext`, which handles the real-world PDF soup, and this module owns only the
//! dispatch and the page-to-section mapping. A page becomes one section headed `Page N`, so
//! citations read `Library/Paper.pdf#Page 3`. Anything with no extractable text — a scan, a
//! cover page, an encrypted file — is an error naming why, which `build` records as a skip
//! (#150).

use crate::chunk::{Document, Section};
use anyhow::{bail, Context, Result};
use std::io::Write;
use std::path::Path;

/// Extract a PDF into a [`Document`], one section per page.
pub fn pdf(path: &str, bytes: &[u8]) -> Result<Document> {
    let text = pdftotext(bytes)?;
    let title = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document")
        .to_string();
    document_from_text(title, &text)
}

/// The [`Document`] behind `pdftotext` output, so the page mapping is testable without the
/// binary. Empty output — a scan, a cover page, an image-only file — is an error naming why.
fn document_from_text(title: String, text: &str) -> Result<Document> {
    let sections = pages(text);
    if sections.is_empty() {
        bail!("no extractable text: the PDF is scanned, empty, or image-only");
    }
    Ok(Document { title, sections })
}

/// Run poppler's `pdftotext` over the bytes, UTF-8 out. Reflowed, not `-layout`: reflowed
/// prose chunks better than columns faithfully laid out.
fn pdftotext(bytes: &[u8]) -> Result<String> {
    use std::process::{Command, Stdio};

    let mut child = Command::new("pdftotext")
        .args(["-enc", "UTF-8", "-", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("pdftotext not found: install poppler-utils to index PDFs")?;
    // A malformed PDF can make pdftotext exit before reading all of stdin; the exit status
    // carries the real error, so a broken pipe here is not the failure.
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(bytes);
    }
    let output = child
        .wait_with_output()
        .context("pdftotext did not finish")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let reason = stderr.lines().last().unwrap_or("").trim();
        bail!("pdftotext failed: {reason}");
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Split `pdftotext` output on the form feed it emits between pages. A page with no text is
/// dropped, but keeps its number, so a citation always matches the printed page.
fn pages(text: &str) -> Vec<Section> {
    text.split('\x0C')
        .enumerate()
        .filter(|(_, page)| !page.trim().is_empty())
        .map(|(i, page)| Section {
            headings: vec![format!("Page {}", i + 1)],
            body: page.trim().to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_split_on_form_feed_and_number_from_one() {
        let sections = pages("first page\x0Csecond\x0Cthird page");
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].headings, vec!["Page 1".to_string()]);
        assert_eq!(sections[0].body, "first page");
        assert_eq!(sections[1].headings, vec!["Page 2".to_string()]);
        assert_eq!(sections[1].body, "second");
        assert_eq!(sections[2].headings, vec!["Page 3".to_string()]);
        assert_eq!(sections[2].body, "third page");
    }

    #[test]
    fn an_empty_page_is_dropped_but_keeps_its_number() {
        // Page 1 and 2 have nothing; the text is on printed page 3.
        let sections = pages("\x0c\x0cthe only text");
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].headings, vec!["Page 3".to_string()]);
        assert_eq!(sections[0].body, "the only text");
    }

    #[test]
    fn a_trailing_form_feed_yields_no_empty_page() {
        assert_eq!(pages("text\x0c").len(), 1);
        assert_eq!(pages("text").len(), 1);
    }

    #[test]
    fn no_text_at_all_is_an_error_naming_why() {
        let err = document_from_text("Scan".to_string(), "")
            .unwrap_err()
            .to_string();
        assert!(err.contains("no extractable text"), "{err}");
    }

    /// Whether the `pdftotext` binary can be run at all. Poppler prints its version and exits
    /// non-zero with no arguments; a successful spawn is the existence check.
    fn pdftotext_available() -> bool {
        std::process::Command::new("pdftotext").output().is_ok()
    }

    /// A valid single-page PDF with one text object, xref offsets computed. Enough for
    /// pdftotext to extract the text; not enough to be interesting.
    fn minimal_pdf(text: &str) -> Vec<u8> {
        let stream = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources \
             << /Font << /F1 5 0 R >> >> >>"
                .to_string(),
            format!(
                "<< /Length {} >>\nstream\n{}\nendstream",
                stream.len(),
                stream
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend(format!("{} 0 obj\n{}\nendobj\n", i + 1, body).as_bytes());
        }
        let xref = out.len();
        out.extend(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        out.extend(b"0000000000 65535 f \n");
        for offset in &offsets {
            out.extend(format!("{offset:010} 00000 n \n").as_bytes());
        }
        out.extend(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
                objects.len() + 1,
                xref
            )
            .as_bytes(),
        );
        out
    }

    #[test]
    fn a_real_pdf_becomes_paged_sections() {
        if !pdftotext_available() {
            eprintln!("skipping: pdftotext (poppler-utils) is not installed");
            return;
        }
        let document = pdf("Library/Calibration.pdf", &minimal_pdf("Hello Ariadne")).unwrap();
        assert_eq!(document.title, "Calibration");
        assert_eq!(document.sections.len(), 1);
        assert_eq!(document.sections[0].headings, vec!["Page 1".to_string()]);
        assert!(
            document.sections[0].body.contains("Hello Ariadne"),
            "{:?}",
            document.sections[0].body
        );
    }
}
