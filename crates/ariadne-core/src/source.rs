//! Walking one source root. Symlinks are never followed, so nothing outside the root is read.

use anyhow::{bail, Context, Result};
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const DEFAULT_EXCLUDES: &[&str] = &[
    ".git/*",
    ".obsidian/*",
    ".opencode/*",
    "*node_modules/*",
    "*-artifacts/*",
    "*rogue-scratch*",
];

pub const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub rel: String,
    pub abs: PathBuf,
}

/// fnmatch-style match where `*` matches any run of characters, including `/`.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0, 0);
    let (mut star, mut mark) = (None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn excluded(rel: &str, excludes: &[String]) -> bool {
    excludes.iter().any(|pat| glob_match(pat, rel))
}

pub fn walk_markdown(root: &Path, excludes: &[String]) -> Result<Vec<SourceFile>> {
    let root = root
        .canonicalize()
        .with_context(|| format!("source root {} is not readable", root.display()))?;
    let mut out = Vec::new();
    let mut stack = vec![(root.clone(), String::new())];
    while let Some((dir, prefix)) = stack.pop() {
        let mut entries: Vec<_> = fs::read_dir(&dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let name = entry.file_name().to_string_lossy().into_owned();
            let rel = format!("{prefix}{name}");
            let meta = entry.metadata()?;
            let kind = entry.file_type()?;
            if kind.is_symlink() {
                continue;
            }
            if kind.is_dir() {
                if !excluded(&format!("{rel}/"), excludes) {
                    stack.push((entry.path(), format!("{rel}/")));
                }
            } else if kind.is_file()
                && name.ends_with(".md")
                && meta.len() <= MAX_FILE_BYTES
                && !excluded(&rel, excludes)
            {
                out.push(SourceFile {
                    rel,
                    abs: entry.path(),
                });
            }
        }
    }
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(out)
}

/// Resolve a repo-relative path to a regular, non-symlinked file inside `root`.
pub fn resolve_inside(root: &Path, rel: &str) -> Result<PathBuf> {
    let rel_path = Path::new(rel.trim_start_matches('/'));
    if rel_path
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        bail!("path must be relative and stay inside the source root");
    }
    let root = root.canonicalize()?;
    let mut current = root.clone();
    for part in rel_path.components() {
        current.push(part);
        let meta =
            fs::symlink_metadata(&current).with_context(|| format!("no such file: {rel}"))?;
        if meta.file_type().is_symlink() {
            bail!("symlinks are not readable");
        }
    }
    if !current.is_file() || !current.starts_with(&root) {
        bail!("no such file: {rel}");
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Vec<String> {
        DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn glob_matches_like_fnmatch() {
        assert!(glob_match(".git/*", ".git/"));
        assert!(glob_match("*node_modules/*", "a/b/node_modules/x"));
        assert!(glob_match("*-artifacts/*", "systems/x/foo-artifacts/a.md"));
        assert!(glob_match("*rogue-scratch*", "a/rogue-scratch-2026/b"));
        assert!(!glob_match(".git/*", "docs/.git-notes.md"));
        assert!(glob_match("a?c", "abc"));
    }

    #[test]
    fn walk_skips_excludes_symlinks_and_non_markdown() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("repo");
        fs::create_dir_all(root.join("docs/sub")).unwrap();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join("x-artifacts")).unwrap();
        fs::write(root.join("README.md"), "# r").unwrap();
        fs::write(root.join("docs/sub/a.md"), "# a").unwrap();
        fs::write(root.join("docs/b.txt"), "b").unwrap();
        fs::write(root.join(".git/c.md"), "# c").unwrap();
        fs::write(root.join("x-artifacts/d.md"), "# d").unwrap();
        fs::write(dir.path().join("outside.md"), "# secret").unwrap();
        std::os::unix::fs::symlink(dir.path().join("outside.md"), root.join("link.md")).unwrap();
        std::os::unix::fs::symlink(dir.path(), root.join("linkdir")).unwrap();

        let rels: Vec<_> = walk_markdown(&root, &defaults())
            .unwrap()
            .into_iter()
            .map(|f| f.rel)
            .collect();
        assert_eq!(rels, ["README.md", "docs/sub/a.md"]);
    }

    #[test]
    fn resolve_inside_refuses_escapes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("repo");
        fs::create_dir_all(root.join("docs")).unwrap();
        fs::write(root.join("docs/a.md"), "a").unwrap();
        fs::write(dir.path().join("outside.md"), "x").unwrap();
        std::os::unix::fs::symlink(dir.path().join("outside.md"), root.join("link.md")).unwrap();
        std::os::unix::fs::symlink(dir.path(), root.join("linkdir")).unwrap();

        assert!(resolve_inside(&root, "docs/a.md").is_ok());
        assert!(resolve_inside(&root, "/docs/a.md").is_ok());
        for bad in [
            "../outside.md",
            "docs/../../outside.md",
            "link.md",
            "linkdir/outside.md",
            "docs",
            "missing.md",
        ] {
            assert!(
                resolve_inside(&root, bad).is_err(),
                "{bad} should be refused"
            );
        }
    }
}
