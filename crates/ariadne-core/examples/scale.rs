//! Measure Ariadne at library scale: build time, database size, query latency, and vector memory.
//!
//! ```text
//! cargo run --release --example scale
//! cargo run --release --example scale -- --chunks 1000,10000,100000 --dims 1024 --queries 100
//! cargo run --release --example scale -- --chunks 100000 --dims 0   # keyword only
//! ```
//!
//! The corpus is generated, the vectors are synthetic, and the reranker is absent, so the numbers
//! measure the engine: ingestion, chunking, SQLite FTS5, vector storage, fusion, and the scoring
//! loop. They say nothing about model quality.
//!
//! Peak RSS comes from `/proc/self/status` on Linux and is per process, so it only grows across
//! the sizes in one run. Run one size per invocation when the memory number matters, or read the
//! largest size.

use ariadne_core::{Embedder, Index, Mode, Models, SearchOptions, Source};
use std::fs;
use std::path::Path;
use std::time::Instant;

const CHAPTERS_PER_BOOK: usize = 40;
const BODY_CHARS: usize = 900;
const DEFAULT_DIMS: usize = 1024;
const DEFAULT_QUERIES: usize = 50;
const DEFAULT_CHUNKS: [usize; 2] = [1_000, 10_000];

/// Real words, so the tokenizer and stemmer do their usual work. Rarer terms get a suffix.
const BASES: &[&str] = &[
    "lattice",
    "harbor",
    "signal",
    "copper",
    "meadow",
    "circuit",
    "anchor",
    "vessel",
    "timber",
    "granite",
    "lantern",
    "orchard",
    "compass",
    "quarry",
    "shelter",
    "furnace",
    "meridian",
    "cobalt",
    "thicket",
    "cistern",
    "alloy",
    "gantry",
    "plinth",
    "weir",
    "spindle",
    "trough",
    "bellows",
    "kiln",
    "flume",
    "brake",
    "shunt",
    "gauge",
    "valve",
    "flange",
    "rivet",
    "spline",
    "bracket",
    "bushing",
    "gearbox",
    "turbine",
    "condenser",
    "breaker",
    "relay",
    "thermostat",
    "manifold",
    "coupler",
    "sensor",
    "actuator",
];

/// Distinct terms in the generated corpus. Large enough that postings look like real prose.
const VOCAB: usize = 8_192;

fn main() -> anyhow::Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;
    let queries = make_queries(args.queries);

    println!(
        "ariadne scale: chunks={:?} dims={} queries={} os={} arch={} cpus={}",
        args.chunks,
        args.dims,
        queries.len(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |n| n.get()),
    );
    println!();
    println!(
        "| chunks | build bm25 | build dense | db bm25 | db dense | vectors | \
         cold dense | bm25 p50/p95 | dense p50/p95 | hybrid p50/p95 | peak rss |"
    );
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");

    for &chunks in &args.chunks {
        let row = measure(chunks, args.dims, &queries)?;
        row.print();
    }
    Ok(())
}

struct Args {
    chunks: Vec<usize>,
    dims: usize,
    queries: usize,
}

impl Args {
    fn parse(mut argv: impl Iterator<Item = String>) -> anyhow::Result<Self> {
        let mut args = Args {
            chunks: DEFAULT_CHUNKS.to_vec(),
            dims: DEFAULT_DIMS,
            queries: DEFAULT_QUERIES,
        };
        while let Some(flag) = argv.next() {
            let value = argv
                .next()
                .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))?;
            match flag.as_str() {
                "--chunks" => {
                    args.chunks = value
                        .split(',')
                        .map(|n| n.trim().parse::<usize>())
                        .collect::<std::result::Result<_, _>>()?;
                }
                "--dims" => args.dims = value.parse()?,
                "--queries" => args.queries = value.parse()?,
                other => anyhow::bail!("unknown flag {other}; use --chunks, --dims, --queries"),
            }
        }
        if args.chunks.is_empty() || args.chunks.contains(&0) {
            anyhow::bail!("--chunks needs at least one positive size");
        }
        if args.queries == 0 {
            anyhow::bail!("--queries must be at least 1");
        }
        Ok(args)
    }
}

struct Row {
    chunks: usize,
    build_bm25_ms: f64,
    build_dense_ms: Option<f64>,
    db_bm25_mb: f64,
    db_dense_mb: Option<f64>,
    vectors_mb: Option<f64>,
    cold_dense_ms: Option<f64>,
    bm25: (f64, f64),
    dense: Option<(f64, f64)>,
    hybrid: (f64, f64),
    peak_rss_mb: Option<f64>,
}

impl Row {
    fn print(&self) {
        println!(
            "| {} | {} | {} | {:.1} MB | {} | {} | {} | {:.2}/{:.2} ms | {} | {:.2}/{:.2} ms | {} |",
            self.chunks,
            seconds(self.build_bm25_ms),
            self.build_dense_ms.map_or("-".into(), seconds),
            self.db_bm25_mb,
            self.db_dense_mb.map_or("-".into(), |mb| format!("{mb:.1} MB")),
            self.vectors_mb.map_or("-".into(), |mb| format!("{mb:.1} MB")),
            self.cold_dense_ms.map_or("-".into(), millis),
            self.bm25.0,
            self.bm25.1,
            self.dense.map_or("-".into(), |(p50, p95)| format!(
                "{p50:.2}/{p95:.2} ms"
            )),
            self.hybrid.0,
            self.hybrid.1,
            self.peak_rss_mb.map_or("-".into(), |mb| format!("{mb:.0} MB")),
        );
    }
}

fn measure(chunks: usize, dims: usize, queries: &[String]) -> anyhow::Result<Row> {
    let dir = tempfile::tempdir()?;
    let corpus = dir.path().join("corpus");
    let db = dir.path().join("index.db");
    generate(&corpus, chunks)?;

    // Keyword-only build.
    let mut index = Index::open(&db, Source::new(&corpus), Models::default())?;
    let started = Instant::now();
    let stats = index.build()?;
    let build_bm25_ms = started.elapsed().as_secs_f64() * 1000.0;
    let chunks = stats.chunks_written;
    drop(index);
    let db_bm25_mb = db_mb(&db);

    let mut row = Row {
        chunks,
        build_bm25_ms,
        build_dense_ms: None,
        db_bm25_mb,
        db_dense_mb: None,
        vectors_mb: None,
        cold_dense_ms: None,
        bm25: (0.0, 0.0),
        dense: None,
        hybrid: (0.0, 0.0),
        peak_rss_mb: None,
    };
    if dims == 0 {
        let mut index = Index::open(&db, Source::new(&corpus), Models::default())?;
        row.bm25 = latencies(&mut index, Mode::Bm25, queries)?;
        row.hybrid = latencies(&mut index, Mode::Hybrid, queries)?;
        row.peak_rss_mb = peak_rss_mb();
        return Ok(row);
    }

    // Dense build: the embedder name differs, so this is a full rebuild with vectors.
    let mut index = Index::open(&db, Source::new(&corpus), models(dims))?;
    let started = Instant::now();
    index.build()?;
    row.build_dense_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
    drop(index);
    row.db_dense_mb = Some(db_mb(&db));

    // Vectors as the index holds them: f32 per dimension, plus a Vec header per chunk.
    row.vectors_mb = Some((chunks * (dims * 4 + 24)) as f64 / (1024.0 * 1024.0));

    // A fresh reader pays for loading every vector on its first dense query.
    let mut index = Index::open(&db, Source::new(&corpus), models(dims))?;
    let started = Instant::now();
    index.search(&queries[0], &SearchOptions::new(5, Mode::Dense, false))?;
    row.cold_dense_ms = Some(started.elapsed().as_secs_f64() * 1000.0);

    row.bm25 = latencies(&mut index, Mode::Bm25, queries)?;
    row.dense = Some(latencies(&mut index, Mode::Dense, queries)?);
    row.hybrid = latencies(&mut index, Mode::Hybrid, queries)?;
    row.peak_rss_mb = peak_rss_mb();
    Ok(row)
}

fn models(dims: usize) -> Models {
    Models {
        embedder: Some(Box::new(SyntheticEmbedder { dims })),
        ..Models::default()
    }
}

fn latencies(index: &mut Index, mode: Mode, queries: &[String]) -> anyhow::Result<(f64, f64)> {
    let opts = SearchOptions::new(5, mode, false);
    let mut times = Vec::with_capacity(queries.len());
    for query in queries {
        let started = Instant::now();
        let hits = index.search(query, &opts)?;
        std::hint::black_box(hits.len());
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    Ok((percentile(&times, 0.50), percentile(&times, 0.95)))
}

/// Nearest-rank percentile over sorted input: the value at or below which `p` of the samples fall.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn seconds(ms: f64) -> String {
    format!("{:.2} s", ms / 1000.0)
}

fn millis(ms: f64) -> String {
    format!("{ms:.1} ms")
}

/// Database size including the write-ahead log, so a build that has not been checkpointed still
/// counts.
fn db_mb(db: &Path) -> f64 {
    ["", "-wal", "-shm"]
        .iter()
        .filter_map(|suffix| fs::metadata(format!("{}{suffix}", db.display())).ok())
        .map(|m| m.len())
        .sum::<u64>() as f64
        / (1024.0 * 1024.0)
}

#[cfg(target_os = "linux")]
fn peak_rss_mb() -> Option<f64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let kb = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()?;
    Some(kb as f64 / 1024.0)
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_mb() -> Option<f64> {
    None
}

fn make_queries(count: usize) -> Vec<String> {
    let mut rng = Rng::new(0x5EED);
    (0..count)
        .map(|_| format!("{} {}", rng.term(), rng.term()))
        .collect()
}

/// Write `chunks` chapters across books, each chapter short enough to stay one chunk.
fn generate(root: &Path, chunks: usize) -> anyhow::Result<()> {
    fs::create_dir_all(root)?;
    let books = chunks.div_ceil(CHAPTERS_PER_BOOK);
    for book in 0..books {
        let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15 ^ (book as u64).wrapping_mul(0x100_0000_01B3));
        let mut text = format!("# Book {book:05}\n\n");
        for chapter in 0..CHAPTERS_PER_BOOK {
            text.push_str(&format!(
                "## Chapter {:06}\n\n{}\n\n",
                book * CHAPTERS_PER_BOOK + chapter,
                body(&mut rng)
            ));
        }
        fs::write(root.join(format!("book-{book:05}.md")), text)?;
    }
    Ok(())
}

/// About `BODY_CHARS` characters of prose, each empty string below the chunk limit.
fn body(rng: &mut Rng) -> String {
    let mut body = String::with_capacity(BODY_CHARS + 64);
    while body.len() < BODY_CHARS {
        let words = 6 + (rng.next() % 6) as usize;
        for i in 0..words {
            if i > 0 {
                body.push(' ');
            }
            body.push_str(&rng.term());
        }
        body.push_str(". ");
    }
    body
}

/// xorshift64: reproducible across runs and platforms.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Zipf-like: low ranks dominate, which is what real term frequencies look like.
    fn term(&mut self) -> String {
        let rank = (self.unit().powi(2) * VOCAB as f64) as usize;
        let base = BASES[rank % BASES.len()];
        if rank < BASES.len() {
            base.to_string()
        } else {
            format!("{base}-{rank}")
        }
    }
}

/// Deterministic bag-of-words vectors: enough to measure storage, loading, and scanning.
struct SyntheticEmbedder {
    dims: usize,
}

impl Embedder for SyntheticEmbedder {
    fn name(&self) -> String {
        format!("scale:hash:{}", self.dims)
    }

    fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| self.vector(text)).collect())
    }
}

impl SyntheticEmbedder {
    fn vector(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dims];
        for token in text.to_lowercase().split_whitespace() {
            v[(fnv(token.as_bytes()) % self.dims as u64) as usize] += 1.0;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}
