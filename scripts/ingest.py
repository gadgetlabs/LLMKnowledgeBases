#!/usr/bin/env python3
"""
ingest.py — Ingestion pipeline for the llm-wiki.

Modes:
  seed  <json_file>   Seed wiki from a structured JSON reading list
  pdf   <path>        Ingest a single PDF into the wiki
  watch               Watch raw/ for new PDFs (runs until interrupted)
  scan  [dir]         Bulk-ingest all PDFs under a directory
  retry               Re-process PDFs that previously failed
  enrich <json>       Cross-link source articles to cited document articles
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import config_loader as cfg

# ---------------------------------------------------------------------------
# Config — values come from config.yaml with env-var / hardcoded fallbacks
# ---------------------------------------------------------------------------
WIKI_PATH = Path(os.environ.get("WIKI_PATH", "~/policing-wiki")).expanduser()

WIKI_DIR = Path(cfg.get("paths.wiki_dir", str(WIKI_PATH / "wiki"))).expanduser()
OUTPUTS_DIR = Path(cfg.get("paths.outputs_dir", str(WIKI_PATH / "outputs"))).expanduser()
RAW_DIR = Path(cfg.get("paths.raw_dir", str(WIKI_PATH / "raw"))).expanduser()
UNPROCESSED_DIR = WIKI_PATH / "unprocessed"
INDEX_PATH = WIKI_PATH / "index.md"
SEARCH_INDEX_DIR = WIKI_PATH / "scripts" / ".search_index"

MODEL = cfg.get("model.name", "claude-sonnet-4-20250514")
DOMAIN = cfg.get("wiki.topic", "your knowledge domain")

MARKER_TIMEOUT = cfg.get("extraction.marker_timeout", 120)
NOUGAT_TIMEOUT = cfg.get("extraction.nougat_timeout", 180)
DEFAULT_WORKERS = cfg.get("extraction.workers", 3)

BASE_READINGS_PATH = Path(
    cfg.get("paths.readings_dir")
    or os.environ.get(
        "READINGS_PATH",
        "/Users/ferret/Projects/GadgetLabs/download-reading-list/Block_A_Readings",
    )
).expanduser()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SEED_SYSTEM_PROMPT = f"""\
You are a wiki article generator for a knowledge base on {DOMAIN}.

Given a source entry (lecture, chapter, or reading), write a concise markdown wiki article. Include:
- A summary of the topic (2-3 paragraphs)
- Key concepts as bullet points
- A "Sources" section listing required and suggested references with URLs where available
- Obsidian-style [[backlinks]] to related concepts whenever you mention them in the text. Be generous with backlinks — any concept that could be its own article should be linked.
- A "Related Topics" section at the bottom listing [[backlinked]] topics

Use the provided description and reading list to inform the article. Do NOT pad or invent content beyond what the source material supports.

Output ONLY the markdown content, no preamble.\
"""

PDF_SYSTEM_PROMPT_TEMPLATE = """\
You are a wiki article generator for a knowledge base on {domain}.

Given the full text of a source document, write a concise wiki article. Include:
- Document metadata (authors, year, publication) at the top as a YAML frontmatter block
- A summary (3-4 paragraphs covering the main argument, methodology, key findings, and implications)
- Key concepts and definitions introduced
- Obsidian-style [[backlinks]] to related concepts. Here are the existing topics in the wiki — link to them where relevant: {{existing_topics}}
- A "Related Topics" section at the bottom

Be accurate to the source material. Do not invent findings or overstate conclusions.

Output ONLY the markdown content, no preamble.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Convert text to kebab-case slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text.strip())
    text = re.sub(r"-+", "-", text)
    return text[:80]


def extract_title(content: str) -> str:
    """Extract first # heading from markdown, or filename fallback."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def get_existing_topics() -> list[str]:
    """Return list of article titles from wiki directory."""
    topics = []
    for md_file in WIKI_DIR.glob("*.md"):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        title = extract_title(content)
        topics.append(title)
    return topics


def call_claude(client: anthropic.Anthropic, system: str, user_content: str) -> str | None:
    """Call Claude API with error handling. Returns text or None on failure."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text
    except anthropic.APIError as e:
        log.error("Claude API error: %s", e)
        return None


def rebuild_search_index() -> None:
    """Rebuild the tantivy full-text search index from wiki/*.md files."""
    try:
        import tantivy
    except ImportError:
        log.warning("tantivy not installed — skipping search index rebuild")
        return

    try:
        SEARCH_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("filename", stored=True)
        schema_builder.add_text_field("title", stored=True)
        schema_builder.add_text_field("body", stored=True)
        schema = schema_builder.build()

        # Always rebuild from scratch for correctness
        index = tantivy.Index(schema, path=str(SEARCH_INDEX_DIR))
        writer = index.writer()

        count = 0
        for md_file in WIKI_DIR.glob("*.md"):
            content = md_file.read_text(encoding="utf-8", errors="replace")
            title = extract_title(content)
            doc = tantivy.Document()
            doc.add_text("filename", md_file.name)
            doc.add_text("title", title)
            doc.add_text("body", content)
            writer.add_document(doc)
            count += 1

        writer.commit()
        log.info("Search index rebuilt with %d articles", count)
    except Exception as e:
        log.error("Failed to rebuild search index: %s", e)


def rebuild_index() -> None:
    """Regenerate index.md from all wiki/*.md files."""
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    articles: list[tuple[str, Path]] = []

    for md_file in sorted(WIKI_DIR.glob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        title = extract_title(content)
        articles.append((title, md_file))

    lines = ["# Wiki Index\n", f"*{len(articles)} articles*\n\n"]

    for title, path in sorted(articles, key=lambda x: x[0].lower()):
        lines.append(f"- [[{path.stem}]] — {title}\n")

    INDEX_PATH.write_text("".join(lines), encoding="utf-8")
    log.info("index.md rebuilt (%d articles)", len(articles))

    rebuild_search_index()


# ---------------------------------------------------------------------------
# Mode 1: Seed from JSON
# ---------------------------------------------------------------------------

def seed_from_json(json_path: Path) -> None:
    """Generate wiki articles for each lecture in the reading list JSON."""
    if not json_path.exists():
        log.error("JSON file not found: %s", json_path)
        sys.exit(1)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    lectures = data.get("lectures", [])
    log.info("Seeding wiki from %d lectures…", len(lectures))

    client = anthropic.Anthropic()
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    success = 0

    for lecture in lectures:
        number = lecture.get("number", 0)
        title = lecture.get("title", "Untitled")
        slug = slugify(title)
        filename = WIKI_DIR / f"{number:02d}-{slug}.md"

        if filename.exists():
            log.info("Skipping (exists): %s", filename.name)
            success += 1
            continue

        log.info("Generating article for lecture %02d: %s", number, title)
        user_content = json.dumps(lecture, indent=2)
        article = call_claude(client, SEED_SYSTEM_PROMPT, user_content)

        if article:
            filename.write_text(article, encoding="utf-8")
            log.info("Saved: %s", filename.name)
            success += 1
        else:
            log.error("Failed to generate article for lecture %02d", number)

    log.info("Seed complete: %d/%d articles generated", success, len(lectures))
    rebuild_index()


# ---------------------------------------------------------------------------
# Mode 2: Ingest a single PDF
# ---------------------------------------------------------------------------

def pdf_to_text_marker(pdf_path: Path) -> str:
    """Convert PDF to markdown using marker-pdf CLI (high quality, slow)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            result = subprocess.run(
                ["marker_single", str(pdf_path), "--output_dir", str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=MARKER_TIMEOUT,
            )
        except FileNotFoundError:
            raise RuntimeError("marker_single not found")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"marker_single timed out after {MARKER_TIMEOUT}s")

        stem = pdf_path.stem
        md_file = tmp_path / stem / f"{stem}.md"
        if md_file.exists():
            return md_file.read_text(encoding="utf-8")
        for f in tmp_path.rglob("*.md"):
            return f.read_text(encoding="utf-8")

        stderr_excerpt = result.stderr[:400] if result.stderr else "(no stderr)"
        raise RuntimeError(f"No markdown output. stderr: {stderr_excerpt}")


def pdf_to_text_nougat(pdf_path: Path) -> str:
    """Convert PDF using Meta's Nougat (academic-paper-optimised, handles equations/tables)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            result = subprocess.run(
                ["nougat", str(pdf_path), "-o", str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=NOUGAT_TIMEOUT,
            )
        except FileNotFoundError:
            raise RuntimeError("nougat not installed — run: pip install nougat-ocr")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"nougat timed out after {NOUGAT_TIMEOUT}s")

        # nougat outputs <stem>.mmd — search recursively in case of subdirs
        for f in tmp_path.rglob("*.mmd"):
            content = f.read_text(encoding="utf-8")
            if content.strip():
                return content
            raise RuntimeError("nougat produced an empty .mmd file")

        stderr_excerpt = result.stderr[:200] if result.stderr else "(no stderr)"
        raise RuntimeError(f"No nougat output found. stderr: {stderr_excerpt}")


def pdf_to_text_pypdf(pdf_path: Path) -> str:
    """Fallback: extract plain text using pypdf (fast, no layout)."""
    try:
        import pypdf
    except ImportError:
        raise RuntimeError("pypdf not installed — run: pip install pypdf")

    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

    full_text = "\n\n".join(text_parts)
    if not full_text.strip():
        raise RuntimeError("pypdf extracted no text (possibly a scanned PDF)")
    return full_text


def _is_html_file(pdf_path: Path) -> bool:
    """Return True if the file is actually HTML despite a .pdf extension."""
    try:
        header = pdf_path.read_bytes()[:512]
        return header.lstrip()[:5].upper() in (b"<!DOC", b"<HTML", b"<?XML")
    except OSError:
        return False


def pdf_to_text_html(pdf_path: Path) -> str:
    """Extract text from an HTML file using the stdlib html.parser."""
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        SKIP_TAGS = {"script", "style", "head"}

        def __init__(self):
            super().__init__()
            self.parts: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, _attrs):
            if tag in self.SKIP_TAGS:
                self._skip += 1

        def handle_endtag(self, tag):
            if tag in self.SKIP_TAGS and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip and data.strip():
                self.parts.append(data.strip())

    raw = pdf_path.read_text(encoding="utf-8", errors="replace")
    extractor = _TextExtractor()
    extractor.feed(raw)
    text = "\n".join(extractor.parts)
    if not text.strip():
        raise RuntimeError("HTML extractor found no text")
    return text


def pdf_to_text(pdf_path: Path, fast: bool = False) -> str:
    """
    Convert a PDF (or HTML-disguised-as-PDF) to text.

    Extraction mode is driven by config.yaml (extraction.mode):
      pypdf   — fast, CPU-only, plain text (also used when --fast flag is set)
      local   — marker-pdf → nougat → pypdf chain (default for high quality)
      modal   — marker-pdf on Modal GPU cloud, falls back to pypdf

    HTML files disguised as PDFs are always handled first regardless of mode.
    """
    # Pre-check: HTML files masquerading as PDFs
    if _is_html_file(pdf_path):
        log.info("Detected HTML file (not a true PDF): %s", pdf_path.name)
        try:
            text = pdf_to_text_html(pdf_path)
            log.info("Extracted with HTML parser: %s", pdf_path.name)
            return text
        except RuntimeError as e:
            raise RuntimeError(f"HTML extraction failed: {e}")

    mode = cfg.get("extraction.mode", "local")

    # --fast flag or pypdf mode: skip straight to pypdf
    if fast or mode == "pypdf":
        try:
            text = pdf_to_text_pypdf(pdf_path)
            log.info("Converted with pypdf: %s", pdf_path.name)
            return text
        except RuntimeError as e:
            raise RuntimeError(f"pypdf failed: {e}")

    # Modal GPU cloud extraction
    if mode == "modal":
        try:
            from modal_extractor import run_modal_extraction
            text = run_modal_extraction(pdf_path)
            log.info("Converted with Modal (GPU): %s", pdf_path.name)
            return text
        except Exception as e:
            log.warning("Modal extraction failed (%s) — falling back to pypdf", e)
        try:
            return pdf_to_text_pypdf(pdf_path)
        except RuntimeError as e:
            raise RuntimeError(f"Modal and pypdf both failed. pypdf: {e}")

    # mode == "local": marker-pdf → nougat → pypdf
    errors: dict[str, str] = {}

    for name, fn in [
        ("marker-pdf", pdf_to_text_marker),
        ("nougat", pdf_to_text_nougat),
    ]:
        try:
            text = fn(pdf_path)
            log.info("Converted with %s: %s", name, pdf_path.name)
            return text
        except RuntimeError as e:
            errors[name] = str(e)
            log.warning("%s failed (%s), trying next extractor…", name, e)

    try:
        text = pdf_to_text_pypdf(pdf_path)
        log.info("Converted with pypdf fallback: %s", pdf_path.name)
        return text
    except RuntimeError as e:
        errors["pypdf"] = str(e)

    raise RuntimeError(
        "All extractors failed — "
        + " | ".join(f"{k}: {v}" for k, v in errors.items())
    )


def _move_to_unprocessed(pdf_path: Path) -> None:
    """Copy a failed PDF to unprocessed/ so it can be retried later."""
    UNPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = UNPROCESSED_DIR / pdf_path.name
    if not dest.exists():
        import shutil
        shutil.copy2(pdf_path, dest)
        log.info("Copied to unprocessed/: %s", pdf_path.name)
    else:
        log.info("Already in unprocessed/: %s", pdf_path.name)


def ingest_pdf(pdf_path: Path, rebuild: bool = True, fast: bool = False) -> bool:
    """
    Convert a PDF to a wiki article. Returns True on success.
    Looks for the PDF in BASE_READINGS_PATH if the path doesn't exist directly.
    Set rebuild=False when running in parallel to defer index rebuild to the caller.
    Set fast=True to skip marker/nougat and go straight to pypdf.
    """
    if not pdf_path.exists():
        # Try resolving relative to BASE_READINGS_PATH
        candidate = BASE_READINGS_PATH / pdf_path
        if candidate.exists():
            pdf_path = candidate
        else:
            log.error("PDF not found: %s", pdf_path)
            return False

    log.info("Converting PDF: %s", pdf_path.name)

    try:
        text = pdf_to_text(pdf_path, fast=fast)
    except RuntimeError as e:
        log.error("Extraction failed on %s: %s", pdf_path.name, e)
        _move_to_unprocessed(pdf_path)
        return False

    existing_topics = get_existing_topics()
    topics_str = ", ".join(existing_topics) if existing_topics else "(none yet)"
    system_prompt = PDF_SYSTEM_PROMPT_TEMPLATE.format(domain=DOMAIN, existing_topics=topics_str)

    # Truncate very long texts to avoid token limits
    max_chars = cfg.get("extraction.max_text_chars", 60_000)
    if len(text) > max_chars:
        log.warning("PDF text truncated to %d chars (was %d)", max_chars, len(text))
        text = text[:max_chars]

    client = anthropic.Anthropic()
    log.info("Generating wiki article for: %s", pdf_path.name)
    article = call_claude(client, system_prompt, f"Paper text:\n\n{text}")

    if not article:
        log.error("Claude failed to generate article for: %s", pdf_path.name)
        return False

    # Derive filename from pdf stem (kebab-cased)
    slug = slugify(pdf_path.stem)
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WIKI_DIR / f"{slug}.md"

    # Avoid clobbering — add suffix if needed
    if out_path.exists():
        out_path = WIKI_DIR / f"{slug}-{int(time.time())}.md"

    out_path.write_text(article, encoding="utf-8")
    log.info("Saved: %s", out_path.name)
    if rebuild:
        rebuild_index()
    return True


# ---------------------------------------------------------------------------
# Mode 3: Watch raw/ directory
# ---------------------------------------------------------------------------

def get_wiki_stems() -> set[str]:
    """Return set of wiki article stems (without extension)."""
    return {f.stem for f in WIKI_DIR.glob("*.md")} if WIKI_DIR.exists() else set()


def watch_raw_dir() -> None:
    """Watch raw/ for new PDFs using watchdog."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        log.error("watchdog not installed — run: pip install watchdog")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    # Process any existing PDFs that don't have wiki articles yet
    existing_stems = get_wiki_stems()
    pending = [
        f for f in RAW_DIR.glob("*.pdf")
        if slugify(f.stem) not in existing_stems
    ]
    if pending:
        log.info("Processing %d pre-existing PDFs…", len(pending))
        for pdf in pending:
            ingest_pdf(pdf)

    class PDFHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() == ".pdf":
                log.info("Detected new PDF: %s", path.name)
                # Brief pause so the file is fully written
                time.sleep(1)
                ingest_pdf(path)

    observer = Observer()
    observer.schedule(PDFHandler(), str(RAW_DIR), recursive=False)
    observer.start()
    log.info("Watching %s for new PDFs… (Ctrl+C to stop)", RAW_DIR)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping watcher…")
        observer.stop()
    observer.join()


# ---------------------------------------------------------------------------
# Mode 4: Scan a directory for PDFs
# ---------------------------------------------------------------------------

def _run_parallel(pdfs: list[Path], workers: int, fast: bool = False) -> int:
    """
    Ingest a list of PDFs in parallel. Returns count of successes.
    Index is rebuilt once at the end rather than after each file.
    """
    total = len(pdfs)
    success = 0

    log.info("Starting parallel ingest: %d PDFs, %d workers%s",
             total, workers, " [fast mode]" if fast else "")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ingest_pdf, pdf, False, fast): pdf for pdf in pdfs}
        done = 0
        for future in as_completed(futures):
            pdf = futures[future]
            done += 1
            try:
                ok = future.result()
            except Exception as e:
                log.error("Unexpected error on %s: %s", pdf.name, e)
                ok = False
            if ok:
                success += 1
            log.info("[%d/%d done] %s — %s", done, total, pdf.name, "ok" if ok else "FAILED")

    # Single index rebuild after all workers finish
    rebuild_index()
    return success


def scan_directory(scan_path: Path, workers: int = DEFAULT_WORKERS, fast: bool = False) -> None:
    """Recursively find all PDFs under scan_path and ingest any without a wiki article."""
    if not scan_path.exists():
        log.error("Scan path not found: %s", scan_path)
        sys.exit(1)

    all_pdfs = sorted(scan_path.rglob("*.pdf"))
    if not all_pdfs:
        log.info("No PDFs found under %s", scan_path)
        return

    log.info("Found %d PDFs under %s", len(all_pdfs), scan_path)

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    existing_slugs = {f.stem for f in WIKI_DIR.glob("*.md")}
    pending = [p for p in all_pdfs if slugify(p.stem) not in existing_slugs]
    skipped = len(all_pdfs) - len(pending)

    if skipped:
        log.info("Skipping %d already-ingested PDFs", skipped)
    if not pending:
        log.info("All PDFs already ingested — nothing to do")
        return

    success = _run_parallel(pending, workers, fast=fast)
    log.info("Scan complete: %d/%d PDFs ingested successfully", success, len(pending))


# ---------------------------------------------------------------------------
# Mode 5: Retry unprocessed PDFs
# ---------------------------------------------------------------------------

def retry_unprocessed(workers: int = DEFAULT_WORKERS, fast: bool = False) -> None:
    """Attempt to re-ingest all PDFs sitting in the unprocessed/ folder."""
    if not UNPROCESSED_DIR.exists() or not any(UNPROCESSED_DIR.glob("*.pdf")):
        log.info("unprocessed/ is empty — nothing to retry")
        return

    pdfs = sorted(UNPROCESSED_DIR.glob("*.pdf"))
    log.info("Retrying %d unprocessed PDFs…", len(pdfs))

    success = _run_parallel(pdfs, workers, fast=fast)

    # Remove successfully ingested files from unprocessed/
    for pdf in pdfs:
        slug = slugify(pdf.stem)
        if any(WIKI_DIR.glob(f"{slug}*.md")):
            pdf.unlink()
            log.info("Removed from unprocessed/: %s", pdf.name)

    log.info("Retry complete: %d/%d succeeded", success, len(pdfs))


# ---------------------------------------------------------------------------
# Mode 6: Enrich lecture articles with links to paper articles
# ---------------------------------------------------------------------------

def _citation_to_match_keys(citation: str) -> tuple[str, str]:
    """
    Extract (author_surname, year) from a citation string for fuzzy matching.
    E.g. "Sherman, L. W. (2016). The Cambridge..." → ("sherman", "2016")
    """
    import re as _re
    year_match = _re.search(r"\b(19|20)\d{2}\b", citation)
    year = year_match.group(0) if year_match else ""
    # First word before a comma or space is the surname
    surname_match = _re.match(r"([A-Za-z''\-]+)", citation.strip())
    surname = surname_match.group(1).lower() if surname_match else ""
    return surname, year


def _find_matching_article(surname: str, year: str, wiki_stems: list[str]) -> str | None:
    """Return the wiki article stem that best matches (surname, year), or None."""
    if not surname or not year:
        return None
    candidates = [
        s for s in wiki_stems
        if surname in s.lower() and year in s
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer the one that starts with 'required-' or 'suggested-'
        for prefix in ("required-", "suggested-"):
            ranked = [c for c in candidates if c.startswith(prefix)]
            if ranked:
                return ranked[0]
        return candidates[0]
    return None


ENRICH_SYSTEM_PROMPT = """\
You are editing a wiki article. You will be given:
1. The current article text
2. A list of related source wiki articles that are cited in this article's reading list

Your task: update the article so that every mention of a cited source includes an \
Obsidian [[wikilink]] to its article. Also ensure the "Sources" section lists each \
item with its [[wikilink]]. Add a "## Related Sources" section at the bottom if one \
doesn't exist, listing all matched articles as [[wikilinks]].

Rules:
- Do NOT change any factual content, only add [[wikilinks]]
- Only link to sources that appear in the provided matched list — don't invent links
- If a source is already linked, leave it as-is
- Output ONLY the updated markdown, no preamble\
"""


def enrich_lecture_articles(json_path: Path) -> None:
    """
    Cross-link lecture wiki articles to their cited paper wiki articles.
    Uses reading_list.json to know which papers each lecture cites, then
    fuzzy-matches those citations to existing wiki article filenames.
    """
    if not json_path.exists():
        log.error("JSON file not found: %s", json_path)
        sys.exit(1)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    lectures = data.get("lectures", [])

    wiki_stems = [f.stem for f in WIKI_DIR.glob("*.md")]
    if not wiki_stems:
        log.error("Wiki is empty — run seed and scan first")
        sys.exit(1)

    client = anthropic.Anthropic()
    enriched = 0

    for lecture in lectures:
        number = lecture.get("number", 0)
        title = lecture.get("title", "")
        slug = slugify(title)
        lecture_file = WIKI_DIR / f"{number:02d}-{slug}.md"

        if not lecture_file.exists():
            log.warning("Lecture article not found: %s", lecture_file.name)
            continue

        # Collect all citations for this lecture
        all_readings = lecture.get("required", []) + lecture.get("suggested", [])
        matched: list[tuple[str, str]] = []  # (citation_text, article_stem)

        for reading in all_readings:
            citation = reading.get("citation", "")
            surname, year = _citation_to_match_keys(citation)
            article_stem = _find_matching_article(surname, year, wiki_stems)
            if article_stem:
                matched.append((citation, article_stem))
                log.debug("Matched: %r → [[%s]]", citation[:60], article_stem)
            else:
                log.debug("No match for: %r (surname=%r, year=%r)", citation[:60], surname, year)

        if not matched:
            log.info("Lecture %02d: no matched papers — skipping", number)
            continue

        log.info("Lecture %02d (%s): enriching with %d paper links",
                 number, title[:50], len(matched))

        current_text = lecture_file.read_text(encoding="utf-8")
        matched_summary = "\n".join(
            f"- [[{stem}]] — {citation[:80]}" for citation, stem in matched
        )
        user_content = (
            f"Current article:\n\n{current_text}\n\n"
            f"---\n\nMatched paper wiki articles for this lecture:\n{matched_summary}"
        )

        updated = call_claude(client, ENRICH_SYSTEM_PROMPT, user_content)
        if updated:
            lecture_file.write_text(updated, encoding="utf-8")
            log.info("Updated: %s", lecture_file.name)
            enriched += 1
        else:
            log.error("Claude failed to enrich lecture %02d", number)

    log.info("Enrich complete: %d/%d lecture articles updated", enriched, len(lectures))
    rebuild_index()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wiki ingestion pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    seed_p = subparsers.add_parser("seed", help="Seed wiki from reading list JSON")
    seed_p.add_argument("json_file", type=Path, help="Path to reading_list.json")

    pdf_p = subparsers.add_parser("pdf", help="Ingest a single PDF")
    pdf_p.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    pdf_p.add_argument("--fast", action="store_true",
                       help="Skip marker/nougat, use pypdf only")

    subparsers.add_parser("watch", help="Watch raw/ for new PDFs")

    retry_p = subparsers.add_parser("retry", help="Retry PDFs that previously failed (in unprocessed/)")
    retry_p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    retry_p.add_argument("--fast", action="store_true",
                         help="Skip marker/nougat, use pypdf only (much faster on CPU)")

    scan_p = subparsers.add_parser("scan", help="Bulk-ingest all PDFs from a directory")
    scan_p.add_argument(
        "scan_path", type=Path, nargs="?", default=None,
        help=f"Directory to scan (default: READINGS_PATH = {BASE_READINGS_PATH})",
    )
    scan_p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    scan_p.add_argument("--fast", action="store_true",
                        help="Skip marker/nougat, use pypdf only (much faster on CPU)")

    enrich_p = subparsers.add_parser(
        "enrich",
        help="Cross-link lecture articles to their cited paper wiki articles",
    )
    enrich_p.add_argument(
        "json_file", type=Path, nargs="?", default=None,
        help="Path to reading_list.json (default: <wiki_root>/reading_list.json)",
    )

    args = parser.parse_args()

    if args.mode == "seed":
        seed_from_json(args.json_file)
    elif args.mode == "pdf":
        success = ingest_pdf(args.pdf_path, fast=args.fast)
        sys.exit(0 if success else 1)
    elif args.mode == "watch":
        watch_raw_dir()
    elif args.mode == "retry":
        retry_unprocessed(workers=args.workers, fast=args.fast)
    elif args.mode == "scan":
        scan_directory(args.scan_path or BASE_READINGS_PATH, workers=args.workers, fast=args.fast)
    elif args.mode == "enrich":
        json_file = args.json_file or (WIKI_PATH / "reading_list.json")
        enrich_lecture_articles(json_file)


if __name__ == "__main__":
    main()
