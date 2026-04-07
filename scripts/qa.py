#!/usr/bin/env python3
"""
qa.py — Q&A agent for the llm-wiki.

Usage:
  python scripts/qa.py "your question here"
  python scripts/qa.py "your question" --slides
  python scripts/qa.py "your question" --stdout-only
  python scripts/qa.py --reindex
"""

import argparse
import datetime
import logging
import os
import re
import sys
from pathlib import Path

import anthropic
import config_loader as cfg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WIKI_PATH = Path(os.environ.get("WIKI_PATH", str(Path(__file__).parent.parent))).expanduser()
WIKI_DIR = Path(cfg.get("paths.wiki_dir", str(WIKI_PATH / "wiki"))).expanduser()
OUTPUTS_DIR = Path(cfg.get("paths.outputs_dir", str(WIKI_PATH / "outputs"))).expanduser()
SEARCH_INDEX_DIR = WIKI_PATH / "scripts" / ".search_index"

MODEL = cfg.get("model.name", "claude-sonnet-4-20250514")
DOMAIN = cfg.get("wiki.topic", "your knowledge domain")
TOP_K = cfg.get("qa.top_k", 15)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = f"""\
You are a Q&A agent for a knowledge base on {DOMAIN}. \
You have been given relevant wiki articles as context.

Answer the user's question based on the wiki content. Be specific and cite which articles \
support your claims. If the wiki doesn't contain enough information to answer fully, say so and \
suggest what additional sources or topics might help.

When referencing wiki articles, use [[article-name]] Obsidian links.\
"""

MARP_FRONTMATTER = """\
---
marp: true
theme: default
paginate: true
---

"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_title(content: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def slugify_question(question: str) -> str:
    slug = question.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug[:60]


# ---------------------------------------------------------------------------
# Tantivy index management
# ---------------------------------------------------------------------------

def _build_schema():
    import tantivy
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("filename", stored=True)
    sb.add_text_field("title", stored=True)
    sb.add_text_field("body", stored=True)
    return sb.build()


def build_index(force: bool = False) -> None:
    """Build (or rebuild) the tantivy search index from wiki/*.md files."""
    try:
        import tantivy
    except ImportError:
        log.error("tantivy not installed — run: pip install tantivy")
        sys.exit(1)

    SEARCH_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    schema = _build_schema()

    log.info("Building search index…")
    index = tantivy.Index(schema, path=str(SEARCH_INDEX_DIR))
    writer = index.writer()

    count = 0
    for md_file in WIKI_DIR.glob("*.md"):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        title = _extract_title(content)
        doc = tantivy.Document()
        doc.add_text("filename", md_file.name)
        doc.add_text("title", title)
        doc.add_text("body", content)
        writer.add_document(doc)
        count += 1

    writer.commit()
    log.info("Search index built: %d articles indexed", count)


def _open_or_build_index():
    """Open the search index, building it first if it doesn't exist."""
    import tantivy

    meta_file = SEARCH_INDEX_DIR / "meta.json"
    if not SEARCH_INDEX_DIR.exists() or not meta_file.exists():
        if not WIKI_DIR.exists() or not any(WIKI_DIR.glob("*.md")):
            log.warning("Wiki is empty — no articles to search")
            return None
        build_index()

    schema = _build_schema()
    return tantivy.Index(schema, path=str(SEARCH_INDEX_DIR))


def _candidate_files() -> list[Path]:
    """Return the pool of markdown files available for search."""
    files = list(WIKI_DIR.glob("*.md"))
    if cfg.get("qa.search_outputs", False) and OUTPUTS_DIR.exists():
        files += list(OUTPUTS_DIR.glob("*.md"))
        log.info("Including outputs/ in search pool (%d total files)", len(files))
    return files


def search_wiki(question: str) -> list[Path]:
    """Search the wiki and return a list of matching article paths."""
    try:
        import tantivy  # noqa: F401
    except ImportError:
        log.warning("tantivy not available — using all-articles fallback")
        return _candidate_files()[:TOP_K]

    index = _open_or_build_index()
    if index is None:
        return []

    try:
        index.reload()
        searcher = index.searcher()
        query = index.parse_query(question, ["title", "body"])
        results = searcher.search(query, limit=TOP_K)

        paths: list[Path] = []
        for _score, doc_addr in results.hits:
            doc = searcher.doc(doc_addr)
            try:
                filename = doc.get_first("filename")
            except Exception:
                filename = doc["filename"][0]
            candidate = WIKI_DIR / filename
            if candidate.exists():
                paths.append(candidate)

        # If search_outputs is enabled, supplement with matching output files
        if cfg.get("qa.search_outputs", False) and OUTPUTS_DIR.exists():
            output_files = list(OUTPUTS_DIR.glob("*.md"))
            remaining = TOP_K - len(paths)
            if remaining > 0 and output_files:
                paths += output_files[:remaining]

        log.info("Search returned %d results", len(paths))
        return paths
    except Exception as e:
        log.error("Search error: %s — falling back to all articles", e)
        return _candidate_files()[:TOP_K]


# ---------------------------------------------------------------------------
# Q&A query
# ---------------------------------------------------------------------------

def query_wiki(question: str, slides: bool = False, stdout_only: bool = False) -> None:
    """Search wiki, build context, call Claude, print and optionally save."""
    if not WIKI_DIR.exists() or not any(WIKI_DIR.glob("*.md")):
        log.error("Wiki is empty. Run 'python scripts/ingest.py seed reading_list.json' first.")
        sys.exit(1)

    # 1. Search
    result_paths = search_wiki(question)
    if not result_paths:
        log.warning("No search results — using all wiki articles")
        result_paths = list(WIKI_DIR.glob("*.md"))[:TOP_K]

    # 2. Read articles into context
    context_parts = []
    for path in result_paths:
        content = path.read_text(encoding="utf-8", errors="replace")
        context_parts.append(f"### Article: {path.stem}\n\n{content}")

    context = "\n\n---\n\n".join(context_parts)
    log.info("Context: %d articles, %d chars", len(result_paths), len(context))

    user_message = f"Wiki context:\n\n{context}\n\n---\n\nQuestion: {question}"

    # 3. Ask Claude
    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=QA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = response.content[0].text
    except anthropic.APIError as e:
        log.error("Claude API error: %s", e)
        sys.exit(1)

    # 4. Format output
    if slides:
        output = MARP_FRONTMATTER
        paragraphs = re.split(r"\n\n+", answer.strip())
        current_slide: list[str] = []
        current_len = 0
        slides_out: list[str] = []

        for para in paragraphs:
            if current_len + len(para) > 800 and current_slide:
                slides_out.append("\n\n".join(current_slide))
                current_slide = [para]
                current_len = len(para)
            else:
                current_slide.append(para)
                current_len += len(para)

        if current_slide:
            slides_out.append("\n\n".join(current_slide))

        output += "\n\n---\n\n".join(slides_out)
    else:
        output = answer

    # 5. Print
    print(output)

    # 6. Save (unless stdout-only)
    if not stdout_only:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.date.today().isoformat()
        slug = slugify_question(question)
        out_path = OUTPUTS_DIR / f"{date_str}-{slug}.md"
        out_path.write_text(output, encoding="utf-8")
        log.info("Saved to: %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wiki Q&A agent")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask")
    parser.add_argument("--slides", action="store_true", help="Output in Marp slide format")
    parser.add_argument("--stdout-only", action="store_true", help="Print only, don't save file")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild search index")
    args = parser.parse_args()

    if args.reindex:
        build_index(force=True)
        return

    if not args.question:
        parser.print_help()
        sys.exit(1)

    query_wiki(args.question, slides=args.slides, stdout_only=args.stdout_only)


if __name__ == "__main__":
    main()
