#!/usr/bin/env python3
"""
lint.py — Quality assurance linter for the llm-wiki.

Reads all wiki articles in batches, sends them to Claude for review,
and produces a consolidated lint report in outputs/.

Usage:
  python scripts/lint.py
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
WIKI_PATH = Path(os.environ.get("WIKI_PATH", "~/policing-wiki")).expanduser()
WIKI_DIR = Path(cfg.get("paths.wiki_dir", str(WIKI_PATH / "wiki"))).expanduser()
OUTPUTS_DIR = Path(cfg.get("paths.outputs_dir", str(WIKI_PATH / "outputs"))).expanduser()

MODEL = cfg.get("model.name", "claude-sonnet-4-20250514")
DOMAIN = cfg.get("wiki.topic", "your knowledge domain")
BATCH_SIZE = cfg.get("lint.batch_size", 7)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LINT_SYSTEM_PROMPT = f"""\
You are a quality assurance agent for a wiki knowledge base on {DOMAIN}. \
Review these wiki articles and report:

1. INCONSISTENCIES: Do any articles contradict each other on facts, dates, or findings?
2. MISSING LINKS: Are there concepts mentioned but not [[backlinked]] that should be?
3. GAPS: What topics are referenced across multiple articles but don't have their own wiki page yet?
4. STALE/VAGUE CLAIMS: Are there claims that lack specific citations or seem unsupported?
5. SUGGESTED QUESTIONS: What interesting research questions arise from reading these articles together?

Be specific — cite the article filenames and quote the relevant passages. \
Format as a markdown report.\
"""


# ---------------------------------------------------------------------------
# Lint counters
# ---------------------------------------------------------------------------

def _count_issues(report_text: str) -> dict[str, int]:
    """Extract rough counts of each issue type from a batch report."""
    counts: dict[str, int] = {
        "inconsistencies": 0,
        "missing_links": 0,
        "gaps": 0,
        "vague_claims": 0,
        "suggested_questions": 0,
    }
    section_map = {
        "INCONSISTENCIES": "inconsistencies",
        "MISSING LINKS": "missing_links",
        "GAPS": "gaps",
        "STALE": "vague_claims",
        "VAGUE": "vague_claims",
        "SUGGESTED QUESTIONS": "suggested_questions",
    }
    current_key: str | None = None
    for line in report_text.splitlines():
        stripped = line.strip().upper()
        for keyword, key in section_map.items():
            if keyword in stripped and (stripped.startswith("#") or stripped.startswith("**")):
                current_key = key
                break
        if current_key and re.match(r"^[-*•]\s+", line.strip()):
            counts[current_key] += 1
    return counts


# ---------------------------------------------------------------------------
# Main lint logic
# ---------------------------------------------------------------------------

def lint_wiki(batch_size: int = BATCH_SIZE) -> None:
    if not WIKI_DIR.exists() or not any(WIKI_DIR.glob("*.md")):
        log.error("Wiki is empty — nothing to lint")
        sys.exit(1)

    articles = sorted(WIKI_DIR.glob("*.md"))
    log.info("Linting %d articles in batches of %d…", len(articles), batch_size)

    client = anthropic.Anthropic()
    batch_reports: list[str] = []
    totals: dict[str, int] = {
        "inconsistencies": 0,
        "missing_links": 0,
        "gaps": 0,
        "vague_claims": 0,
        "suggested_questions": 0,
    }

    # Process in batches
    for batch_start in range(0, len(articles), BATCH_SIZE):
        batch = articles[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        log.info("Batch %d: %s", batch_num, [f.name for f in batch])

        # Assemble article text for this batch
        parts = []
        for md_file in batch:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            parts.append(f"## File: {md_file.name}\n\n{content}")

        user_content = "\n\n---\n\n".join(parts)

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=LINT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            report_text = response.content[0].text
        except anthropic.APIError as e:
            log.error("Claude API error on batch %d: %s", batch_num, e)
            report_text = f"*Error processing batch {batch_num}: {e}*"

        batch_reports.append(
            f"## Batch {batch_num}: {', '.join(f.name for f in batch)}\n\n{report_text}"
        )

        # Accumulate counts
        counts = _count_issues(report_text)
        for k, v in counts.items():
            totals[k] += v

    # Build combined report
    date_str = datetime.date.today().isoformat()
    header = f"# Wiki Lint Report — {date_str}\n\n"
    header += f"**Articles reviewed:** {len(articles)}  \n"
    header += f"**Batches:** {len(batch_reports)}\n\n"
    header += "---\n\n"

    combined = header + "\n\n".join(batch_reports)

    # Save report
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"lint-report-{date_str}.md"
    out_path.write_text(combined, encoding="utf-8")
    log.info("Lint report saved: %s", out_path)

    # Print summary
    print(f"\nLint complete — {len(articles)} articles reviewed")
    print(f"  Inconsistencies:      {totals['inconsistencies']}")
    print(f"  Missing links:        {totals['missing_links']}")
    print(f"  Gap topics:           {totals['gaps']}")
    print(f"  Vague/stale claims:   {totals['vague_claims']}")
    print(f"  Suggested questions:  {totals['suggested_questions']}")
    print(f"\nFull report: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wiki quality linter")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Articles per Claude batch (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    lint_wiki(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
