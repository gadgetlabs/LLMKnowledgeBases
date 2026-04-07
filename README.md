# llm-wiki

Karpathy proposes using LLMs to build personal knowledge bases stored as plain markdown files. Raw sources — papers, articles, repos — go into a folder. An LLM "compiles" them into an interlinked wiki of markdown articles with summaries, backlinks, and concept categories. Obsidian serves as the frontend for browsing. The key insight: you never write the wiki yourself. The LLM writes everything; you just steer with questions.

Once the wiki reaches sufficient scale (~100 articles, ~400K words), you query it through an LLM agent that reads the relevant files directly — no vector databases or RAG pipelines needed at this scale. Answers get rendered as markdown, Marp slides, or plots, then filed back into the wiki so every question compounds the knowledge base. Periodic "linting" passes catch inconsistencies, fill gaps, and suggest new connections.

We like this because the entire system is just files in folders. No databases, no infrastructure, no frameworks. Markdown is universal, readable, and version-controllable. Obsidian is optional — it's just a viewer. The LLM does the tedious work of summarising, linking, and organising while you focus on asking better questions. The compounding loop — where outputs become inputs — means the system gets more valuable the more you use it, without you manually maintaining anything.

---

## Setup

```bash
pip install anthropic pypdf tantivy watchdog pyyaml
export ANTHROPIC_API_KEY="sk-ant-..."
```

Optional extras:

```bash
# Local GPU extraction (higher quality than pypdf)
pip install marker-pdf nougat-ocr

# Cloud GPU extraction via Modal (fast + high quality, no local GPU needed)
pip install modal
modal token new          # one-time auth — writes ~/.modal.toml
```

Set `WIKI_PATH` to use a non-default vault location (default: `~/policing-wiki`):

```bash
export WIKI_PATH=/path/to/your/vault
```

---

## Configuration

`config.yaml` lives at the vault root and is **gitignored** — it contains your local paths and should not be committed. An annotated example is provided in `example_config.yaml`.

```bash
cp example_config.yaml config.yaml
# then edit config.yaml with your paths and preferences
```

| Key | Default | Description |
|-----|---------|-------------|
| `wiki.topic` | `"your knowledge domain"` | Domain description injected into all LLM prompts |
| `paths.readings_dir` | — | Default directory for `ingest.py scan` |
| `paths.raw_dir` | `<vault>/raw` | Drop folder for watch mode |
| `extraction.mode` | `pypdf` | `pypdf` \| `local` \| `modal` |
| `extraction.workers` | `3` | Parallel workers for scan/retry |
| `model.name` | `claude-sonnet-4-20250514` | Claude model for all API calls |
| `lint.batch_size` | `7` | Articles per Claude call |
| `qa.top_k` | `15` | Search results used as context |
| `qa.search_outputs` | `false` | Include prior Q&A answers in search |

---

## Quick start

```bash
# 1. Create your config
cp example_config.yaml config.yaml
# edit config.yaml — set wiki.topic and paths.readings_dir at minimum

# 2. (Optional) Create a structured reading list for seed/enrich
cp reading_list.example.json reading_list.json
# edit reading_list.json with your lectures and citations

# 3. Seed from a structured reading list (optional)
python scripts/ingest.py seed reading_list.json

# 3. Ingest all PDFs from a directory
python scripts/ingest.py scan /path/to/papers --fast   # --fast = pypdf, no GPU

# 4. Ask a question
python scripts/qa.py "What does the evidence say about X?"

# 5. Check wiki quality
python scripts/lint.py
```

---

## Scripts

### `ingest.py`

| Subcommand | Usage | Description |
|-----------|-------|-------------|
| `seed` | `ingest.py seed reading_list.json` | Generate articles from a structured JSON list |
| `pdf` | `ingest.py pdf paper.pdf [--fast]` | Ingest a single document |
| `scan` | `ingest.py scan [dir] [--workers N] [--fast]` | Bulk-ingest a directory |
| `watch` | `ingest.py watch` | Auto-ingest documents dropped in `raw/` |
| `retry` | `ingest.py retry [--workers N] [--fast]` | Re-process documents that previously failed |
| `enrich` | `ingest.py enrich [reading_list.json]` | Cross-link source articles to cited documents |

### `qa.py`

```bash
python scripts/qa.py "your question"
python scripts/qa.py "your question" --slides       # Marp presentation output
python scripts/qa.py "your question" --stdout-only  # Print only, don't save
python scripts/qa.py --reindex                      # Rebuild search index
```

Answers are saved to `outputs/YYYY-MM-DD-question-slug.md` and can be browsed in Obsidian.

### `lint.py`

```bash
python scripts/lint.py
python scripts/lint.py --batch-size 3   # smaller batches = cheaper per call
```

Saves a full report to `outputs/lint-report-YYYY-MM-DD.md`.

---

## Extraction modes

| Mode | Speed | Quality | Requires |
|------|-------|---------|----------|
| `pypdf` | Fast | Plain text | `pip install pypdf` |
| `local` | Slow on CPU | Good layout, equations | `pip install marker-pdf nougat-ocr` |
| `modal` | Fast (cloud GPU) | Same as local | `pip install modal` + `modal token new` |

Set `extraction.mode` in `config.yaml`. Modal charges per second (~$0.10–0.30 per paper on T4); the container image is built once then cached.

---

## Folder structure

```
<vault>/
├── config.yaml          # configuration (edit this)
├── reading_list.json    # optional structured source list
├── index.md             # auto-generated topic index
├── raw/                 # drop documents here (watch mode)
├── wiki/                # LLM-generated articles (Obsidian vault)
├── outputs/             # Q&A answers, lint reports, slide decks
├── unprocessed/         # documents that failed extraction (retry later)
└── scripts/
    ├── ingest.py
    ├── qa.py
    ├── lint.py
    ├── config_loader.py
    ├── modal_extractor.py
    └── raycast/         # Raycast script commands
```

---

## Raycast

Add `scripts/raycast/` as a Script Commands directory in Raycast → Settings → Extensions → Script Commands.

```bash
chmod +x scripts/raycast/*.sh
```

---

## Open in Obsidian

Open the vault root as an Obsidian vault to browse [[backlinks]] and the graph view. Obsidian is purely a viewer — the wiki is just markdown files and works without it.
