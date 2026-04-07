"""
modal_extractor.py — Run marker-pdf on a Modal GPU cloud instance.

Setup (one-time):
  1. pip install modal
  2. modal token new   # browser auth, writes ~/.modal.toml

After that, run_modal_extraction() works from any Python code with no
further CLI interaction.

Authentication via env vars instead of ~/.modal.toml:
  export MODAL_TOKEN_ID=...
  export MODAL_TOKEN_SECRET=...
"""

import threading
from pathlib import Path

import config_loader as cfg
import modal

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------

_app_name = cfg.get("wiki.app_name", "llm-wiki-pdf")
_gpu      = cfg.get("modal.gpu", "T4")
_memory   = cfg.get("modal.memory", 8192)
_timeout  = cfg.get("modal.timeout", 300)

app = modal.App(_app_name)

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "poppler-utils",
    )
    .pip_install("marker-pdf")
)


@app.function(gpu=_gpu, image=_image, timeout=_timeout, memory=_memory)
def _extract_pdf_remote(pdf_bytes: bytes, filename: str) -> str:
    """Runs inside Modal's cloud — do not call directly."""
    import subprocess
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = P(tmp)
        pdf_path = tmp_path / filename
        pdf_path.write_bytes(pdf_bytes)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        subprocess.run(
            ["marker_single", str(pdf_path), "--output_dir", str(out_dir)],
            capture_output=True,
            timeout=_timeout - 30,
        )

        for md in out_dir.rglob("*.md"):
            text = md.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return text

    return ""


# ---------------------------------------------------------------------------
# Public interface — called from ingest.py
# ---------------------------------------------------------------------------

# app.run() is not re-entrant — serialise all calls through a lock.
# ingest.py also caps workers=1 for modal mode, so this is belt-and-braces.
_lock = threading.Lock()


def run_modal_extraction(pdf_path: Path) -> str:
    """
    Send a local PDF to Modal for GPU-accelerated extraction.
    Thread-safe. Raises RuntimeError on failure.
    """
    pdf_bytes = pdf_path.read_bytes()

    with _lock:
        try:
            with app.run():
                text = _extract_pdf_remote.remote(pdf_bytes, pdf_path.name)
        except Exception as e:
            raise RuntimeError(f"Modal extraction failed: {e}") from e

    if not text or not text.strip():
        raise RuntimeError("Modal returned empty text")

    return text
