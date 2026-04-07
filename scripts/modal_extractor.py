"""
modal_extractor.py — Run marker-pdf on a Modal GPU cloud instance.

No CLI required. Authentication is handled via:
  1. Running `modal token new` once (writes ~/.modal.toml), OR
  2. Setting MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.

Usage from other scripts:
    from modal_extractor import run_modal_extraction
    text = run_modal_extraction(pdf_path)   # returns extracted markdown string
"""

from pathlib import Path

import config_loader as cfg

import modal

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------

app = modal.App(cfg.get("wiki.app_name", "llm-wiki-pdf"))

_gpu = cfg.get("modal.gpu", "T4")
_memory = cfg.get("modal.memory", 8192)
_timeout = cfg.get("modal.timeout", 300)

# Build a Debian image with marker-pdf installed.
# The image is built once and cached by Modal; subsequent runs are fast.
_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",    # OpenCV dependency for marker-pdf
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "poppler-utils",      # pdfinfo/pdfimages used by some marker backends
    )
    .pip_install("marker-pdf")
)


@app.function(gpu=_gpu, image=_image, timeout=_timeout, memory=_memory)
def _extract_pdf_remote(pdf_bytes: bytes, filename: str) -> str:
    """
    Run marker_single on a GPU container and return the extracted markdown.
    This function runs entirely inside Modal's cloud — do not call it directly.
    """
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
            timeout=_timeout - 30,  # leave 30s headroom
        )

        for md in out_dir.rglob("*.md"):
            text = md.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return text

    return ""


# ---------------------------------------------------------------------------
# Public interface — called from ingest.py
# ---------------------------------------------------------------------------

def run_modal_extraction(pdf_path: Path) -> str:
    """
    Send a local PDF to Modal for GPU-accelerated extraction.

    Returns the extracted markdown text, or raises RuntimeError on failure.
    Requires Modal auth (see module docstring).
    """
    pdf_bytes = pdf_path.read_bytes()

    try:
        with app.run():
            text = _extract_pdf_remote.remote(pdf_bytes, pdf_path.name)
    except Exception as e:
        raise RuntimeError(f"Modal extraction failed: {e}") from e

    if not text.strip():
        raise RuntimeError("Modal extraction returned empty text")

    return text
