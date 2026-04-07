"""
config_loader.py — Load config.yaml and expose a simple get() helper.

Usage in other scripts:
    import config_loader as cfg
    model = cfg.get("model.name", "claude-sonnet-4-20250514")
"""

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

WIKI_PATH = Path(os.environ.get("WIKI_PATH", "~/policing-wiki")).expanduser()
CONFIG_PATH = WIKI_PATH / "config.yaml"

_cache: dict | None = None


def load() -> dict:
    """Load config.yaml once and cache the result."""
    global _cache
    if _cache is not None:
        return _cache

    if not CONFIG_PATH.exists():
        _cache = {}
        return _cache

    try:
        import yaml  # pyyaml
        _cache = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except ImportError:
        log.warning("pyyaml not installed — config.yaml ignored (pip install pyyaml)")
        _cache = {}
    except Exception as e:
        log.warning("Failed to parse config.yaml: %s", e)
        _cache = {}

    return _cache


def get(key: str, default=None):
    """
    Retrieve a config value using dot-notation.

    Examples:
        cfg.get("model.name")
        cfg.get("extraction.mode", "pypdf")
        cfg.get("qa.top_k", 15)
    """
    cfg = load()
    parts = key.split(".")
    val = cfg
    for part in parts:
        if not isinstance(val, dict):
            return default
        val = val.get(part)
        if val is None:
            return default
    return val
