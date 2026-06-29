"""
utils.py — Shared helper utilities for the Ahoum evaluation pipeline.

Provides:
  - Configuration loading (YAML)
  - Reproducible seeding
  - Structured logger
  - JSON schema validation helpers
  - Timing decorators
"""

from __future__ import annotations

import functools
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import yaml
from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Console (rich)
# ---------------------------------------------------------------------------
console = Console()


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def get_logger(name: str = "ahoum", level: str = "INFO") -> logging.Logger:
    """Return a rich-formatted logger."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


logger = get_logger()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file.

    Searches for config.yaml in:
      1. ``config_path`` (if provided)
      2. Current working directory
      3. Project root (two levels up from this file)
    """
    search_paths: list[Path] = []
    if config_path:
        search_paths.append(Path(config_path))
    search_paths.extend(
        [
            Path.cwd() / "config.yaml",
            Path(__file__).resolve().parent.parent / "config.yaml",
        ]
    )
    for path in search_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            logger.debug("Loaded config from %s", path)
            return cfg
    raise FileNotFoundError(
        f"config.yaml not found in any of: {[str(p) for p in search_paths]}"
    )


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.debug("Seed set to %d", seed)


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------
def timeit(func: F) -> F:
    """Decorator that logs wall-clock execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("%s completed in %.3fs", func.__qualname__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON; return ``default`` on failure."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Try to extract first {...} block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
    return default


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Serialize data to a JSON file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False, default=str)
    logger.info("Saved JSON to %s", path)


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Score validation
# ---------------------------------------------------------------------------
def clamp_score(score: Any, lo: int = 1, hi: int = 5) -> int:
    """Force a value to be an integer in [lo, hi]."""
    try:
        val = int(round(float(score)))
    except (TypeError, ValueError):
        val = (lo + hi) // 2
    return max(lo, min(hi, val))


def clamp_confidence(conf: Any) -> float:
    """Force a value to be a float in [0, 1]."""
    try:
        val = float(conf)
    except (TypeError, ValueError):
        val = 0.5
    return round(max(0.0, min(1.0, val)), 4)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """Create directory (including parents) and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent
