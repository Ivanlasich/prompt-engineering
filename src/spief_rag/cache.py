"""Exact result cache, isolated from primary uncached quality experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .corpus import digest


class ExactCache:
    """TTL cache keyed by exact question and all result-affecting versions."""

    def __init__(self, folder: Path, ttl: float = 3600):
        """Keep local result caching opt-in and bounded by TTL."""
        self.folder, self.ttl = folder, ttl

    def key(self, question: str, corpus_version: str, config: dict, model: str) -> str:
        """No semantic equivalence assumptions, cross-corpus reuse or secret material."""
        return digest(
            {
                "question": question,
                "corpus": corpus_version,
                "config": config,
                "model": model,
            }
        )

    def get(self, key: str) -> dict | None:
        """Read only unexpired entries; malformed entries behave as misses."""
        try:
            row = json.loads((self.folder / f"{key}.json").read_text())
            return row["value"] if time.time() - row["timestamp"] < self.ttl else None
        except (OSError, ValueError, KeyError):
            return None

    def put(self, key: str, value: dict) -> None:
        """Atomically replace a local exact-match entry."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{key}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"timestamp": time.time(), "value": value}, ensure_ascii=False)
        )
        temp.replace(path)
