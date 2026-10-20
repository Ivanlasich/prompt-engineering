"""Load and rebuild the provided SPIEF index without fabricating source documents."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from .settings import ROOT
from .telemetry import Ledger

ARCHIVE = ROOT / "task" / "assets (2).zip"


def digest(value: object) -> str:
    """Stable content hash for configs, data and cache keys."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


class Corpus:
    """Immutable chunk metadata and row-aligned normalized embedding matrix."""

    def __init__(self, chunks: list[dict], matrix: np.ndarray, stats: dict):
        """Validate row identity before normalization and versioning."""
        self.chunks, self.stats = chunks, stats
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or not chunks:
            raise ValueError("index_row_mismatch")
        required = {
            "chunk_id",
            "source_id",
            "session",
            "time_start",
            "speakers",
            "text",
        }
        if any(not required.issubset(r) or not r["text"].strip() for r in chunks):
            raise ValueError("invalid_chunk")
        self.by_id = {r["chunk_id"]: r for r in chunks}
        if len(self.by_id) != len(chunks):
            raise ValueError("duplicate_chunk_id")
        matrix = np.asarray(matrix, dtype=np.float32)
        self.raw_matrix = matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(matrix).all() or np.any(norms == 0):
            raise ValueError("invalid_embedding_matrix")
        self.matrix = matrix / norms
        self.version = digest(
            {
                "chunks": chunks,
                "model": stats.get("embedding_model"),
                "vectors_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
            }
        )

    @classmethod
    def load(
        cls, directory: str | Path | None = None, archive: Path = ARCHIVE
    ) -> "Corpus":
        """Read a prepared folder or the exact three assets directly from ZIP."""
        if directory:
            p = Path(directory)
            raw = (p / "chunks.jsonl").read_bytes()
            vectors = np.load(p / "embeddings.npy", allow_pickle=False)
            stats = json.loads((p / "index_stats.json").read_text())
        else:
            with zipfile.ZipFile(archive) as z:
                raw = z.read("assets/spief_index/chunks.jsonl")
                vectors = np.load(
                    io.BytesIO(z.read("assets/spief_index/embeddings.npy")),
                    allow_pickle=False,
                )
                stats = json.loads(z.read("assets/spief_index/index_stats.json"))
        chunks = [
            json.loads(line) for line in raw.decode().splitlines() if line.strip()
        ]
        return cls(chunks, vectors, stats)

    def prepare(self, target: Path) -> None:
        """Materialize the canonical chunk source and supplied vectors safely."""
        if target.exists() and any(target.iterdir()):
            raise ValueError("target_not_empty")
        target.mkdir(parents=True, exist_ok=True)
        (target / "chunks.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.chunks)
        )
        np.save(target / "embeddings.npy", self.raw_matrix)
        (target / "index_stats.json").write_text(
            json.dumps(
                {
                    **self.stats,
                    "corpus_version": self.version,
                    "origin": "provided course index; historical build cost is not a new measurement",
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def rebuild_index(
    client,
    source: Path,
    target: Path,
    *,
    limit: int | None = None,
    batch_size: int = 16,
) -> dict:
    """Re-embed canonical chunks, without query instruction, and save row mapping."""
    import time

    if target.exists() and any(target.iterdir()):
        raise ValueError("target_not_empty")
    chunks = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    if limit:
        chunks = chunks[:limit]
    ledger, started, vectors = Ledger(max_tokens=5000000), time.perf_counter(), []
    try:
        for offset in range(0, len(chunks), batch_size):
            vectors.extend(
                client.embed(
                    [r["text"] for r in chunks[offset : offset + batch_size]], ledger
                )
            )
    except Exception:
        target.mkdir(parents=True, exist_ok=True)
        (target / "failed_build_usage.json").write_text(
            json.dumps(ledger.events, indent=2)
        )
        raise
    stats = {
        "documents": len({r["source_id"] for r in chunks}),
        "chunks": len(chunks),
        "embedding_model": client.settings.embedding_model,
        "vector_dim": len(vectors[0]),
        "usage": ledger.summary(),
        "events": ledger.events,
        "build_seconds": time.perf_counter() - started,
        "input_sha256": digest(chunks),
        "partial_sample": bool(limit),
    }
    corpus = Corpus(chunks, np.asarray(vectors, dtype=np.float32), stats)
    corpus.prepare(target)
    return stats
