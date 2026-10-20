"""Record portable source/data hashes and package versions without local secrets."""

import hashlib
import json
import platform
from importlib.metadata import version

from spief_rag.corpus import Corpus
from spief_rag.evaluation import read_jsonl
from spief_rag.settings import ROOT


def main():
    """Snapshot the code used for reproducibility, not user-specific environment values."""
    hashes = {}
    for folder in [
        "src",
        "scripts",
        "tests",
        "configs",
        "data/gold",
        "docs",
        "notebooks",
        "certs",
        "media",
    ]:
        for path in sorted((ROOT / folder).rglob("*")):
            if any(
                part.endswith(".egg-info") or part == "__pycache__"
                for part in path.parts
            ):
                continue
            if path.is_file() and path.suffix in {
                ".py",
                ".json",
                ".jsonl",
                ".txt",
                ".md",
                ".html",
                ".ipynb",
                ".crt",
                ".mp4",
                ".vtt",
                ".png",
            }:
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    fixed = [
        "README.md",
        "pyproject.toml",
        "requirements.lock",
        "requirements-media.txt",
        ".env.example",
        ".gitignore",
        "task/assets (2).zip",
    ]
    fixed += [str(p.relative_to(ROOT)) for p in (ROOT / "task").glob("*.pdf")]
    for name in fixed:
        path = ROOT / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    coverage = {}
    models = set()
    counts = {}
    for path in sorted((ROOT / "reports/runs").glob("*.jsonl")):
        rows = read_jsonl(path)
        counts[path.name] = len(rows)
        for row in rows:
            for event in row.get("run", {}).get("trace", []):
                if event["type"] == "tool_call":
                    coverage[event["name"]] = coverage.get(event["name"], 0) + 1
            for event in row.get("run", {}).get("events", []):
                if event.get("operation") == "generation":
                    models.add(event["model"])
    demos = ROOT / "reports/native_tools_demo.json"
    forced = {}
    if demos.exists():
        for row in json.loads(demos.read_text()):
            if not row["error"]:
                forced[row["tool"]] = forced.get(row["tool"], 0) + 1
    (ROOT / "reports/live_tool_coverage.json").write_text(
        json.dumps(
            {
                "automatic_model_calls": coverage,
                "successful_forced_demonstrations": forced,
            },
            indent=2,
        )
    )
    result = {
        "author": "Иван Лазичный",
        "python": platform.python_version(),
        "packages": {
            p: version(p)
            for p in [
                "spief-rag",
                "numpy",
                "pydantic",
                "httpx",
                "matplotlib",
                "pytest",
                "nbformat",
                "nbclient",
            ]
        },
        "code_and_data_sha256": hashes,
        "corpus_version": Corpus.load().version,
        "actual_generation_models": sorted(models),
        "run_record_counts": counts,
        "runtime": "public GigaChat REST v1; OAuth; scope confirmed GIGACHAT_API_B2B",
        "snapshot_date": "2026-10-20",
        "snapshot_purpose": "Final portable deliverable; archived primary runs retain their original signatures and separate legacy snapshot",
        "archived_benchmark_manifest": "reports/reproducibility_legacy.json",
        "excluded_build_metadata": "*.egg-info and __pycache__ are installation artifacts, not portable sources",
        "credentials_included": False,
    }
    (ROOT / "reports/reproducibility.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(
        "Manifest written:",
        len(hashes),
        "hashed files; observed model versions:",
        sorted(models),
    )


if __name__ == "__main__":
    main()
