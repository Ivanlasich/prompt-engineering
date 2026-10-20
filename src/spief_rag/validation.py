"""Data and artifact checks independent of API availability."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from .contracts import normalized
from .evaluation import read_jsonl
from .settings import ROOT


def approved_author() -> str | None:
    """Read the author's confirmation only while the approved gold files still match."""
    path = ROOT / "reports/submission/author_approval.json"
    if not path.exists():
        return None
    approval = json.loads(path.read_text())
    if approval.get("status") != "checked_and_approved_by_author":
        return None
    hashes = approval.get("artifact_sha256", {})
    for name in [
        "data/gold/dev.jsonl",
        "data/gold/test.jsonl",
        "data/gold/judge_calibration.jsonl",
    ]:
        if hashes.get(name) != hashlib.sha256((ROOT / name).read_bytes()).hexdigest():
            return None
    return approval["author"]["name"]


def validate_gold(corpus) -> dict:
    """Verify split isolation, reference coverage and every annotated source span."""
    dev = read_jsonl(ROOT / "data/gold/dev.jsonl")
    test = read_jsonl(ROOT / "data/gold/test.jsonl")
    assert len(dev) == 20 and len(test) == 40, "unexpected_split_size"
    assert not ({r["group_id"] for r in dev} & {r["group_id"] for r in test}), (
        "source_leakage"
    )
    allrows = dev + test
    assert len({r["id"] for r in allrows}) == 60, "duplicate_case_id"
    for case in allrows:
        assert case["source_ids"] and set(case["source_ids"]) <= set(
            r["source_id"] for r in corpus.chunks
        )
        if case["expected_status"] == "ok":
            assert case["required_facts"]
        for fact in case["required_facts"]:
            assert fact["quote"] in corpus.by_id[fact["primary_chunk_id"]]["text"], (
                case["id"]
            )
            assert fact["evidence_ids"], case["id"]
            assert all(
                fact["needle"] in corpus.by_id[c]["text"] for c in fact["evidence_ids"]
            ), case["id"]
            assert any(
                re.search(p, normalized(case["reference_answer"]), re.I)
                for p in fact["patterns"]
            ), (case["id"], fact["id"])
    return {
        "status": "passed",
        "cases": 60,
        "dev": 20,
        "test": 40,
        "source_groups": len({r["group_id"] for r in allrows}),
        "source_split_overlap": 0,
        "facts": sum(len(r["required_facts"]) for r in allrows),
        "categories": dict(Counter(r["category"] for r in allrows)),
        "annotation_provenance": (
            f"checked and approved by author: {approved_author()}"
            if approved_author()
            else "authored source annotations; author confirmation not recorded"
        ),
    }
