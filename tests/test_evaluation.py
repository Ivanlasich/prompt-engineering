"""Checks prevent unsupported economic or quality conclusions."""

import json

import pytest

from spief_rag.benchmark import select_variant
from spief_rag.corpus import Corpus
from spief_rag.evaluation import aggregate, calibration_metrics, score_run
from spief_rag.validation import validate_gold


def test_real_gold_source_spans_and_group_isolation():
    assert validate_gold(Corpus.load())["cases"] == 60


def case_run():
    case = {
        "id": "a",
        "category": "fact",
        "expected_status": "ok",
        "required_facts": [{"patterns": ["альфа"], "evidence_ids": ["chunk"]}],
        "forbidden_patterns": [],
    }
    run = {
        "run_id": "test",
        "answer": {
            "status": "ok",
            "answer": "Альфа",
            "citations": [{"chunk_id": "chunk", "quote": "Организация Альфа"}],
        },
        "evidence": [{"chunk_id": "chunk", "text": "Организация Альфа"}],
        "trace": [
            {"type": "retrieval", "raw_hit_ids": ["chunk"], "context_ids": ["chunk"]}
        ],
        "usage": {
            "total_tokens": 100,
            "estimated_cost_rub": 1,
            "unknown_usage_attempts": 0,
        },
        "error": None,
    }
    return case, run


def test_validity_is_not_semantic_usefulness():
    c, r = case_run()
    score = score_run(c, r)
    assert score["schema_valid"] and score["deterministic_pass"]
    assert score["useful_auto"] is None
    result = aggregate([score])
    assert result["useful_rate"] is None
    assert result["cost_per_useful_rub"] is None


def test_cost_per_useful_includes_failures():
    c, r = case_run()
    good = score_run(c, r, {"supported": True, "complete": True})
    r = {**r, "answer": None, "error": {"code": "failure"}}
    bad = score_run(c, r)
    result = aggregate([good, bad])
    assert result["useful_rate"] == 0.5
    assert result["cost_per_useful_rub"] == 2
    assert result["total_tokens"] == 200


def test_judge_rejects_unsupported_answer():
    c, r = case_run()
    score = score_run(c, r, {"supported": False, "complete": True})
    assert score["useful_auto"] is False


def test_calibration_incomplete_is_not_admitted():
    assert not calibration_metrics([])["admitted"]


def test_selection_rejects_test_tuning(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"split": "test", "name": "baseline"}))
    with pytest.raises(ValueError, match="development_split"):
        select_variant([p], tmp_path / "selected.json")


def test_precached_tokens_are_not_subtracted_twice():
    from spief_rag.telemetry import Ledger

    ledger = Ledger()
    ledger.record(
        "generation",
        "test",
        {
            "prompt_tokens": 380,
            "completion_tokens": 110,
            "total_tokens": 490,
            "precached_prompt_tokens": 1020,
        },
        1,
    )
    tariff = {
        "generation_rub_per_1000": 1,
        "embedding_rub_per_1000": 0,
        "label": "test",
    }
    assert ledger.summary(tariff)["estimated_cost_rub"] == 0.49
    assert ledger.summary(tariff)["logical_tokens_including_cached"] == 1510


def test_supplemental_multihop_really_needs_several_chunks():
    from spief_rag.evaluation import read_jsonl
    from spief_rag.settings import ROOT

    corpus = Corpus.load()
    dev_sources = {r["group_id"] for r in read_jsonl(ROOT / "data/gold/dev.jsonl")}
    for case in read_jsonl(ROOT / "data/gold/multihop.jsonl"):
        assert case["group_id"] in dev_sources
        groups = [set(f["evidence_ids"]) for f in case["required_facts"]]
        assert not set.intersection(*groups)
        for fact in case["required_facts"]:
            assert fact["quote"] in corpus.by_id[fact["primary_chunk_id"]]["text"]


def test_known_candidate_can_beat_even_baseline_lower_bound(tmp_path):
    from copy import deepcopy

    from spief_rag.benchmark import paired_comparison

    c, run = case_run()
    run.update(corpus_version="v", model="m")
    run["usage"]["unknown_usage_attempts"] = 1
    a = {
        "case_id": "a",
        "repeat": 0,
        "run": run,
        "score": score_run(c, run, {"supported": True, "complete": True}),
    }
    br = deepcopy(run)
    br["usage"].update(
        unknown_usage_attempts=0, total_tokens=50, estimated_cost_rub=0.5
    )
    b = {
        "case_id": "a",
        "repeat": 0,
        "run": br,
        "score": score_run(c, br, {"supported": True, "complete": True}),
    }
    pa = tmp_path / "a.jsonl"
    pb = tmp_path / "b.jsonl"
    pa.write_text(json.dumps(a) + "\n")
    pb.write_text(json.dumps(b) + "\n")
    result = paired_comparison(pa, pb)
    assert result["baseline"]["cost_is_lower_bound"]
    assert result["accept_optimization"]
    b["score"]["usage_complete"] = False
    pb.write_text(json.dumps(b) + "\n")
    assert not paired_comparison(pa, pb)["accept_optimization"]
