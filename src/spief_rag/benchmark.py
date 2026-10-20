"""Resumable benchmark, development selection and separate judge accounting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import pipeline as pipeline_protocol
from .client import APIError
from .corpus import Corpus, digest
from .evaluation import (
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    aggregate,
    calibration_metrics,
    judge_answer,
    read_jsonl,
    score_run,
)
from .pipeline import Pipeline
from .search import SearchEngine
from .settings import ROOT, Experiment
from .telemetry import Ledger, append_jsonl

BENCHMARK_PROTOCOL_VERSION = 2


def calibration_contract(client, corpus: Corpus) -> dict:
    """Bind calibration to its actual labels, source cases and judge wire contract."""
    return {
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "inputs": read_jsonl(ROOT / "data/gold/judge_calibration.jsonl"),
        "cases": {c["id"]: c for c in read_jsonl(ROOT / "data/gold/dev.jsonl")},
        "model": client.settings.model,
        "judge_prompt": JUDGE_SYSTEM,
        "judge_schema": JUDGE_SCHEMA,
        "corpus": corpus.version,
    }


def validated_judge_provenance(
    client, corpus: Corpus, calibration: dict | None
) -> dict:
    """Reject admission detached from the current calibration or its judge results."""
    if not calibration or not calibration.get("admitted"):
        raise ValueError("judge_calibration_required_for_admitted_judge")
    contract = calibration_contract(client, corpus)
    provenance = calibration.get("provenance", {})
    if (
        calibration.get("protocol_version") != BENCHMARK_PROTOCOL_VERSION
        or calibration.get("signature") != digest(contract)
        or not isinstance(provenance.get("judgments_hash"), str)
        or len(provenance["judgments_hash"]) != 64
    ):
        raise ValueError("judge_calibration_mismatch_run_new_calibration")
    return {
        "calibration_signature": calibration["signature"],
        "provenance": provenance,
        "admission": {
            k: calibration.get(k)
            for k in (
                "agreement",
                "kappa",
                "false_acceptance_rate",
                "cases",
                "valid_judgments",
                "expected_cases",
                "label_provenance",
            )
        },
    }


def benchmark_contract(
    client,
    engine: SearchEngine,
    config: Experiment,
    cases: list[dict],
    judge_provenance: dict | None,
) -> dict:
    """Fingerprint actual generation/search/judge contracts, beyond manual versions."""
    return {
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "config": config.to_dict(),
        "model": client.settings.model,
        "corpus": engine.corpus.version,
        "cases": cases,
        "generation": pipeline_protocol.generation_contract(config),
        "public_schema": pipeline_protocol.public_schema(),
        "search_prompt": pipeline_protocol.SEARCH_SYSTEM,
        "search_functions": pipeline_protocol.tool_specs(),
        "judge": {
            "admitted": judge_provenance is not None,
            "prompt": JUDGE_SYSTEM,
            "schema": JUDGE_SCHEMA,
            "calibration": judge_provenance,
        },
    }


def load_tariff(client, path: Path | None = None) -> dict | None:
    """Refuse a monetary estimate for a model/scope not covered by the profile."""
    tariff = json.loads((path or ROOT / "configs/tariff.json").read_text())
    if (
        tariff["model"] != client.settings.model
        or tariff["scope"] != client.settings.scope
    ):
        return None
    return tariff


def calibrate(client, corpus: Corpus, output: Path) -> dict:
    """Run the authored judge audit set once; retain costs and individual failures."""
    contract = calibration_contract(client, corpus)
    cases, inputs = contract["cases"], contract["inputs"]
    signature = digest(contract)
    metadata = output.with_suffix(".meta.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output) if output.exists() else []
    if existing and (
        not metadata.exists()
        or json.loads(metadata.read_text()).get("protocol_version")
        != BENCHMARK_PROTOCOL_VERSION
        or json.loads(metadata.read_text()).get("signature") != signature
    ):
        raise ValueError("calibration_resume_mismatch_use_new_output")
    metadata.write_text(
        json.dumps(
            {
                "signature": signature,
                "model": client.settings.model,
                "protocol_version": BENCHMARK_PROTOCOL_VERSION,
            },
            indent=2,
        )
    )
    done = {r["id"] for r in existing}
    for item in inputs:
        if item["id"] in done:
            continue
        ledger = Ledger()
        record = {k: v for k, v in item.items() if k != "evidence"}
        try:
            record["judgment"] = judge_answer(
                client, cases[item["case_id"]], item["answer"], item["evidence"], ledger
            )
        except (APIError, ValueError) as exc:
            record["judgment"] = None
            record["error"] = getattr(exc, "code", type(exc).__name__)
        record["events"], record["usage"] = (
            ledger.events,
            ledger.summary(load_tariff(client)),
        )
        append_jsonl(output, record)
        existing.append(record)
        print(
            json.dumps(
                {"calibration": item["id"], "ok": record["judgment"] is not None}
            ),
            flush=True,
        )
        if record.get("error") in {"oauth_transport_error", "oauth_error"}:
            break
    metrics = calibration_metrics(existing)
    metrics["expected_cases"] = len(inputs)
    metrics["admitted"] = metrics["admitted"] and len(existing) == len(inputs)
    metrics.update(
        protocol_version=BENCHMARK_PROTOCOL_VERSION,
        signature=signature,
        label_provenance="; ".join(
            sorted({r.get("label_provenance", "unspecified") for r in inputs})
        ),
        provenance={
            "judge_prompt_hash": digest(JUDGE_SYSTEM),
            "judge_schema_hash": digest(JUDGE_SCHEMA),
            "labels_hash": digest(inputs),
            "judgments_hash": digest(
                [
                    {
                        k: r.get(k)
                        for k in ("id", "expected_accept", "judgment", "error")
                    }
                    for r in existing
                ]
            ),
        },
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    return metrics


def evaluate(
    client,
    engine: SearchEngine,
    config: Experiment,
    *,
    split: str,
    repeats: int,
    output: Path,
    judge_admitted: bool = False,
    judge_calibration: dict | None = None,
    limit: int | None = None,
    categories: set[str] | None = None,
    dataset: Path | None = None,
) -> dict:
    """Resume only identical configurations; preserve question and judge expenses separately."""
    cases = read_jsonl(dataset or ROOT / f"data/gold/{split}.jsonl")
    if any(c["split"] != split for c in cases):
        raise ValueError("dataset_split_mismatch")
    if categories:
        cases = [r for r in cases if r["category"] in categories]
    if limit:
        cases = cases[:limit]
    tariff = load_tariff(client)
    judge_provenance = (
        validated_judge_provenance(client, engine.corpus, judge_calibration)
        if judge_admitted
        else None
    )
    contract = benchmark_contract(client, engine, config, cases, judge_provenance)
    signature = digest(contract)
    existing = read_jsonl(output) if output.exists() else []
    if any(
        r.get("benchmark_protocol_version") != BENCHMARK_PROTOCOL_VERSION
        or r.get("benchmark_signature") != signature
        for r in existing
    ):
        raise ValueError("resume_signature_mismatch_use_new_output")
    metadata = output.with_suffix(".meta.json")
    if (
        metadata.exists()
        and json.loads(metadata.read_text()).get("signature") != signature
    ):
        raise ValueError("resume_metadata_mismatch_use_new_output")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        json.dumps(
            {"signature": signature, "contract": contract}, ensure_ascii=False, indent=2
        )
    )
    done = {(r["case_id"], r["repeat"]) for r in existing}
    pipeline = Pipeline(engine, config, tariff)
    for repeat in range(repeats):
        for case in cases:
            if (case["id"], repeat) in done:
                continue
            run = pipeline.run(case["question"])
            judgment = None
            judge_ledger = Ledger()
            initial = score_run(case, run)
            if (
                judge_admitted
                and initial["deterministic_pass"]
                and case["expected_status"] == "ok"
            ):
                try:
                    judgment = judge_answer(
                        client, case, run["answer"], run["evidence"], judge_ledger
                    )
                except (APIError, ValueError):
                    judgment = None
            scored = score_run(case, run, judgment)
            record = {
                "case_id": case["id"],
                "repeat": repeat,
                "split": split,
                "benchmark_signature": signature,
                "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                "run": run,
                "score": scored,
                "judge_usage": judge_ledger.summary(tariff),
                "judge_events": judge_ledger.events,
            }
            append_jsonl(output, record)
            existing.append(record)
            print(
                json.dumps(
                    {
                        "variant": config.name,
                        "case": case["id"],
                        "repeat": repeat,
                        "tokens": scored["tokens"],
                        "deterministic": scored["deterministic_pass"],
                        "useful": scored["useful_auto"],
                        "error": run["error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if run.get("error", {}) and run["error"]["code"] in {
                "oauth_transport_error",
                "oauth_error",
            }:
                summary = {
                    "status": "blocked_by_auth_transport",
                    "completed": len(existing),
                    "expected": len(cases) * repeats,
                }
                output.with_suffix(".summary.json").write_text(
                    json.dumps(summary, indent=2)
                )
                return summary
    summary = aggregate([r["score"] for r in existing])
    summary.update(
        name=config.name,
        split=split,
        repeats=repeats,
        signature=signature,
        benchmark_protocol_version=BENCHMARK_PROTOCOL_VERSION,
        judge_provenance=judge_provenance,
        expected=len(cases) * repeats,
        complete=len(existing) == len(cases) * repeats,
        config=config.to_dict(),
        judge_admitted=judge_admitted,
        judge_total_tokens=sum(r["judge_usage"]["total_tokens"] for r in existing),
        judge_total_cost_rub=sum(
            r["judge_usage"].get("estimated_cost_rub") or 0 for r in existing
        ),
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary


def select_variant(summary_paths: list[Path], output: Path) -> dict:
    """Select only from complete development runs with full semantic evaluation."""
    summaries = [json.loads(p.read_text()) for p in summary_paths]
    if any(r.get("split") != "dev" for r in summaries):
        raise ValueError("selection_requires_development_split")
    baseline = next(r for r in summaries if r["name"] == "baseline")
    if not baseline.get("complete") or baseline.get("useful_rate") is None:
        raise ValueError("baseline_not_fully_evaluated")
    eligible = [
        r
        for r in summaries
        if r.get("complete")
        and r.get("useful_rate") is not None
        and not r.get("cost_is_lower_bound", False)
        and r["useful_rate"] >= baseline["useful_rate"]
        and r["false_answers_on_unanswerable"]
        <= baseline["false_answers_on_unanswerable"]
    ]
    if not eligible:
        raise ValueError("no_variant_with_complete_usage_and_quality")
    chosen = min(eligible, key=lambda r: r["mean_tokens"])
    decision = {
        "selected": chosen["name"],
        "selection_split": "dev",
        "baseline": baseline,
        "selected_summary": chosen,
        "config": chosen["config"],
        "quality_guard": "useful_rate>=baseline; no additional false answers on unanswerable cases",
        "no_improvement": chosen["name"] == "baseline",
    }
    output.write_text(json.dumps(decision, ensure_ascii=False, indent=2))
    config_path = output.with_name("selected_config.json")
    config_path.write_text(json.dumps(chosen["config"], ensure_ascii=False, indent=2))
    return decision


def paired_comparison(baseline_path: Path, candidate_path: Path) -> dict:
    """Describe each regression on the same case/repeat pairs, without hiding it in a mean."""
    a = {(r["case_id"], r["repeat"]): r for r in read_jsonl(baseline_path)}
    b = {(r["case_id"], r["repeat"]): r for r in read_jsonl(candidate_path)}
    if set(a) != set(b):
        raise ValueError("comparison_case_repeat_mismatch")
    if any(
        a[k]["run"]["corpus_version"] != b[k]["run"]["corpus_version"]
        or a[k]["run"]["model"] != b[k]["run"]["model"]
        for k in a
    ):
        raise ValueError("comparison_model_or_corpus_mismatch")
    regressions = []
    improvements = []
    for key in a:
        av, bv = a[key]["score"]["useful_auto"], b[key]["score"]["useful_auto"]
        item = {
            "case_id": key[0],
            "repeat": key[1],
            "baseline_answer": a[key]["run"]["answer"],
            "candidate_answer": b[key]["run"]["answer"],
            "candidate_error": b[key]["run"]["error"],
        }
        if av is True and bv is False:
            regressions.append(item)
        if av is False and bv is True:
            improvements.append(item)
    sa, sb = (
        aggregate([r["score"] for r in a.values()]),
        aggregate([r["score"] for r in b.values()]),
    )
    # Repeated generations of one question are not independent observations.
    # Bootstrap whole questions, keeping all repeat pairs inside each cluster.
    deltas = []
    for case_id in sorted({key[0] for key in a}):
        values = [
            (a[k]["score"]["useful_auto"], b[k]["score"]["useful_auto"])
            for k in a
            if k[0] == case_id
        ]
        if any(x is None or y is None for x, y in values):
            deltas = []
            break
        deltas.append(sum(int(y) - int(x) for x, y in values) / len(values))
    interval = None
    if deltas:
        rng = np.random.default_rng(20261020)
        samples = np.asarray(deltas)[
            rng.integers(0, len(deltas), size=(2000, len(deltas)))
        ].mean(axis=1)
        interval = [float(x) for x in np.quantile(samples, [0.025, 0.975])]
    cost_ok = (
        sa["cost_per_useful_rub"] is not None
        and sb["cost_per_useful_rub"] is not None
        and sb["cost_per_useful_rub"] < sa["cost_per_useful_rub"]
    )
    return {
        "baseline": sa,
        "candidate": sb,
        "regressions": regressions,
        "improvements": improvements,
        "useful_rate_delta_case_cluster_bootstrap_95": interval,
        "bootstrap_unit": "question, preserving repeated generations; 2000 resamples, seed 20261020",
        "token_reduction": 1 - sb["total_tokens"] / sa["total_tokens"]
        if sa["total_tokens"]
        else None,
        "accept_optimization": bool(
            sa["useful_rate"] is not None
            and sb["useful_rate"] is not None
            and not sb["cost_is_lower_bound"]
            and sb["useful_rate"] >= sa["useful_rate"]
            and sb["total_tokens"] < sa["total_tokens"]
            and cost_ok
            and sb["false_answers_on_unanswerable"]
            <= sa["false_answers_on_unanswerable"]
        ),
        "cost_comparison_rule": "Candidate usage must be complete. A lower bound on baseline cost is sufficient when candidate cost is already below that lower bound.",
    }
