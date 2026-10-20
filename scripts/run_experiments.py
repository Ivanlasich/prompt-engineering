"""Execute the entire paid experiment protocol explicitly and resumably.

Development selection is completed before any test run. Result caching is disabled.
Credentials come from the environment/.env.local and are never CLI arguments.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from spief_rag.benchmark import calibrate, evaluate, paired_comparison, select_variant
from spief_rag.client import APIError, GigaChatClient
from spief_rag.corpus import Corpus
from spief_rag.report import write_report
from spief_rag.search import SearchEngine
from spief_rag.settings import ROOT, VARIANTS, Experiment, Settings


def main():
    """Stop on missing capabilities/evaluation instead of manufacturing a winner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "reports/runs/protocol_v2"
    )
    args = parser.parse_args()
    runs = args.run_dir.resolve()
    if runs == (ROOT / "reports/runs").resolve():
        parser.error("Legacy results are frozen; choose a new --run-dir")
    runs.mkdir(parents=True, exist_ok=True)
    report = runs / "report"
    report.mkdir(parents=True, exist_ok=True)
    client = GigaChatClient(Settings.from_env())
    try:
        preflight = report / "preflight.json"
        try:
            capabilities = client.preflight()
            preflight.write_text(json.dumps(capabilities, ensure_ascii=False, indent=2))
        except APIError as exc:
            preflight.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "error": exc.code,
                        "http_status": exc.status,
                        "api_base": client.settings.api_base,
                        "scope": client.settings.scope,
                        "tls_verification": True,
                    },
                    indent=2,
                )
            )
            print("Preflight failed:", exc.code, flush=True)
            return
        corpus = Corpus.load()
        engine = SearchEngine(corpus, client)
        calibration = calibrate(client, corpus, runs / "calibration.jsonl")
        if not calibration["admitted"]:
            print(
                "Judge not admitted; review calibration labels/results before continuing.",
                flush=True,
            )
            return
        summaries = []
        # One factor changes at each step; all model and dataset settings stay fixed.
        for name in ["baseline", "context", "short_prompt", "one_example", "compact"]:
            path = runs / f"dev_{name}.jsonl"
            evaluate(
                client,
                engine,
                VARIANTS[name],
                split="dev",
                repeats=1,
                output=path,
                judge_admitted=True,
                judge_calibration=calibration,
            )
            summaries.append(path.with_suffix(".summary.json"))
        decision = select_variant(summaries, runs / "selection.json")
        selected = Experiment(**decision["config"])
        for name, config in [
            ("baseline", VARIANTS["baseline"]),
            ("selected", selected),
        ]:
            evaluate(
                client,
                engine,
                config,
                split="test",
                repeats=3,
                output=runs / f"test_{name}.jsonl",
                judge_admitted=True,
                judge_calibration=calibration,
            )
        comparison = paired_comparison(
            runs / "test_baseline.jsonl", runs / "test_selected.jsonl"
        )
        (report / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2)
        )
        for name in ["self_consistency", "react"]:
            config = replace(
                selected,
                name=name,
                candidates=3 if name == "self_consistency" else 1,
                max_search_calls=3 if name == "react" else 1,
            )
            evaluate(
                client,
                engine,
                config,
                split="dev",
                repeats=1,
                output=runs / f"extra_{name}.jsonl",
                judge_admitted=True,
                judge_calibration=calibration,
                categories={"multi"},
            )
        # Matching control subset is needed for a fair comparison of extra techniques.
        evaluate(
            client,
            engine,
            replace(selected, name="extra_control"),
            split="dev",
            repeats=1,
            output=runs / "extra_control.jsonl",
            judge_admitted=True,
            judge_calibration=calibration,
            categories={"multi"},
        )
        for name in ["self_consistency", "react"]:
            result = paired_comparison(
                runs / "extra_control.jsonl", runs / f"extra_{name}.jsonl"
            )
            (report / f"comparison_{name}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
        challenge = ROOT / "data/gold/multihop.jsonl"
        if challenge.exists():
            for name, search_calls in [("multihop_control", 1), ("multihop_react", 3)]:
                config = replace(selected, name=name, max_search_calls=search_calls)
                evaluate(
                    client,
                    engine,
                    config,
                    split="dev",
                    repeats=1,
                    output=runs / f"extra_{name}.jsonl",
                    judge_admitted=True,
                    judge_calibration=calibration,
                    dataset=challenge,
                )
            result = paired_comparison(
                runs / "extra_multihop_control.jsonl",
                runs / "extra_multihop_react.jsonl",
            )
            (report / "comparison_multihop.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
    finally:
        client.close()
        print(write_report(runs, report), flush=True)


if __name__ == "__main__":
    main()
