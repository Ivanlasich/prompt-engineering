"""Command-line entrypoints for data, answering, evaluation and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import (
    calibrate,
    evaluate,
    load_tariff,
    paired_comparison,
    select_variant,
)
from .cache import ExactCache
from .client import APIError, GigaChatClient
from .corpus import Corpus, rebuild_index
from .pipeline import Pipeline
from .search import SearchEngine
from .settings import ROOT, VARIANTS, Settings, get_experiment


def main() -> None:
    """Expose explicit commands; no API call occurs on import or notebook opening."""
    parser = argparse.ArgumentParser(description="ПМЭФ RAG на GigaChat")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, default=ROOT / "data/index")
    p = sub.add_parser("inspect")
    p = sub.add_parser("preflight")
    p.add_argument("--output", type=Path, default=ROOT / "reports/preflight.json")
    p = sub.add_parser("build-index")
    p.add_argument("--source", type=Path, default=ROOT / "data/index/chunks.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument(
        "--variant",
        choices=VARIANTS,
        help="Explicit experiment variant; otherwise use configs/selected_config.json",
    )
    p.add_argument(
        "--config", help="Override the selected or explicitly named configuration"
    )
    p.add_argument("--cache", action="store_true")
    p.add_argument("--output", type=Path)
    p = sub.add_parser("calibrate")
    p.add_argument(
        "--output", type=Path, default=ROOT / "reports/runs/calibration.jsonl"
    )
    p = sub.add_parser("evaluate")
    p.add_argument("--variant", choices=VARIANTS, default="baseline")
    p.add_argument("--config")
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--categories")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "reports/runs/calibration.summary.json",
    )
    p = sub.add_parser("select")
    p.add_argument("summaries", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "configs/selection.json")
    p = sub.add_parser("compare")
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "reports/comparison.json")
    p = sub.add_parser("report")
    p.add_argument("--runs", type=Path, default=ROOT / "reports/runs")
    p.add_argument("--output", type=Path, default=ROOT / "reports")
    p = sub.add_parser("validate-gold")
    args = parser.parse_args()
    client = None
    try:
        if args.command == "report":
            from .report import write_report

            print(write_report(args.runs, args.output))
            return
        if args.command == "compare":
            result = paired_comparison(args.baseline, args.candidate)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "select":
            result = select_variant(args.summaries, args.output)
        else:
            corpus = Corpus.load()
            if args.command == "prepare":
                corpus.prepare(args.output)
                result = {"prepared": str(args.output), "chunks": len(corpus.chunks)}
            elif args.command == "inspect":
                result = {
                    "stats": corpus.stats,
                    "corpus_version": corpus.version,
                    "actual_max_chars": max(len(r["text"]) for r in corpus.chunks),
                }
            elif args.command == "validate-gold":
                from .validation import validate_gold

                result = validate_gold(corpus)
            else:
                client = GigaChatClient(Settings.from_env())
                if args.command == "preflight":
                    try:
                        result = client.preflight()
                    except APIError as exc:
                        result = {
                            "status": "failed",
                            "error": exc.code,
                            "http_status": exc.status,
                            "tls_verification": True,
                        }
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2)
                    )
                elif args.command == "build-index":
                    result = rebuild_index(
                        client, args.source, args.output, limit=args.limit
                    )
                elif args.command == "calibrate":
                    result = calibrate(client, corpus, args.output)
                elif args.command == "ask":
                    config_path = args.config
                    if args.variant is None and config_path is None:
                        config_path = str(ROOT / "configs/selected_config.json")
                    pipeline = Pipeline(
                        SearchEngine(corpus, client),
                        get_experiment(args.variant or "short_prompt", config_path),
                        load_tariff(client),
                        ExactCache(ROOT / ".cache/answers") if args.cache else None,
                    )
                    result = pipeline.run(args.question)
                    if args.output:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(
                            json.dumps(result, ensure_ascii=False, indent=2)
                        )
                elif args.command == "evaluate":
                    calibration_data = (
                        json.loads(args.calibration.read_text())
                        if args.calibration.exists()
                        else None
                    )
                    admitted = bool(
                        calibration_data and calibration_data.get("admitted", False)
                    )
                    result = evaluate(
                        client,
                        SearchEngine(corpus, client),
                        get_experiment(args.variant, args.config),
                        split=args.split,
                        repeats=args.repeats,
                        output=args.output,
                        judge_admitted=admitted,
                        judge_calibration=calibration_data,
                        limit=args.limit,
                        categories=set(args.categories.split(","))
                        if args.categories
                        else None,
                    )
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if isinstance(result, dict) and (
            result.get("error") or result.get("status") == "failed"
        ):
            sys.exit(1)
    except (APIError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "error": getattr(exc, "code", type(exc).__name__),
                    "message": str(exc)[:150],
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
