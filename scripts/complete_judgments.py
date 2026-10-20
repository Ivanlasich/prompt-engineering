"""Complete missing semantic audits without regenerating or dropping paid answers."""

import argparse
import json
from pathlib import Path

from spief_rag.benchmark import load_tariff
from spief_rag.client import APIError, GigaChatClient
from spief_rag.evaluation import aggregate, judge_answer, read_jsonl, score_run
from spief_rag.settings import ROOT, Settings
from spief_rag.telemetry import Ledger


def main():
    """Retain prior judge attempts and explicitly report any still unknown result."""
    p = argparse.ArgumentParser()
    p.add_argument("runs", type=Path, nargs="+")
    a = p.parse_args()
    client = GigaChatClient(Settings.from_env())
    try:
        tariff = load_tariff(client)
        cases = {
            r["id"]: r
            for split in ["dev", "test"]
            for r in read_jsonl(ROOT / f"data/gold/{split}.jsonl")
        }
        extra = ROOT / "data/gold/multihop.jsonl"
        if extra.exists():
            cases.update({r["id"]: r for r in read_jsonl(extra)})
        for path in a.runs:
            rows = read_jsonl(path)
            for row in rows:
                if row["score"]["useful_auto"] is not None:
                    continue
                ledger = Ledger()
                try:
                    judgment = judge_answer(
                        client,
                        cases[row["case_id"]],
                        row["run"]["answer"],
                        row["run"]["evidence"],
                        ledger,
                    )
                    row["score"] = score_run(
                        cases[row["case_id"]], row["run"], judgment
                    )
                except (APIError, ValueError):
                    pass
                row["judge_events"].extend(ledger.events)
                all_judge = Ledger(events=row["judge_events"])
                row["judge_usage"] = all_judge.summary(tariff)
                print(row["case_id"], row["score"]["useful_auto"], flush=True)
                # Atomic rewrite retains all previous answer and judge expenses.
                temp = path.with_suffix(".partial")
                temp.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
                )
                temp.replace(path)
            summary_path = path.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text())
            summary.update(
                aggregate([r["score"] for r in rows]),
                judge_total_tokens=sum(r["judge_usage"]["total_tokens"] for r in rows),
                judge_total_cost_rub=sum(
                    r["judge_usage"].get("estimated_cost_rub") or 0 for r in rows
                ),
            )
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
