"""Export blind review sheets as JSONL and import explicit human judgments."""

import argparse
import json
from pathlib import Path

from spief_rag.evaluation import aggregate, read_jsonl
from spief_rag.settings import ROOT


def main():
    """Require named reviewers and complete booleans; never auto-fill human labels."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=[
            "export-calibration",
            "import-calibration",
            "export-runs",
            "import-runs",
        ],
    )
    p.add_argument("review", type=Path)
    p.add_argument("--runs", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    calibration = ROOT / "data/gold/judge_calibration.jsonl"
    path = calibration if "calibration" in a.action else a.runs
    if path is None:
        p.error("--runs is required for run review")
    rows = read_jsonl(path)
    if a.action.startswith("export"):
        cases = (
            {case["id"]: case for case in read_jsonl(ROOT / "data/gold/dev.jsonl")}
            if "calibration" in a.action
            else {}
        )
        out = []
        for row in rows:
            if "calibration" in a.action:
                value = {k: row[k] for k in ["id", "case_id", "answer", "evidence"]}
                case = cases[row["case_id"]]
                value.update(
                    {
                        k: case[k]
                        for k in [
                            "question",
                            "reference_answer",
                            "expected_status",
                            "required_facts",
                            "forbidden_patterns",
                        ]
                    }
                )
            else:
                value = {
                    "id": row["run"]["run_id"],
                    "case_id": row["case_id"],
                    "question": row["run"]["question"],
                    "answer": row["run"]["answer"],
                    "evidence": row["run"].get("evidence", []),
                }
            value.update(human_accept=None, reviewer=None, rationale=None)
            out.append(value)
        a.review.parent.mkdir(parents=True, exist_ok=True)
        a.review.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
        )
    else:
        reviews = read_jsonl(a.review)
        by_id = {r["id"]: r for r in reviews}
        ids = {
            r["id"] if "calibration" in a.action else r["run"]["run_id"] for r in rows
        }
        if set(by_id) != ids or len(by_id) != len(reviews):
            raise ValueError("review_ids_mismatch")
        for row in rows:
            key = row["id"] if "calibration" in a.action else row["run"]["run_id"]
            review = by_id[key]
            if (
                type(review.get("human_accept")) is not bool
                or not str(review.get("reviewer") or "").strip()
                or not str(review.get("rationale") or "").strip()
            ):
                raise ValueError("explicit_boolean_reviewer_and_rationale_required")
            if "calibration" in a.action:
                row.update(
                    expected_accept=review["human_accept"],
                    rationale=review["rationale"],
                    label_provenance="human_reviewed",
                    reviewer=review["reviewer"],
                )
            else:
                if review["human_accept"] and not row["score"]["deterministic_pass"]:
                    raise ValueError(
                        "cannot_accept_invalid_structure_or_missing_required_facts"
                    )
                row["score"]["useful_auto"] = review["human_accept"]
                row["score"]["human_review"] = {
                    k: review[k] for k in ["reviewer", "rationale"]
                }
        output = a.output or path
        output.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        if "runs" in a.action:
            result = aggregate([r["score"] for r in rows])
            result["evaluation_mode"] = "human_reviewed"
            output.with_suffix(".summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
    print("Completed", a.action)


if __name__ == "__main__":
    main()
