"""Summarize new paid verification/recording calls without recounting frozen runs."""

import json

from spief_rag.evaluation import read_jsonl
from spief_rag.settings import ROOT
from spief_rag.telemetry import Ledger


def main():
    """Account once for each saved generation/embedding attempt, including errors."""
    folder = ROOT / "reports/submission"
    tariff = json.loads((ROOT / "configs/tariff.json").read_text())
    stages = {}
    preflight = folder / "preflight.json"
    if preflight.exists():
        stages["preflight"] = json.loads(preflight.read_text())["events"]
    calibration = folder / "calibration.jsonl"
    if calibration.exists():
        stages["judge_calibration"] = [
            e for row in read_jsonl(calibration) for e in row["events"]
        ]
    validation = folder / "validation.jsonl"
    if validation.exists():
        rows = read_jsonl(validation)
        stages["diagnostic_questions"] = [
            e for row in rows for e in row["run"]["events"]
        ]
        stages["diagnostic_judge"] = [e for row in rows for e in row["judge_events"]]
    recording = folder / "screencast_calls.jsonl"
    if recording.exists():
        stages["screencast"] = [
            e for row in read_jsonl(recording) for e in row["events"]
        ]
    result = {
        "scope": "New submission verification and screen recording only; frozen benchmark excluded",
        "cost_label": tariff["label"],
        "cash_charge_confirmed": False,
        "stages": {
            name: Ledger(events=events).summary(tariff)
            for name, events in stages.items()
        },
        "total": Ledger(
            events=[e for events in stages.values() for e in events]
        ).summary(tariff),
        "local_speech_synthesis_api_cost_rub": 0,
    }
    (folder / "expenses.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result["total"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
