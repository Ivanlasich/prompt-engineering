"""Run a separate paid diagnostic slice after fixes, preserving the frozen benchmark.

This is a diagnostic recheck, never a replacement for the 40-question test with
three repeats. It includes all eight questions that regressed and four controls.
"""

import json

from spief_rag.benchmark import calibrate, evaluate
from spief_rag.client import GigaChatClient
from spief_rag.corpus import Corpus
from spief_rag.evaluation import read_jsonl
from spief_rag.search import SearchEngine
from spief_rag.settings import ROOT, Settings, get_experiment


def main():
    """Recalibrate under protocol v2 and record selected-model diagnostic answers."""
    output = ROOT / "reports/submission"
    output.mkdir(parents=True, exist_ok=True)
    wanted = {
        "test-01",
        "test-02",
        "test-06",
        "test-10",
        "test-12",
        "test-17",
        "test-28",
        "test-33",
        "test-09",
        "test-13",
        "test-27",
        "test-31",
    }
    cases = [c for c in read_jsonl(ROOT / "data/gold/test.jsonl") if c["id"] in wanted]
    dataset = output / "diagnostic_cases.jsonl"
    dataset.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    client = GigaChatClient(Settings.from_env())
    try:
        corpus = Corpus.load()
        calibration = calibrate(client, corpus, output / "calibration.jsonl")
        if not calibration["admitted"]:
            raise ValueError("new_calibration_not_admitted")
        summary = evaluate(
            client,
            SearchEngine(corpus, client),
            get_experiment("short_prompt", str(ROOT / "configs/selected_config.json")),
            split="test",
            repeats=1,
            output=output / "validation.jsonl",
            judge_admitted=True,
            judge_calibration=calibration,
            dataset=dataset,
        )
        summary["purpose"] = (
            "Diagnostic recheck after logging/protocol fixes; not an independent final test or new optimization"
        )
        (output / "validation.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
