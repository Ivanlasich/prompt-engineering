"""A resumed run must preserve actual prompts, schemas and judge provenance."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from spief_rag import benchmark
from spief_rag.corpus import digest
from spief_rag.settings import BASELINE


@pytest.fixture
def benchmark_setup(tmp_path, monkeypatch):
    gold = tmp_path / "data/gold"
    gold.mkdir(parents=True)
    case = {
        "id": "case-1",
        "split": "dev",
        "category": "not_found",
        "question": "Есть ли отсутствующий факт?",
        "expected_status": "not_found",
        "required_facts": [],
        "reference_answer": "Нет сведений.",
    }
    dataset = gold / "dev.jsonl"
    dataset.write_text(json.dumps(case) + "\n")
    calibration = {
        "id": "cal-1",
        "case_id": case["id"],
        "answer": {"status": "not_found", "answer": "Нет сведений.", "citations": []},
        "evidence": [],
        "expected_accept": True,
        "label_provenance": "human_reviewed",
    }
    (gold / "judge_calibration.jsonl").write_text(json.dumps(calibration) + "\n")
    calls = []

    class OfflinePipeline:
        """Return one fixed valid refusal; a resume must not execute it again."""

        def __init__(self, *args):
            """Accept the normal pipeline constructor without a provider."""

        def run(self, question):
            """Record execution so tests distinguish resume from regeneration."""
            calls.append(question)
            return {
                "run_id": "offline-run",
                "answer": calibration["answer"],
                "error": None,
                "usage": {"total_tokens": 10, "estimated_cost_rub": 0.1},
            }

    class OfflineJudge:
        """Serve the schema contract without HTTP calls or credentials."""

        settings = SimpleNamespace(model="test-model")

        def chat(self, messages, ledger, **kwargs):
            """Accept the one explicitly positive calibration fixture."""
            return {
                "message": {
                    "content": json.dumps(
                        {"supported": True, "complete": True, "reason": "Проверено."}
                    )
                }
            }

    client = OfflineJudge()
    engine = SimpleNamespace(corpus=SimpleNamespace(version="corpus-v1"), client=client)
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "Pipeline", OfflinePipeline)
    monkeypatch.setattr(benchmark, "load_tariff", lambda client: None)
    return SimpleNamespace(
        root=tmp_path,
        dataset=dataset,
        output=tmp_path / "run.jsonl",
        client=client,
        engine=engine,
        calls=calls,
    )


def run_fixture(setup, **kwargs):
    """Run one tiny local benchmark using the production resume implementation."""
    return benchmark.evaluate(
        setup.client,
        setup.engine,
        BASELINE,
        split="dev",
        repeats=1,
        output=setup.output,
        dataset=setup.dataset,
        **kwargs,
    )


def test_identical_protocol_resumes_without_new_calls(benchmark_setup):
    setup = benchmark_setup
    first = run_fixture(setup)
    second = run_fixture(setup)
    assert first == second
    assert setup.calls == ["Есть ли отсутствующий факт?"]
    metadata = json.loads(setup.output.with_suffix(".meta.json").read_text())
    assert metadata["contract"]["protocol_version"] == 2
    assert metadata["contract"]["generation"]["system"]
    assert metadata["contract"]["search_functions"]


@pytest.mark.parametrize(
    "changed",
    [
        "answer",
        "search",
        "answer_schema",
        "search_schema",
        "judge_prompt",
        "judge_schema",
    ],
)
def test_actual_contract_change_cannot_append_to_old_run(
    benchmark_setup, monkeypatch, changed
):
    setup = benchmark_setup
    run_fixture(setup)
    before = setup.output.read_bytes()
    if changed == "answer":
        monkeypatch.setattr(
            benchmark.pipeline_protocol,
            "answer_system",
            lambda config: "A changed answer prompt",
        )
    elif changed == "search":
        monkeypatch.setattr(
            benchmark.pipeline_protocol, "SEARCH_SYSTEM", "A changed search prompt"
        )
    elif changed == "answer_schema":
        original = benchmark.pipeline_protocol.public_schema()
        monkeypatch.setattr(
            benchmark.pipeline_protocol,
            "public_schema",
            lambda: {**deepcopy(original), "description": "Changed schema"},
        )
    elif changed == "search_schema":
        original = benchmark.pipeline_protocol.tool_specs()
        monkeypatch.setattr(
            benchmark.pipeline_protocol,
            "tool_specs",
            lambda: [*deepcopy(original), {"name": "another_tool"}],
        )
    elif changed == "judge_prompt":
        monkeypatch.setattr(benchmark, "JUDGE_SYSTEM", "A changed judge prompt")
    else:
        monkeypatch.setattr(
            benchmark,
            "JUDGE_SCHEMA",
            {**benchmark.JUDGE_SCHEMA, "description": "Changed judge schema"},
        )
    with pytest.raises(ValueError, match="resume_signature_mismatch"):
        run_fixture(setup)
    assert setup.output.read_bytes() == before
    assert len(setup.calls) == 1


def test_legacy_run_is_refused_without_rewriting_it(benchmark_setup):
    setup = benchmark_setup
    run_fixture(setup)
    row = json.loads(setup.output.read_text())
    row.pop("benchmark_protocol_version")
    setup.output.write_text(json.dumps(row) + "\n")
    before = setup.output.read_bytes()
    with pytest.raises(ValueError, match="resume_signature_mismatch"):
        run_fixture(setup)
    assert setup.output.read_bytes() == before


def test_judge_admission_requires_current_calibration_and_results(benchmark_setup):
    setup = benchmark_setup
    with pytest.raises(ValueError, match="judge_calibration_required"):
        run_fixture(setup, judge_admitted=True)
    calibration = benchmark.calibrate(
        setup.client, setup.engine.corpus, setup.root / "calibration.jsonl"
    )
    assert calibration["admitted"]
    assert calibration["label_provenance"] == "human_reviewed"
    assert calibration["signature"] == digest(
        benchmark.calibration_contract(setup.client, setup.engine.corpus)
    )
    run_fixture(setup, judge_admitted=True, judge_calibration=calibration)
    changed_results = deepcopy(calibration)
    changed_results["provenance"]["judgments_hash"] = "a" * 64
    with pytest.raises(ValueError, match="resume_signature_mismatch"):
        run_fixture(setup, judge_admitted=True, judge_calibration=changed_results)
    labels = setup.root / "data/gold/judge_calibration.jsonl"
    row = json.loads(labels.read_text())
    row["expected_accept"] = False
    labels.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="judge_calibration_mismatch"):
        run_fixture(setup, judge_admitted=True, judge_calibration=calibration)


def test_calibration_refuses_legacy_metadata(benchmark_setup):
    setup = benchmark_setup
    path = setup.root / "calibration.jsonl"
    benchmark.calibrate(setup.client, setup.engine.corpus, path)
    metadata = path.with_suffix(".meta.json")
    old = json.loads(metadata.read_text())
    old.pop("protocol_version")
    metadata.write_text(json.dumps(old))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="calibration_resume_mismatch"):
        benchmark.calibrate(setup.client, setup.engine.corpus, path)
    assert path.read_bytes() == before
