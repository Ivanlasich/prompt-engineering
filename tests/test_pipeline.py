"""Behavior tests for grounding, budgets, search, caching and consensus."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from spief_rag.cache import ExactCache
from spief_rag.client import APIError
from spief_rag.contracts import Answer, validate_grounding
from spief_rag.corpus import Corpus
from spief_rag.pipeline import Pipeline, fact_fingerprint
from spief_rag.search import SearchEngine, select_context
from spief_rag.settings import BASELINE
from spief_rag.telemetry import Ledger


@pytest.fixture
def corpus():
    chunks = [
        {
            "chunk_id": "one",
            "source_id": "source",
            "session": "ПМЭФ-2016 it",
            "time_start": "00:01:00",
            "speakers": ["Иван"],
            "text": "Иван представляет компанию Альфа.",
        },
        {
            "chunk_id": "two",
            "source_id": "other",
            "session": "ПМЭФ-2014 oil",
            "time_start": "00:00:00",
            "speakers": ["Анна"],
            "text": "Анна представляет компанию Бета.",
        },
    ]
    return Corpus(
        chunks, np.array([[1.0, 0.0], [0.0, 1.0]]), {"embedding_model": "test"}
    )


class FakeClient:
    """Scripted provider that accounts successful calls before simulating failure."""

    settings = SimpleNamespace(model="test-model")

    def __init__(self, tool="keyword_search", fail=False, invalid=False):
        self.calls = 0
        self.tool = tool
        self.fail = fail
        self.invalid = invalid

    def embed(self, texts, ledger):
        ledger.record("embedding", "test", {"total_tokens": 3}, 1)
        return [[1.0, 0.0]]

    def chat(self, messages, ledger, **kwargs):
        ledger.check({"messages": messages}, kwargs.get("max_tokens", 0))
        self.calls += 1
        ledger.record(
            "generation",
            "test",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            1,
        )
        if kwargs.get("functions"):
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "functions_state_id": "s",
                    "function_call": {
                        "name": self.tool,
                        "arguments": {"query": "Иван", "year": 2016},
                    },
                }
            }
        if self.fail:
            raise APIError("after_paid_call")
        answer = {
            "status": "ok",
            "answer": "Иван представляет Альфу.",
            "citations": [
                {
                    "chunk_id": "missing" if self.invalid else "one",
                    "quote": "Иван представляет компанию Альфа.",
                }
            ],
        }
        return {
            "message": {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False),
            }
        }


@pytest.mark.parametrize("tool", ["semantic_search", "keyword_search"])
def test_both_real_tools(corpus, tool):
    client = FakeClient(tool=tool)
    result = Pipeline(SearchEngine(corpus, client), BASELINE).run(
        "Кого представляет Иван в 2016 году?"
    )
    assert not result["error"]
    assert result["answer"]["status"] == "ok"
    assert result["sources"][0]["time_start"] == "00:01:00"
    assert result["usage"]["total_tokens"] == 30 + (
        3 if tool == "semantic_search" else 0
    )


def test_partial_usage_survives_failure(corpus):
    result = Pipeline(SearchEngine(corpus, FakeClient(fail=True)), BASELINE).run("Иван")
    assert result["error"]["code"] == "after_paid_call"
    assert result["answer"] is None
    assert result["usage"]["total_tokens"] == 30
    assert len(result["trace"]) >= 2


def test_invalid_quote_rejected(corpus):
    answer = Answer.model_validate(
        {
            "status": "ok",
            "answer": "x",
            "citations": [{"chunk_id": "one", "quote": "выдуманное утверждение"}],
        }
    )
    with pytest.raises(ValueError, match="quote_not_in_context"):
        validate_grounding(answer, corpus.by_id)
    result = Pipeline(SearchEngine(corpus, FakeClient(invalid=True)), BASELINE).run(
        "Иван"
    )
    assert result["answer"] is None
    assert result["usage"]["total_tokens"] == 45
    failures = [t for t in result["trace"] if t["type"] == "validation_error"]
    assert len(failures) == 2
    assert failures[0]["reasons"] == [
        {"type": "ValueError", "message": "citation_not_retrieved"}
    ]
    assert (
        json.loads(failures[0]["rejected_public_content"])["citations"][0]["chunk_id"]
        == "missing"
    )


def test_rejected_candidate_diagnostic_excludes_reasoning_and_credentials(corpus):
    class SecretClient(FakeClient):
        _credentials = "unit-test-authorization-secret"
        _token = "unit-test-access-secret"

        def chat(self, messages, ledger, **kwargs):
            response = super().chat(messages, ledger, **kwargs)
            if kwargs.get("schema"):
                payload = json.loads(response["message"]["content"])
                payload["answer"] = (
                    f"Public answer {self._credentials} Bearer {self._token}"
                )
                payload["reasoning_content"] = "PRIVATE-REASONING-NEVER-LOG"
                response["message"]["reasoning_content"] = "PRIVATE-REASONING-NEVER-LOG"
                response["message"]["content"] = json.dumps(payload)
            return response

    result = Pipeline(SearchEngine(corpus, SecretClient()), BASELINE).run("Иван")
    failures = [t for t in result["trace"] if t["type"] == "validation_error"]
    assert failures[0]["error_type"] == "ValidationError"
    assert failures[0]["reasons"][0]["type"] == "extra_forbidden"
    public = json.loads(failures[0]["rejected_public_content"])
    assert set(public) == {"status", "answer", "citations"}
    assert "[REDACTED]" in public["answer"]
    serialized = json.dumps(result)
    for excluded in [
        SecretClient._credentials,
        SecretClient._token,
        "PRIVATE-REASONING-NEVER-LOG",
    ]:
        assert excluded not in serialized
    assert result["usage"]["total_tokens"] == 45


def test_unparseable_rejected_candidate_keeps_error_without_raw_private_prose(corpus):
    class BrokenJSONClient(FakeClient):
        def chat(self, messages, ledger, **kwargs):
            response = super().chat(messages, ledger, **kwargs)
            if kwargs.get("schema"):
                response["message"]["content"] = (
                    "<think>PRIVATE-UNPARSED-REASONING</think>{broken"
                )
            return response

    result = Pipeline(SearchEngine(corpus, BrokenJSONClient()), BASELINE).run("Иван")
    failure = next(t for t in result["trace"] if t["type"] == "validation_error")
    assert failure["error_type"] == "JSONDecodeError"
    assert failure["reasons"][0]["type"] == "json_decode_error"
    assert failure["rejected_public_content"] is None
    assert failure["raw_content_chars"] > 0
    assert len(failure["raw_content_sha256"]) == 64
    assert "PRIVATE-UNPARSED-REASONING" not in json.dumps(result)


def test_unknown_tool_not_executed(corpus):
    result = Pipeline(
        SearchEngine(corpus, FakeClient(tool="delete_files")), BASELINE
    ).run("Иван")
    assert result["error"]["code"] == "invalid_tool_name_or_arguments"
    assert result["usage"]["total_tokens"] == 15


def test_budget_blocks_before_network(corpus):
    client = FakeClient()
    result = Pipeline(
        SearchEngine(corpus, client), replace(BASELINE, max_total_tokens=1)
    ).run("Иван")
    assert result["error"]["type"] == "BudgetExceeded"
    assert client.calls == 0


def test_exact_cache_and_invalidation(corpus, tmp_path):
    cache = ExactCache(tmp_path)
    client = FakeClient()
    pipeline = Pipeline(SearchEngine(corpus, client), BASELINE, cache=cache)
    first = pipeline.run("Иван")
    second = pipeline.run("Иван")
    assert not first["cache_hit"] and second["cache_hit"]
    assert second["usage"]["total_tokens"] == 0
    assert client.calls == 2
    assert cache.key("a", "corpus", {}, "model") != cache.key(
        "a", "changed", {}, "model"
    )
    assert cache.key("a", "corpus", {}, "model") != cache.key(
        "a", "corpus", {"schema": "v2"}, "model"
    )
    assert cache.key("a", "corpus", {}, "model") != cache.key(
        "a", "corpus", {}, "model-v2"
    )


def test_search_filters_and_argument_validation(corpus):
    engine = SearchEngine(corpus, FakeClient())
    assert (
        engine.execute("keyword_search", {"query": "Анна", "year": 2016}, Ledger())
        == []
    )
    assert (
        engine.execute("keyword_search", {"query": "Анна", "year": 2014}, Ledger())[0][
            "chunk_id"
        ]
        == "two"
    )
    with pytest.raises(ValueError):
        engine.execute("keyword_search", {"query": "Иван", "year": "2016"}, Ledger())


def test_context_never_slices_evidence(corpus):
    assert select_context(corpus.chunks, deduplicate=True, byte_budget=1) == []
    hit = corpus.chunks[0]
    assert len(select_context([hit, hit], deduplicate=True, byte_budget=10000)) == 1


def test_consensus_compares_facts_not_answer_words(corpus):
    a = Answer(
        status="ok",
        answer="Альфа.",
        citations=[{"chunk_id": "one", "quote": "Иван представляет компанию Альфа."}],
    )
    b = a.model_copy(update={"answer": "Компания Альфа представлена Иваном."})
    facts = [
        {
            "entity": "Иван",
            "attribute": "организация",
            "value": "Альфа",
            "chunk_id": "one",
        }
    ]
    assert fact_fingerprint(facts, a, corpus.by_id) == fact_fingerprint(
        facts, b, corpus.by_id
    )


def test_index_rejects_invalid_matrix(corpus):
    with pytest.raises(ValueError):
        Corpus(corpus.chunks, np.zeros((2, 2)), {})
    with pytest.raises(ValueError):
        Corpus(corpus.chunks, np.ones((1, 2)), {})


def test_prepared_index_preserves_version_and_vectors(corpus, tmp_path):
    corpus.prepare(tmp_path)
    loaded = Corpus.load(tmp_path)
    assert loaded.version == corpus.version
    assert np.array_equal(loaded.raw_matrix, corpus.raw_matrix)


def test_react_search_bound(corpus):
    client = FakeClient()
    result = Pipeline(
        SearchEngine(corpus, client), replace(BASELINE, max_search_calls=3)
    ).run("Иван")
    assert not result["error"]
    assert len([t for t in result["trace"] if t["type"] == "tool_call"]) == 3
    assert client.calls == 4


def test_self_consistency_runs_three_candidates(corpus):
    class CandidateClient(FakeClient):
        def chat(self, messages, ledger, **kwargs):
            response = super().chat(messages, ledger, **kwargs)
            if kwargs.get("schema"):
                payload = json.loads(response["message"]["content"])
                payload["facts"] = [
                    {
                        "entity": "Иван",
                        "attribute": "организация",
                        "value": "Альфа",
                        "chunk_id": "one",
                    }
                ]
                payload["answer"] = (
                    "Альфа." if self.calls % 2 else "Иван представляет компанию Альфа."
                )
                response["message"]["content"] = json.dumps(payload, ensure_ascii=False)
            return response

    client = CandidateClient()
    result = Pipeline(
        SearchEngine(corpus, client), replace(BASELINE, candidates=3)
    ).run("Иван")
    assert not result["error"]
    assert client.calls == 4
    assert next(t for t in result["trace"] if t["type"] == "consensus")["agreeing"] == 3
