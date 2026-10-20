"""Provider contract tests use local HTTP mocks and never spend API tokens."""

import json
import ssl
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from spief_rag.client import APIError, GigaChatClient, tls_context
from spief_rag.settings import Settings
from spief_rag.telemetry import Ledger


def mock_client(handler, retries=0):
    """Wrap an API handler with a safe local OAuth endpoint."""

    def transport(request):
        if request.url.path.endswith("oauth"):
            assert request.headers["RqUID"]
            assert b"scope=GIGACHAT_API_PERS" in request.content
            return httpx.Response(
                200,
                json={
                    "access_token": "test-token",
                    "expires_at": (time.time() + 1800) * 1000,
                },
            )
        return handler(request)

    return GigaChatClient(
        Settings(retries=retries), "test-only", httpx.MockTransport(transport)
    )


def reply(message, reason="stop", tokens=12):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": message, "finish_reason": reason}],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": tokens - 8,
                "total_tokens": tokens,
            },
            "model": "test-model",
        },
    )


def test_native_function_history_and_state():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert "tools" not in body
        if len(seen) == 1:
            assert body["function_call"] == {"name": "keyword_search"}
            return reply(
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "keyword_search",
                        "arguments": '{"query":"ПМЭФ"}',
                    },
                    "functions_state_id": "state-1",
                    "reasoning_content": "PRIVATE",
                },
                "function_call",
            )
        assert body["messages"][1]["functions_state_id"] == "state-1"
        assert body["messages"][2]["role"] == "function"
        assert isinstance(body["messages"][2]["content"], str)
        return reply({"role": "assistant", "content": "готов"})

    client = mock_client(handler)
    ledger = Ledger()
    initial = [{"role": "user", "content": "Поиск"}]
    response = client.chat(
        initial,
        ledger,
        functions=[{"name": "keyword_search"}],
        function_call={"name": "keyword_search"},
    )
    assert "reasoning_content" not in response["message"]
    client.chat(
        initial
        + [
            response["message"],
            {"role": "function", "name": "keyword_search", "content": "{}"},
        ],
        ledger,
    )
    assert ledger.total_tokens == 24


def test_truncated_generation_still_costs_tokens():
    client = mock_client(
        lambda r: reply({"role": "assistant", "content": "{"}, "length", 22)
    )
    ledger = Ledger()
    with pytest.raises(APIError, match="incomplete_generation"):
        client.chat([{"role": "user", "content": "x"}], ledger)
    assert ledger.total_tokens == 22


@pytest.mark.parametrize(
    "status,retry_count",
    [(400, 1), (401, 1), (402, 1), (403, 1), (404, 1), (422, 1), (429, 2), (503, 2)],
)
def test_retry_classification(status, retry_count, monkeypatch):
    monkeypatch.setattr("spief_rag.client.time.sleep", lambda _: None)
    calls = []

    def handler(r):
        calls.append(r)
        return httpx.Response(status, json={"error": "test"})

    client = mock_client(handler, retries=1)
    ledger = Ledger()
    with pytest.raises(APIError):
        client.chat([{"role": "user", "content": "x"}], ledger)
    assert len(calls) == retry_count
    assert ledger.summary()["unknown_usage_attempts"] == retry_count


def test_oauth_lock_and_cache():
    count = []

    def handler(request):
        if request.url.path.endswith("oauth"):
            count.append(1)
            return httpx.Response(
                200,
                json={"access_token": "test-token", "expires_at": time.time() + 1800},
            )
        return httpx.Response(200, json={"data": [{"id": "model"}]})

    client = GigaChatClient(Settings(), "test-only", httpx.MockTransport(handler))
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda _: client.models(), range(8))) == [["model"]] * 8
    assert len(count) == 1


def test_embeddings_order_and_invalid_indices():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [2.0] * 2560},
                    {"index": 0, "embedding": [1.0] * 2560},
                ],
                "usage": {"total_tokens": 8},
            },
        )

    client = mock_client(handler)
    assert client.embed(["a", "b"], Ledger())[0][0] == 1
    with pytest.raises(APIError, match="invalid_embeddings"):
        client.embed(["a"], Ledger())


def test_tls_and_no_secret_in_errors():
    assert tls_context().verify_mode == ssl.CERT_REQUIRED
    assert tls_context().check_hostname
    client = mock_client(
        lambda _: httpx.Response(403, json={"error": "sensitive body test-only"})
    )
    with pytest.raises(APIError) as exc:
        client.models()
    assert "sensitive" not in str(exc.value)
    assert "test-only" not in str(exc.value)


def test_structured_payload():
    def handler(request):
        assert json.loads(request.content)["response_format"] == {
            "type": "json_schema",
            "schema": {"type": "object"},
            "strict": True,
        }
        return reply({"role": "assistant", "content": "{}"})

    mock_client(handler).chat(
        [{"role": "user", "content": "x"}], Ledger(), schema={"type": "object"}
    )


def test_embedding_usage_per_item_is_billed():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": 0,
                        "embedding": [1.0] * 2560,
                        "usage": {"prompt_tokens": 17},
                    }
                ]
            },
        )

    ledger = Ledger()
    mock_client(handler).embed(["text"], ledger)
    assert ledger.summary()["embedding_tokens"] == 17
    assert ledger.summary()["unknown_usage_attempts"] == 0


def test_token_counter_accepts_native_array():
    client = mock_client(
        lambda _: httpx.Response(200, json=[{"tokens": 3, "characters": 6}])
    )
    assert client.count_tokens(["пример"]) == [{"tokens": 3, "characters": 6}]
