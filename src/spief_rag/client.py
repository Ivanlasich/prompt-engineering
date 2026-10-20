"""Public GigaChat REST provider: OAuth, TLS, native functions and embeddings."""

from __future__ import annotations

import json
import math
import os
import ssl
import threading
import time
import uuid
from typing import Any

import certifi
import httpx

from .settings import Settings
from .telemetry import Ledger


class APIError(RuntimeError):
    """Safe error without request headers, credentials or provider response body."""

    def __init__(self, code: str, status: int | None = None):
        """Keep only a safe symbolic code and HTTP status."""
        self.code, self.status = code, status
        super().__init__(f"{code}" + (f" (HTTP {status})" if status else ""))


def tls_context(ca_bundle: str | None = None) -> ssl.SSLContext:
    """Trust standard roots and an optional explicit additional CA bundle."""
    context = ssl.create_default_context(cafile=certifi.where())
    if ca_bundle:
        context.load_verify_locations(cafile=ca_bundle)
    return context


def clean_message(message: dict) -> dict:
    """Whitelist native continuation fields; never retain private reasoning."""
    return {
        k: v
        for k, v in message.items()
        if k in {"role", "content", "name", "function_call", "functions_state_id"}
        and v is not None
    }


class GigaChatClient:
    """Small provider with injectable HTTP transport for offline contract tests."""

    def __init__(
        self,
        settings: Settings,
        credentials: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        """Create one reusable TLS transport and isolated OAuth cache."""
        self.settings = settings
        self._credentials = credentials or os.getenv("GIGACHAT_CREDENTIALS", "")
        if not self._credentials:
            raise APIError("missing_GIGACHAT_CREDENTIALS")
        self._http = httpx.Client(
            verify=tls_context(settings.ca_bundle),
            timeout=settings.timeout,
            transport=transport,
            follow_redirects=False,
        )
        self._token = ""
        self._expires = 0.0
        self._lock = threading.Lock()

    def close(self) -> None:
        """Release transport resources."""
        self._http.close()

    def _access_token(self) -> str:
        """Refresh under a lock; retry transient authorization transport failures only."""
        with self._lock:
            if self._token and time.time() < self._expires - 60:
                return self._token
            for attempt in range(self.settings.retries + 1):
                try:
                    response = self._http.post(
                        self.settings.oauth_url,
                        headers={
                            "Authorization": f"Basic {self._credentials}",
                            "RqUID": str(uuid.uuid4()),
                        },
                        data={"scope": self.settings.scope},
                    )
                except httpx.TransportError:
                    if attempt < self.settings.retries:
                        time.sleep(min(2**attempt, 4))
                        continue
                    raise APIError("oauth_transport_error") from None
                if (
                    response.status_code in {429, 500, 502, 503, 504}
                    and attempt < self.settings.retries
                ):
                    time.sleep(min(2**attempt, 4))
                    continue
                break
            if response.status_code != 200:
                code = "oauth_error"
                try:
                    if response.json().get(
                        "code"
                    ) == 7 and "scope" in response.json().get("message", ""):
                        code = "oauth_scope_mismatch"
                except (ValueError, AttributeError):
                    pass
                raise APIError(code, response.status_code)
            try:
                body = response.json()
                self._token = body["access_token"]
                expires = float(body.get("expires_at", time.time() + 1500))
                self._expires = expires / 1000 if expires > 1e11 else expires
            except (ValueError, KeyError, TypeError):
                raise APIError("invalid_oauth_response") from None
            return self._token

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        ledger: Ledger | None = None,
        operation: str = "diagnostic",
    ) -> dict:
        """Retry only transient failures, accounting every observable attempt."""
        model = (payload or {}).get("model", self.settings.model)
        if ledger and payload:
            ledger.check(payload, payload.get("max_tokens", 0))
        for attempt in range(self.settings.retries + 1):
            if ledger and payload:
                ledger.check(payload, payload.get("max_tokens", 0))
            token = self._access_token()
            started = time.perf_counter()
            try:
                response = self._http.request(
                    method,
                    self.settings.api_base.rstrip("/") + path,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "X-Request-ID": str(uuid.uuid4()),
                    },
                )
            except httpx.TransportError:
                if ledger:
                    ledger.record(
                        operation,
                        model,
                        None,
                        (time.perf_counter() - started) * 1000,
                        status="transport_error",
                        attempt=attempt,
                    )
                if attempt < self.settings.retries:
                    time.sleep(min(2**attempt, 4))
                    continue
                raise APIError("transport_error") from None
            latency = (time.perf_counter() - started) * 1000
            try:
                decoded = response.json()
                body = decoded
            except ValueError:
                decoded = None
                body = {}
            if not isinstance(body, dict):
                body = {}
            usage = body.get("usage")
            # Public embeddings place usage on each data item, not at top level.
            if (
                operation == "embedding"
                and usage is None
                and isinstance(body.get("data"), list)
            ):
                item_usage = [
                    r.get("usage", {}).get("prompt_tokens")
                    for r in body["data"]
                    if isinstance(r, dict)
                ]
                if (
                    item_usage
                    and len(item_usage) == len(body["data"])
                    and all(type(n) is int and n >= 0 for n in item_usage)
                ):
                    usage = {
                        "prompt_tokens": sum(item_usage),
                        "completion_tokens": 0,
                        "total_tokens": sum(item_usage),
                    }
            # Record even failed/empty/invalid responses before any validation.
            if ledger:
                ledger.record(
                    operation,
                    body.get("model", model),
                    usage,
                    latency,
                    status=response.status_code,
                    attempt=attempt,
                    request_id=response.headers.get("x-request-id"),
                )
            if response.is_success:
                if isinstance(decoded, list) and path == "/tokens/count":
                    return {"data": decoded}
                if not isinstance(decoded, dict):
                    raise APIError("invalid_response")
                return body
            if (
                response.status_code in {429, 500, 502, 503, 504}
                and attempt < self.settings.retries
            ):
                try:
                    delay = min(
                        float(response.headers.get("Retry-After", 2**attempt)), 10
                    )
                except ValueError:
                    delay = 2**attempt
                time.sleep(max(delay, 0))
                continue
            raise APIError("api_error", response.status_code)
        raise APIError("retry_exhausted")

    def models(self) -> list[str]:
        """Read the model IDs available to the configured project."""
        return [v["id"] for v in self._request("GET", "/models").get("data", [])]

    def chat(
        self,
        messages: list[dict],
        ledger: Ledger,
        *,
        functions: list[dict] | None = None,
        function_call: str | dict = "auto",
        schema: dict | None = None,
        temperature: float = 0.01,
        max_tokens: int = 1200,
    ) -> dict:
        """Return a normalized native response, preserving function continuation."""
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [clean_message(x) for x in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if functions:
            payload.update(functions=functions, function_call=function_call)
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "schema": schema,
                "strict": True,
            }
        body = self._request("POST", "/chat/completions", payload, ledger, "generation")
        try:
            choice = body["choices"][0]
            message = clean_message(choice["message"])
            reason = choice["finish_reason"]
        except (KeyError, IndexError, TypeError):
            raise APIError("invalid_chat_response") from None
        if reason not in {"stop", "function_call"}:
            raise APIError("incomplete_generation_" + str(reason))
        call = message.get("function_call")
        if call:
            try:
                arguments = call["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError()
                message["function_call"] = {
                    "name": call["name"],
                    "arguments": arguments,
                }
            except (KeyError, ValueError, TypeError):
                raise APIError("invalid_function_arguments") from None
        return {
            "message": message,
            "finish_reason": reason,
            "model": body.get("model", self.settings.model),
        }

    def embed(self, texts: list[str], ledger: Ledger) -> list[list[float]]:
        """Validate embedding ordering, count, dimensions and finite coordinates."""
        if not texts or any(not t.strip() for t in texts):
            raise ValueError("empty_embedding_input")
        body = self._request(
            "POST",
            "/embeddings",
            {"model": self.settings.embedding_model, "input": texts},
            ledger,
            "embedding",
        )
        try:
            rows = body["data"]
            if len(rows) != len(texts) or sorted(v["index"] for v in rows) != list(
                range(len(texts))
            ):
                raise ValueError()
            vectors = [v["embedding"] for v in sorted(rows, key=lambda x: x["index"])]
            dimension = len(vectors[0])
            if self.settings.embedding_model == "EmbeddingsGigaR" and dimension != 2560:
                raise ValueError()
            if any(
                len(v) != dimension
                or not v
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not math.isfinite(x)
                    for x in v
                )
                for v in vectors
            ):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise APIError("invalid_embeddings") from None
        return vectors

    def count_tokens(self, texts: list[str]) -> list[dict]:
        """Exact text count where supported; request overhead remains separate."""
        return self._request(
            "POST", "/tokens/count", {"model": self.settings.model, "input": texts}
        ).get("data", [])

    def preflight(self) -> dict:
        """One explicit safe capability probe, without executing business tools."""
        models = self.models()
        if self.settings.model not in models:
            raise APIError("configured_model_unavailable")
        ledger = Ledger()
        plain = self.chat(
            [{"role": "user", "content": "Ответь одним словом: готов"}],
            ledger,
            max_tokens=20,
        )
        spec = {
            "name": "probe_echo",
            "description": "Вернуть заданный текст без внешних действий.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
        messages = [{"role": "user", "content": "Вызови probe_echo с text=готов"}]
        called = self.chat(
            messages,
            ledger,
            functions=[spec],
            function_call={"name": "probe_echo"},
            max_tokens=80,
        )
        call = called["message"].get("function_call")
        if not call or call["name"] != "probe_echo":
            raise APIError("function_probe_failed")
        messages += [
            called["message"],
            {
                "role": "function",
                "name": "probe_echo",
                "content": json.dumps({"text": "готов"}, ensure_ascii=False),
            },
        ]
        continued = self.chat(messages, ledger, max_tokens=20)
        schema = {
            "type": "object",
            "properties": {"ready": {"type": "boolean"}},
            "required": ["ready"],
        }
        structured = self.chat(
            [{"role": "user", "content": "Верни ready=true"}],
            ledger,
            schema=schema,
            max_tokens=40,
        )
        if json.loads(structured["message"]["content"]) != {"ready": True}:
            raise APIError("structured_probe_failed")
        vector = self.embed(["проверка эмбеддингов"], ledger)
        return {
            "models": models,
            "model": self.settings.model,
            "api_base": self.settings.api_base,
            "scope": self.settings.scope,
            "tls_verification": True,
            "text": bool(plain["message"].get("content")),
            "native_function_call": True,
            "function_continuation": bool(continued["message"].get("content")),
            "functions_state_id": bool(called["message"].get("functions_state_id")),
            "json_schema": True,
            "embedding_dimension": len(vector[0]),
            "usage": ledger.summary(),
            "events": ledger.events,
        }
