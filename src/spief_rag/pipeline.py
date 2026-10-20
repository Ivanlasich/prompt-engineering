"""Bounded native function-calling RAG with strict grounded answers."""

from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter
from typing import Any

from pydantic import ValidationError

from .cache import ExactCache
from .client import APIError, clean_message
from .contracts import Answer, normalized, public_schema, validate_grounding
from .corpus import digest
from .prompts import SEARCH_SYSTEM, answer_system
from .search import SearchEngine, select_context, tool_specs
from .settings import Experiment
from .telemetry import BudgetExceeded, Ledger, run_identity

FINAL_REQUEST = (
    "Дай окончательный ответ на исходный вопрос по найденным источникам. JSON по схеме."
)
REPAIR_REQUEST = "Исправь JSON: все поля по схеме; цитаты дословны из выданного контекста; ok требует подтверждения всех фактов, иначе not_found без цитат."


def generation_contract(config: Experiment) -> dict:
    """Expose the exact candidate prompt/schema used by execution and run signatures."""
    schema = public_schema()
    system = answer_system(config)
    if config.candidates > 1:
        schema["properties"]["facts"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    k: {"type": "string"}
                    for k in ["entity", "attribute", "value", "chunk_id"]
                },
                "required": ["entity", "attribute", "value", "chunk_id"],
                "additionalProperties": False,
            },
        }
        schema["required"].append("facts")
        system += "\nДополнительно facts: краткие проверяемые факты entity/attribute/value/chunk_id. value — точная подстрока источника; канонические имена без пояснений. При not_found facts=[]."
    return {
        "system": system,
        "schema": schema,
        "final_request": FINAL_REQUEST,
        "repair_request": REPAIR_REQUEST,
    }


def validation_diagnostic(exc: Exception, content: str, payload: Any, client) -> dict:
    """Retain bounded public fields and error codes, excluding reasoning and auth secrets."""

    def redact(value: str) -> str:
        """Mask known authorization values and standard Authorization header tokens."""
        for name in ("_credentials", "_token"):
            secret = getattr(client, name, None)
            if isinstance(secret, str) and secret:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(
            r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9+/_.=-]+", r"\1 [REDACTED]", value
        )

    if isinstance(exc, ValidationError):
        reasons = [
            {"type": e["type"], "location": list(e["loc"]), "message": redact(e["msg"])}
            for e in exc.errors(
                include_url=False, include_context=False, include_input=False
            )
        ]
    elif isinstance(exc, json.JSONDecodeError):
        reasons = [
            {
                "type": "json_decode_error",
                "message": exc.msg,
                "line": exc.lineno,
                "column": exc.colno,
            }
        ]
    else:
        safe_codes = {
            "response_not_object",
            "citation_not_retrieved",
            "quote_not_in_context",
            "missing_candidate_facts",
            "invalid_candidate_fact",
            "candidate_fact_not_grounded",
        }
        reasons = [
            {
                "type": type(exc).__name__,
                "message": str(exc)
                if str(exc) in safe_codes
                else "candidate_validation_failed",
            }
        ]
    public = {}
    if isinstance(payload, dict):
        for name in ("status", "answer"):
            if isinstance(payload.get(name), str):
                public[name] = payload[name]
        if isinstance(payload.get("citations"), list):
            public["citations"] = [
                {k: c[k] for k in ("chunk_id", "quote") if isinstance(c.get(k), str)}
                for c in payload["citations"]
                if isinstance(c, dict)
            ]
    serialized = redact(json.dumps(public, ensure_ascii=False)) if public else None
    return {
        "error_type": type(exc).__name__,
        "reasons": reasons,
        "rejected_public_content": serialized[:12000] if serialized else None,
        "diagnostic_truncated": bool(serialized and len(serialized) > 12000),
        # Unparseable prose may contain private reasoning; keep only its fingerprint.
        "raw_content_chars": len(content),
        "raw_content_sha256": digest(content),
    }


def parse_json(text: str) -> dict:
    """Parse JSON without accepting arbitrary prose surrounding an answer."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise TypeError("response_not_object")
    return value


def fact_fingerprint(
    facts: list[dict], answer: Answer, evidence: dict[str, dict]
) -> tuple:
    """Agree on explicit entity/attribute/value facts with evidence, not answer wording."""
    if answer.status == "not_found":
        return ("not_found",)
    if not facts:
        raise ValueError("missing_candidate_facts")
    normalized_facts = []
    for fact in facts:
        if set(fact) != {"entity", "attribute", "value", "chunk_id"}:
            raise ValueError("invalid_candidate_fact")
        chunk = evidence.get(fact["chunk_id"])
        if (
            not chunk
            or not fact["value"].strip()
            or normalized(fact["value"]) not in normalized(chunk["text"])
        ):
            raise ValueError("candidate_fact_not_grounded")
        def normal(s):
            """Canonicalize the comparison fields without comparing prose answers."""
            return " ".join(re.findall(r"\w+", normalized(s)))
        normalized_facts.append(
            tuple(normal(fact[k]) for k in ["entity", "attribute", "value"])
        )
    return tuple(sorted(set(normalized_facts)))


class Pipeline:
    """Execute search and generation under one ledger and explicit call limits."""

    def __init__(
        self,
        engine: SearchEngine,
        config: Experiment,
        tariff: dict | None = None,
        cache: ExactCache | None = None,
    ):
        """Bind immutable settings; per-question state stays inside run()."""
        self.engine, self.client, self.config = engine, engine.client, config
        self.tariff, self.cache = tariff, cache

    def run(self, question: str) -> dict:
        """Return answer or operational error; preserve every paid partial attempt."""
        started, identity = time.perf_counter(), run_identity()
        config = self.config
        contract = generation_contract(config)
        ledger = Ledger(config.max_total_tokens, config.max_request_byte_tokens)
        trace: list[dict] = []
        evidence: dict[str, dict] = {}
        result: dict[str, Any] = {
            **identity,
            "question": question,
            "config": config.to_dict(),
            "model": self.client.settings.model,
            "corpus_version": self.engine.corpus.version,
            "prompt_hash": digest(contract["system"]),
            "schema_hash": digest(contract["schema"]),
            "cache_hit": False,
            "answer": None,
            "error": None,
            "sources": [],
            "trace": trace,
        }
        cache_key = (
            self.cache.key(
                question,
                self.engine.corpus.version,
                config.to_dict(),
                self.client.settings.model
                + "|"
                + result["prompt_hash"]
                + "|"
                + result["schema_hash"],
            )
            if self.cache
            else None
        )
        if self.cache and (cached := self.cache.get(cache_key)):
            # Cached costs belong to the original run, never to this hit.
            result.update(
                answer=cached["answer"],
                sources=cached["sources"],
                cache_hit=True,
                cached_from_run=cached["run_id"],
                cached_evidence=cached.get("evidence", []),
            )
            result["usage"], result["events"], result["latency_ms"] = (
                ledger.summary(self.tariff),
                [],
                (time.perf_counter() - started) * 1000,
            )
            return result
        try:
            if not question.strip():
                raise ValueError("empty_question")
            history: list[dict] = [
                {"role": "system", "content": SEARCH_SYSTEM},
                {"role": "user", "content": question},
            ]
            for step in range(config.max_search_calls):
                response = self.client.chat(
                    history,
                    ledger,
                    functions=tool_specs(),
                    max_tokens=300,
                    temperature=config.temperature,
                )
                message = response["message"]
                call = message.get("function_call")
                if not call:
                    if evidence:
                        break
                    raise APIError("model_did_not_call_search")
                trace.append(
                    {
                        "type": "tool_call",
                        "step": step + 1,
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                )
                history.append(clean_message(message))
                try:
                    hits = self.engine.execute(
                        call["name"],
                        call["arguments"],
                        ledger,
                        top_k=config.top_k,
                        min_score=config.min_score,
                    )
                except (ValueError, ValidationError):
                    # The invalid tool is never executed. Give the model a bounded correction opportunity.
                    tool_result = {
                        "error": "invalid_tool_name_or_arguments",
                        "hits": [],
                    }
                    trace.append(
                        {
                            "type": "tool_error",
                            "step": step + 1,
                            "code": tool_result["error"],
                        }
                    )
                    history.append(
                        {
                            "role": "function",
                            "name": call["name"],
                            "content": json.dumps(tool_result),
                        }
                    )
                    if step + 1 == config.max_search_calls:
                        raise ValueError("invalid_tool_name_or_arguments") from None
                    continue
                selected = select_context(
                    list(evidence.values()) + hits,
                    deduplicate=config.deduplicate,
                    byte_budget=config.context_byte_budget,
                )
                evidence = {r["chunk_id"]: r for r in selected}
                # Each tool result is a complete bounded snapshot, not an arbitrary prefix.
                shown = [
                    {
                        k: r[k]
                        for k in [
                            "chunk_id",
                            "session",
                            "time_start",
                            "speakers",
                            "text",
                        ]
                    }
                    for r in selected
                ]
                tool_result = {"hits": shown}
                history.append(
                    {
                        "role": "function",
                        "name": call["name"],
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
                trace.append(
                    {
                        "type": "retrieval",
                        "step": step + 1,
                        "tool": call["name"],
                        "raw_hit_ids": [h["chunk_id"] for h in hits],
                        "context_ids": list(evidence),
                        "scores": [h["score"] for h in hits],
                    }
                )
            if not evidence:
                # Empty retrieval is an abstention; API/validation failures never enter this branch.
                answer = Answer(
                    status="not_found",
                    answer="В базе знаний не найдено достаточно сведений.",
                    citations=[],
                )
                result["answer"] = answer.model_dump()
            else:
                # Preserve native assistant-call / function-result pairs and state IDs.
                generation_history = [
                    {"role": "system", "content": contract["system"]},
                    *history[1:],
                ]
                generation_history.append(
                    {
                        "role": "user",
                        "content": contract["final_request"],
                    }
                )
                candidates, fingerprints = [], []
                for number in range(config.candidates):
                    schema = contract["schema"]
                    messages = copy.deepcopy(generation_history)
                    candidate = None
                    for repair in range(config.max_repairs + 1):
                        response = self.client.chat(
                            messages,
                            ledger,
                            schema=schema,
                            max_tokens=config.max_output_tokens,
                            temperature=0.6
                            if config.candidates > 1
                            else config.temperature,
                        )
                        payload = None
                        try:
                            payload = parse_json(response["message"].get("content", ""))
                            facts = payload.pop("facts", [])
                            candidate = Answer.model_validate(payload)
                            validate_grounding(candidate, evidence)
                            fingerprint = (
                                fact_fingerprint(facts, candidate, evidence)
                                if config.candidates > 1
                                else ()
                            )
                            trace.append(
                                {
                                    "type": "candidate",
                                    "number": number + 1,
                                    "answer": candidate.model_dump(),
                                    "facts": facts,
                                    "repairs": repair,
                                }
                            )
                            fingerprints.append(fingerprint)
                            break
                        except (ValueError, ValidationError, TypeError) as exc:
                            candidate = None
                            trace.append(
                                {
                                    "type": "validation_error",
                                    "number": number + 1,
                                    "repair": repair,
                                    **validation_diagnostic(
                                        exc,
                                        response["message"].get("content", ""),
                                        payload,
                                        self.client,
                                    ),
                                }
                            )
                            if repair == config.max_repairs:
                                break
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": response["message"].get("content", ""),
                                }
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": contract["repair_request"],
                                }
                            )
                    if candidate is not None:
                        candidates.append(candidate)
                if not candidates:
                    raise ValueError("no_valid_candidate")
                if config.candidates > 1:
                    fingerprint, agreement = Counter(fingerprints).most_common(1)[0]
                    trace.append(
                        {
                            "type": "consensus",
                            "agreeing": agreement,
                            "requested": config.candidates,
                        }
                    )
                    if agreement < 2:
                        raise ValueError("self_consistency_no_consensus")
                    answer = candidates[fingerprints.index(fingerprint)]
                else:
                    answer = candidates[0]
                result["answer"] = answer.model_dump()
            if result["answer"]:
                for citation in result["answer"]["citations"]:
                    chunk = evidence[citation["chunk_id"]]
                    result["sources"].append(
                        {
                            k: chunk[k]
                            for k in [
                                "chunk_id",
                                "source_id",
                                "session",
                                "time_start",
                                "speakers",
                            ]
                        }
                    )
        except (APIError, BudgetExceeded, ValueError, ValidationError, KeyError) as exc:
            # Safe symbolic error; never stringify raw provider bodies or credentials.
            result["error"] = {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", str(exc).split("\n")[0][:100]),
                "http_status": getattr(exc, "status", None),
            }
            result["answer"] = None
        result["evidence"] = [
            {
                k: r[k]
                for k in [
                    "chunk_id",
                    "source_id",
                    "session",
                    "time_start",
                    "speakers",
                    "text",
                ]
            }
            for r in evidence.values()
        ]
        result["events"], result["usage"] = ledger.events, ledger.summary(self.tariff)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if self.cache and not result["error"]:
            self.cache.put(cache_key, result)
        return result
