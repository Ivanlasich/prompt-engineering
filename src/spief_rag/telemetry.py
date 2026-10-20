"""Usage accounting that survives validation and transport failures."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path


class BudgetExceeded(RuntimeError):
    """A local budget prevented an additional paid request."""


@dataclass
class Ledger:
    """Append-only per-question observations, including unsuccessful attempts."""

    max_tokens: int = 100000
    max_request_byte_tokens: int = 120000
    events: list[dict] = field(default_factory=list)

    def check(self, payload: dict, reserve: int = 0) -> None:
        """Reserve a conservative UTF-8 byte bound, explicitly not API token count."""
        bound = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + reserve
        if (
            bound > self.max_request_byte_tokens
            or self.total_tokens + bound > self.max_tokens
        ):
            raise BudgetExceeded("preflight_byte_bound_exceeded")

    @property
    def total_tokens(self) -> int:
        """Sum provider-reported usage, excluding unknown failed attempts."""
        return sum(e.get("total_tokens", 0) for e in self.events)

    def record(
        self,
        operation: str,
        model: str,
        usage: dict | None,
        latency_ms: float,
        **metadata,
    ) -> None:
        """Record usage immediately, before parsing or validating the response."""
        raw = usage or {}
        prompt = raw.get("prompt_tokens", raw.get("input_tokens"))
        output = raw.get("completion_tokens", raw.get("output_tokens"))
        total = raw.get("total_tokens")
        if total is None and prompt is not None:
            total = prompt + (output or 0)
        cached = raw.get(
            "precached_prompt_tokens",
            raw.get("input_tokens_details", {}).get("cached_tokens", 0),
        )
        self.events.append(
            {
                "operation": operation,
                "model": model,
                "prompt_tokens": prompt or 0,
                "completion_tokens": output or 0,
                "total_tokens": total or 0,
                "cached_prompt_tokens": cached or 0,
                "usage_known": total is not None,
                "latency_ms": round(latency_ms, 2),
                **metadata,
            }
        )

    def summary(self, tariff: dict | None = None) -> dict:
        """Report known usage and estimated package-equivalent cost separately."""
        result = {
            k: sum(e.get(k, 0) for e in self.events)
            for k in [
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cached_prompt_tokens",
            ]
        }
        generation = sum(
            e["total_tokens"] for e in self.events if e["operation"] == "generation"
        )
        embedding = sum(
            e["total_tokens"] for e in self.events if e["operation"] == "embedding"
        )
        result.update(
            generation_tokens=generation,
            embedding_tokens=embedding,
            api_attempts=len(self.events),
            unknown_usage_attempts=sum(not e["usage_known"] for e in self.events),
            api_latency_ms=round(sum(e["latency_ms"] for e in self.events), 2),
        )
        if tariff:
            # GigaChat v1 prompt/total already exclude precached tokens. Do not subtract twice.
            billable_gen = Decimal(generation)
            cost = (
                billable_gen * Decimal(str(tariff["generation_rub_per_1000"]))
                + Decimal(embedding) * Decimal(str(tariff["embedding_rub_per_1000"]))
            ) / 1000
            result["estimated_cost_rub"] = float(cost)
            result["cost_label"] = tariff["label"]
            result["cash_charge_confirmed"] = tariff.get("cash_charge_confirmed", False)
        else:
            result["estimated_cost_rub"] = None
        result["logical_tokens_including_cached"] = (
            result["total_tokens"] + result["cached_prompt_tokens"]
        )
        return result


def run_identity() -> dict:
    """Generate a safe local run identity."""
    return {
        "run_id": str(uuid.uuid4()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def append_jsonl(path: Path, row: dict) -> None:
    """Flush each completed run so an interrupted benchmark can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
