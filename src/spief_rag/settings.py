"""Configuration and versioned experiment definitions."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Transport settings; credentials are read separately and never serialized."""

    api_base: str = "https://api.giga.chat/v1"
    oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    scope: str = "GIGACHAT_API_PERS"
    model: str = "GigaChat-2-Max"
    embedding_model: str = "EmbeddingsGigaR"
    ca_bundle: str | None = None
    timeout: float = 90
    retries: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        """Process environment wins over ignored local dotenv files."""
        load_dotenv(ROOT / ".env.local", override=False)
        load_dotenv(ROOT / ".env", override=False)
        ca = os.getenv("GIGACHAT_CA_BUNDLE_FILE") or None
        if ca and not Path(ca).is_absolute():
            ca = str(ROOT / ca)
        return cls(
            api_base=os.getenv("GIGACHAT_API_BASE", cls.api_base),
            oauth_url=os.getenv("GIGACHAT_OAUTH_URL", cls.oauth_url),
            scope=os.getenv("GIGACHAT_SCOPE", cls.scope),
            model=os.getenv("GIGACHAT_MODEL", cls.model),
            ca_bundle=ca,
            timeout=float(os.getenv("GIGACHAT_TIMEOUT", "90")),
        )


@dataclass(frozen=True)
class Experiment:
    """One reproducible pipeline variant; all changes appear in run artifacts."""

    name: str = "baseline"
    top_k: int = 8
    min_score: float = -1
    deduplicate: bool = False
    short_prompt: bool = False
    examples: int = 3
    max_output_tokens: int = 1200
    context_byte_budget: int = 40000
    max_search_calls: int = 1
    candidates: int = 1
    max_total_tokens: int = 100000
    max_request_byte_tokens: int = 120000
    temperature: float = 0.01
    max_repairs: int = 1
    schema_version: str = "answer-1.0"
    prompt_version: str = "1.0"

    def __post_init__(self) -> None:
        """Reject configurations that silently remove bounds or invalidate the protocol."""
        ranges = {
            "top_k": (1, 20),
            "examples": (0, 3),
            "max_output_tokens": (64, 4096),
            "context_byte_budget": (256, 200000),
            "max_search_calls": (1, 3),
            "max_repairs": (0, 1),
        }
        for field, (lower, upper) in ranges.items():
            value = getattr(self, field)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("invalid_experiment_" + field)
        if self.candidates not in {1, 3} or type(self.candidates) is not int:
            raise ValueError("candidates_must_be_1_or_3")
        if self.max_total_tokens <= 0 or self.max_request_byte_tokens <= 0:
            raise ValueError("budgets_must_be_positive")
        if not -1 <= self.min_score <= 1 or not 0 <= self.temperature <= 2:
            raise ValueError("invalid_score_or_temperature")

    def to_dict(self) -> dict:
        """Serialize only non-secret settings."""
        return asdict(self)


BASELINE = Experiment()
CONTEXT = replace(
    BASELINE, name="context", top_k=4, deduplicate=True, context_byte_budget=20000
)
SHORT = replace(CONTEXT, name="short_prompt", short_prompt=True, prompt_version="1.1")
ONE = replace(SHORT, name="one_example", examples=1, prompt_version="1.2")
COMPACT = replace(ONE, name="compact", max_output_tokens=800)
VARIANTS = {
    x.name: x
    for x in [
        BASELINE,
        CONTEXT,
        SHORT,
        ONE,
        COMPACT,
        replace(COMPACT, name="self_consistency", candidates=3),
        replace(COMPACT, name="react", max_search_calls=3),
    ]
}


def get_experiment(name: str, override: str | None = None) -> Experiment:
    """Load a named configuration or a saved development-selected configuration."""
    import json

    base = VARIANTS[name]
    if override:
        base = replace(base, **json.loads(Path(override).read_text()))
    return base
