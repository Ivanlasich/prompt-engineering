"""Canonical domain contracts, independent of the GigaChat wire protocol."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Citation(BaseModel):
    """A verbatim excerpt from a retrieved chunk."""

    model_config = ConfigDict(extra="forbid", strict=True)
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=4)


class Answer(BaseModel):
    """Stable public answer; operational failures live outside this contract."""

    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["ok", "not_found"]
    answer: str = Field(min_length=1)
    citations: list[Citation]

    @model_validator(mode="after")
    def coherent(self) -> "Answer":
        """Forbid unsupported success and misleading citations on abstention."""
        if self.status == "ok" and not self.citations:
            raise ValueError("ok requires citations")
        if self.status == "not_found" and self.citations:
            raise ValueError("not_found requires empty citations")
        return self


class SearchArgs(BaseModel):
    """Validated arguments for both search functions."""

    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=2000)
    year: int | None = Field(default=None, ge=1900, le=2100)
    session: str | None = Field(default=None, max_length=300)


def normalized(text: str) -> str:
    """Normalize harmless whitespace and ё for fact matching."""
    return re.sub(r"\s+", " ", text.casefold().replace("ё", "е")).strip()


def validate_grounding(answer: Answer, evidence: dict[str, dict]) -> None:
    """Check identifiers and verbatim quotations against supplied context only."""
    for citation in answer.citations:
        chunk = evidence.get(citation.chunk_id)
        if chunk is None:
            raise ValueError("citation_not_retrieved")
        # Only whitespace is relaxed: spelling and punctuation must remain verbatim.
        quote = " ".join(citation.quote.split())
        if quote not in " ".join(chunk["text"].split()):
            raise ValueError("quote_not_in_context")


def public_schema() -> dict:
    """Return an inlined schema supported by native response_format."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["ok", "not_found"]},
            "answer": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "chunk_id": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["chunk_id", "quote"],
                },
            },
        },
        "required": ["status", "answer", "citations"],
    }
