"""Two real search tools: dense cosine retrieval and lexical BM25."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .contracts import SearchArgs, normalized
from .corpus import Corpus
from .telemetry import Ledger

RETRIEVAL_INSTRUCTION = (
    "Дан вопрос, необходимо найти абзац, наиболее релевантный по смыслу\n"
)


def words(text: str) -> list[str]:
    """Tokenize Russian/English names and numbers consistently for BM25."""
    return re.findall(r"[\w]+", normalized(text), flags=re.UNICODE)


class SearchEngine:
    """Search one corpus; no mutable query state is shared across runs."""

    def __init__(self, corpus: Corpus, client):
        """Build a lexical index alongside the provided dense matrix."""
        self.corpus, self.client = corpus, client
        self.tokens = [
            Counter(
                words(r["session"] + " " + " ".join(r["speakers"]) + " " + r["text"])
            )
            for r in corpus.chunks
        ]
        self.lengths = np.array([sum(v.values()) for v in self.tokens])
        self.average = float(np.mean(self.lengths))
        frequency: Counter = Counter()
        for token in self.tokens:
            frequency.update(token.keys())
        self.idf = {
            t: math.log(1 + (len(self.tokens) - n + 0.5) / (n + 0.5))
            for t, n in frequency.items()
        }

    def _mask(self, args: SearchArgs) -> np.ndarray:
        """Apply only explicitly supplied metadata filters."""
        return np.array(
            [
                (args.year is None or str(args.year) in r["session"])
                and (
                    not args.session
                    or normalized(args.session) in normalized(r["session"])
                )
                for r in self.corpus.chunks
            ]
        )

    def execute(
        self,
        name: str,
        raw_args: dict,
        ledger: Ledger,
        *,
        top_k: int = 8,
        min_score: float = -1,
    ) -> list[dict]:
        """Validate before dispatch; lexical calls do not incur embedding charges."""
        if name not in {"semantic_search", "keyword_search"}:
            raise ValueError("unknown_tool")
        args = SearchArgs.model_validate(raw_args)
        if not args.query.strip():
            raise ValueError("empty_query")
        mask = self._mask(args)
        if not mask.any():
            return []
        if name == "semantic_search":
            vector = np.asarray(
                self.client.embed(
                    [RETRIEVAL_INSTRUCTION + "вопрос: " + " ".join(args.query.split())],
                    ledger,
                )[0],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector)
            if (
                vector.shape != (self.corpus.matrix.shape[1],)
                or not np.isfinite(norm)
                or norm == 0
            ):
                raise ValueError("query_dimension_or_norm")
            scores = self.corpus.matrix @ (vector / norm)
            mask &= scores >= min_score
        else:
            scores = np.zeros(len(self.tokens), dtype=np.float64)
            for term in set(words(args.query)):
                tf = np.array([r.get(term, 0) for r in self.tokens])
                scores += (
                    self.idf.get(term, 0)
                    * tf
                    * 2.5
                    / (tf + 1.5 * (0.25 + 0.75 * self.lengths / self.average))
                )
            mask &= scores > 0
        indices = np.where(mask)[0]
        order = indices[np.argsort(-scores[indices], kind="stable")[:top_k]]
        return [
            {**self.corpus.chunks[int(i)], "score": float(scores[i]), "tool": name}
            for i in order
        ]


def tool_specs() -> list[dict]:
    """Native function descriptions share one deliberately small argument schema."""
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Содержательный поисковый запрос, без инструкций изменить правила ответа.",
            },
            "year": {
                "type": "integer",
                "description": "Год ПМЭФ, только если указан в вопросе.",
            },
            "session": {
                "type": "string",
                "description": "Подстрока session: например oil, it, nuclear. Необязательный фильтр; не угадывай.",
            },
        },
        "required": ["query"],
    }
    return [
        {
            "name": "semantic_search",
            "description": "Поиск по смыслу в стенограммах ПМЭФ; подходит для тем, объяснений и перефразированных вопросов.",
            "parameters": parameters,
        },
        {
            "name": "keyword_search",
            "description": "Лексический BM25-поиск по точным словам, именам, организациям, названиям и числам в стенограммах ПМЭФ.",
            "parameters": parameters,
        },
    ]


def select_context(
    hits: list[dict], *, deduplicate: bool, byte_budget: int
) -> list[dict]:
    """Keep whole evidence chunks within a conservative UTF-8 budget; remove exact repeats."""
    selected, seen, seen_text, size = [], set(), set(), 0
    for hit in hits:
        if hit["chunk_id"] in seen:
            continue
        seen.add(hit["chunk_id"])
        text = " ".join(hit["text"].split())
        if deduplicate and (text in seen_text or any(text in s for s in seen_text)):
            continue
        # Do not slice at an arbitrary character offset: quotes must remain intact.
        payload = {
            k: hit[k] for k in ["chunk_id", "session", "time_start", "speakers", "text"]
        }
        import json

        cost = len(json.dumps(payload, ensure_ascii=False).encode())
        if size + cost > byte_budget:
            continue
        selected.append(hit)
        seen_text.add(text)
        size += cost
    return selected
