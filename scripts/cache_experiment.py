"""Measure exact-cache hits separately from uncached quality comparisons."""

import json

from spief_rag.benchmark import load_tariff
from spief_rag.cache import ExactCache
from spief_rag.client import GigaChatClient
from spief_rag.corpus import Corpus
from spief_rag.pipeline import Pipeline
from spief_rag.search import SearchEngine
from spief_rag.settings import ROOT, VARIANTS, Settings


def main():
    """Use a dedicated cache namespace; repeated and changed-year questions differ."""
    client = GigaChatClient(Settings.from_env())
    try:
        pipeline = Pipeline(
            SearchEngine(Corpus.load(), client),
            VARIANTS["compact"],
            load_tariff(client),
            ExactCache(ROOT / ".cache/cache-experiment"),
        )
        questions = [
            "Кто модерировал IT-завтрак ПМЭФ-2016?",
            "Кто модерировал IT-завтрак ПМЭФ-2016?",
            "Кто модерировал IT-завтрак ПМЭФ-2014?",
        ]
        rows = [pipeline.run(q) for q in questions]
        (ROOT / "reports/cache_experiment.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2)
        )
        print(
            json.dumps(
                [
                    {
                        "question": r["question"],
                        "cache_hit": r["cache_hit"],
                        "tokens": r["usage"]["total_tokens"],
                        "error": r["error"],
                    }
                    for r in rows
                ],
                ensure_ascii=False,
            )
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
