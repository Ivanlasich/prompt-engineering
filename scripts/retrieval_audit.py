"""Reproducible local BM25 audit; never presented as an LLM benchmark."""

import json
from statistics import mean
from types import SimpleNamespace

from spief_rag.corpus import Corpus
from spief_rag.evaluation import read_jsonl
from spief_rag.search import SearchEngine, select_context
from spief_rag.settings import ROOT
from spief_rag.telemetry import Ledger


def main():
    """Choose lexical top_k only on dev; test it once and retain per-case evidence."""
    corpus = Corpus.load()
    engine = SearchEngine(corpus, SimpleNamespace())
    dev = read_jsonl(ROOT / "data/gold/dev.jsonl")
    test = read_jsonl(ROOT / "data/gold/test.jsonl")

    def audit(cases, k):
        """Measure one lexical top-k setting against annotated evidence groups."""
        rows = []
        for case in cases:
            import re

            match = re.search(r"ПМЭФ[- ‑–—]*(20\d{2})", case["question"])
            arguments = {"query": case["question"]}
            if match:
                arguments["year"] = int(match.group(1))
            hits = engine.execute("keyword_search", arguments, Ledger(), top_k=k)
            selected = select_context(hits, deduplicate=True, byte_budget=20000)
            ids = [h["chunk_id"] for h in hits]
            groups = [set(f["evidence_ids"]) for f in case["required_facts"]]
            recall = mean(bool(g & set(ids)) for g in groups) if groups else None
            ranks = [i + 1 for i, x in enumerate(ids) if any(x in g for g in groups)]
            rows.append(
                {
                    "case_id": case["id"],
                    "k": k,
                    "recall": recall,
                    "mrr": 1 / min(ranks) if ranks else (0 if groups else None),
                    "raw_ids": ids,
                    "selected_ids": [h["chunk_id"] for h in selected],
                    "raw_context_bytes": sum(len(h["text"].encode()) for h in hits),
                    "selected_context_bytes": sum(
                        len(h["text"].encode()) for h in selected
                    ),
                }
            )
        applicable = [r for r in rows if r["recall"] is not None]
        return {
            "top_k": k,
            "recall": mean(r["recall"] for r in applicable),
            "mrr": mean(r["mrr"] for r in applicable),
            "answerable_cases": len(applicable),
            "mean_context_bytes": mean(r["raw_context_bytes"] for r in rows),
            "rows": rows,
        }

    candidates = [audit(dev, k) for k in [2, 4, 8, 12]]
    best = max(candidates, key=lambda r: (r["recall"], -r["top_k"]))
    output = {
        "kind": "local_lexical_retrieval_only_not_LLM_quality_or_token_measurement",
        "filter_policy": "year extracted from user question by a fixed regex; no gold metadata used by retrieval",
        "selection_split": "dev",
        "corpus_version": corpus.version,
        "dev_candidates": candidates,
        "selected_top_k": best["top_k"],
        "test": audit(test, best["top_k"]),
    }
    (ROOT / "reports/retrieval_audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {k: v for k, v in output.items() if k not in ["dev_candidates", "test"]}
        )
    )
    print("dev recall", best["recall"], "test recall", output["test"]["recall"])


if __name__ == "__main__":
    main()
