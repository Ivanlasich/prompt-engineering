"""Tune dense retrieval only on development questions; never call the generator."""

import json
import re
from statistics import mean

import numpy as np

from spief_rag.benchmark import load_tariff
from spief_rag.client import GigaChatClient
from spief_rag.corpus import Corpus
from spief_rag.evaluation import read_jsonl
from spief_rag.search import RETRIEVAL_INSTRUCTION
from spief_rag.settings import ROOT, Settings
from spief_rag.telemetry import Ledger


def main():
    """Embed each dev question once; evaluate all thresholds against frozen evidence groups."""
    client = GigaChatClient(Settings.from_env())
    ledger = Ledger()
    output = ROOT / "reports/dense_retrieval_tuning.json"
    try:
        corpus = Corpus.load()
        cases = read_jsonl(ROOT / "data/gold/dev.jsonl")
        vectors = np.asarray(
            client.embed(
                [RETRIEVAL_INSTRUCTION + "вопрос: " + c["question"] for c in cases],
                ledger,
            ),
            dtype=np.float32,
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        scores = vectors @ corpus.matrix.T
        candidates = []
        for top_k in [2, 4, 8, 12]:
            for threshold in [-1, 0.35, 0.5, 0.65]:
                recalls = []
                byte_counts = []
                for case, s in zip(cases, scores):
                    match = re.search(r"ПМЭФ[- ‑–—]*(20\d{2})", case["question"])
                    mask = np.array(
                        [
                            not match or match.group(1) in r["session"]
                            for r in corpus.chunks
                        ]
                    )
                    ids = np.where(mask & (s >= threshold))[0]
                    indices = ids[np.argsort(-s[ids], kind="stable")[:top_k]]
                    found = {corpus.chunks[int(i)]["chunk_id"] for i in indices}
                    if case["required_facts"]:
                        recalls.append(
                            mean(
                                bool(found & set(f["evidence_ids"]))
                                for f in case["required_facts"]
                            )
                        )
                    byte_counts.append(
                        sum(
                            len(corpus.chunks[int(i)]["text"].encode()) for i in indices
                        )
                    )
                candidates.append(
                    {
                        "top_k": top_k,
                        "min_score": threshold,
                        "recall": mean(recalls),
                        "mean_context_bytes": mean(byte_counts),
                    }
                )
        chosen = max(
            candidates,
            key=lambda r: (r["recall"], -r["mean_context_bytes"], -r["min_score"]),
        )
        result = {
            "selection_split": "dev",
            "input": "literal user questions; agent-rewritten queries may differ",
            "candidates": candidates,
            "best_retrieval_only": chosen,
            "warning": "Retrieval tuning is diagnostic. Final pipeline is chosen by end-to-end dev quality and cost, not this recall alone.",
            "usage": ledger.summary(load_tariff(client)),
            "events": ledger.events,
        }
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(chosen))
    finally:
        if not output.exists():
            output.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "usage": ledger.summary(load_tariff(client)),
                        "events": ledger.events,
                    },
                    indent=2,
                )
            )
        client.close()


if __name__ == "__main__":
    main()
