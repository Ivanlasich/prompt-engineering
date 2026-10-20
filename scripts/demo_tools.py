"""Live forced calls prove both native search tools and their continuation protocol."""

import json

from spief_rag.benchmark import load_tariff
from spief_rag.client import APIError, GigaChatClient
from spief_rag.contracts import Answer, public_schema, validate_grounding
from spief_rag.corpus import Corpus
from spief_rag.pipeline import parse_json
from spief_rag.prompts import answer_system
from spief_rag.search import SearchEngine, select_context, tool_specs
from spief_rag.settings import ROOT, VARIANTS, Settings
from spief_rag.telemetry import Ledger


def main():
    """Use the same question and grounded schema, recording observed costs."""
    client = GigaChatClient(Settings.from_env())
    rows = []
    engine = SearchEngine(Corpus.load(), client)
    question = "Кто были сомодераторами IT-завтрака на ПМЭФ-2016 и какие организации они представляли?"
    try:
        for name in ["semantic_search", "keyword_search"]:
            ledger = Ledger()
            row = {
                "tool": name,
                "selector": "forced",
                "question": question,
                "answer": None,
                "error": None,
            }
            try:
                history = [
                    {
                        "role": "system",
                        "content": answer_system(VARIANTS["short_prompt"]),
                    },
                    {"role": "user", "content": question},
                ]
                response = client.chat(
                    history,
                    ledger,
                    functions=tool_specs(),
                    function_call={"name": name},
                    max_tokens=250,
                )
                call = response["message"].get("function_call")
                if not call or call["name"] != name:
                    raise ValueError("wrong_tool")
                row["native_call"] = call
                row["functions_state_id_present"] = bool(
                    response["message"].get("functions_state_id")
                )
                hits = select_context(
                    engine.execute(name, call["arguments"], ledger, top_k=4),
                    deduplicate=True,
                    byte_budget=20000,
                )
                evidence = {r["chunk_id"]: r for r in hits}
                history.extend(
                    [
                        response["message"],
                        {
                            "role": "function",
                            "name": name,
                            "content": json.dumps({"hits": hits}, ensure_ascii=False),
                        },
                    ]
                )
                history.append(
                    {
                        "role": "user",
                        "content": "Верни окончательный ответ по источникам в JSON.",
                    }
                )
                final = client.chat(
                    history, ledger, schema=public_schema(), max_tokens=1000
                )
                answer = Answer.model_validate(parse_json(final["message"]["content"]))
                validate_grounding(answer, evidence)
                row["answer"] = answer.model_dump()
                row["evidence"] = hits
            except (APIError, ValueError) as exc:
                row["error"] = getattr(exc, "code", type(exc).__name__)
            row.update(events=ledger.events, usage=ledger.summary(load_tariff(client)))
            rows.append(row)
            print(
                json.dumps(
                    {
                        "tool": name,
                        "selector": "forced",
                        "status": (row["answer"] or {}).get("status"),
                        "error": row["error"],
                        "tokens": row["usage"]["total_tokens"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        (ROOT / "reports/native_tools_demo.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2)
        )
        client.close()


if __name__ == "__main__":
    main()
