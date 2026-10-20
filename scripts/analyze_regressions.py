"""Explain recorded paired regressions without regenerating or relabeling answers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from spief_rag.evaluation import read_jsonl
from spief_rag.settings import ROOT


def diagnose(row: dict) -> tuple[str, str, str]:
    """Separate observable failures from causes absent in historical diagnostics."""
    run, score = row["run"], row["score"]
    answer = run.get("answer")
    error = (run.get("error") or {}).get("code")
    if error == "no_valid_candidate":
        coverage = score.get("context_recall")
        retrieval = (
            "Эталонные свидетельства присутствовали в контексте. "
            if coverage == 1
            else "Эталонные свидетельства в контекст не попали. "
        )
        return (
            "validation_opaque",
            retrieval
            + "Кандидат и исправление отклонены валидатором; сохранён только no_valid_candidate. "
            "Старый журнал не содержит текста отклонённого ответа и конкретной причины проверки. "
            "Нельзя установить, была ли проблема в JSON, дословности цитаты, источнике или другом ограничении.",
            "Для новых запусков сохранять безопасные коды проверок и finish_reason; отдельно проверять поиск. "
            "Не приписывать старой попытке причину, которой нет в её журнале.",
        )
    if answer and answer["status"] == "not_found":
        retrieval = [x for x in run["trace"] if x["type"] == "retrieval"]
        empty = not any(x["raw_hit_ids"] for x in retrieval)
        return (
            "false_refusal",
            (
                "Поиск вернул пустую выдачу после переданного моделью фильтра session; приложение сформировало not_found. "
                if empty
                else "Поиск вернул фрагменты, но без эталонного свидетельства; модель ответила not_found. "
            )
            + "Вопрос имеет ответ в корпусе по размеченным источникам, поэтому отказ ошибочен для сквозного пайплайна. "
            "Одна эта попытка не доказывает подчинение инъекции.",
            "Проверить соответствие пользовательских названий служебным метаданным сессий и повторный поиск без избыточного фильтра; "
            "для непустой нерелевантной выдачи проверить переформулировку запроса на dev.",
        )
    if score.get("supported") is False:
        return (
            "unsupported_relation",
            "Текст ответа содержит ожидаемые имя и организацию, но приведённые цитаты не устанавливают запрошенную связь между ними. "
            "Проверка подстроки цитаты и regex фактов проходит, содержательный судья отвергает подтверждённость. "
            "Это регрессия обоснования, даже если сам факт совпал с эталоном.",
            "Искать фрагмент с явной связью человека и организации; проверять смысл цитаты, а не только её дословность. "
            "Независимо проверить решение судьи.",
        )
    detail = (
        "В данном ответе служебное имя cybersecurity принято за название панели; цитаты обсуждают другую информацию. "
        "Ожидаемое название «Построение эффективных систем кибербезопасности» в найденный контекст не попало."
        if answer and "cybersecurity" in answer["answer"]
        else "Сохранённые показатели не доказывают более конкретную причину ошибки."
    )
    return (
        "incorrect_fact_with_valid_quotes",
        "Схема и дословность цитат корректны, но обязательный факт отсутствует. "
        + detail,
        "Отделять служебное имя сессии от её названия в стенограмме; добавлять поиск по названию мероприятия. "
        "Не считать наличие настоящей цитаты достаточным обоснованием ответа.",
    )


def analyze(baseline: Path, candidate: Path) -> dict:
    """Pair identical question/repeat keys and retain original automated judgments."""
    left = {(r["case_id"], r["repeat"]): r for r in read_jsonl(baseline)}
    right = {(r["case_id"], r["repeat"]): r for r in read_jsonl(candidate)}
    if left.keys() != right.keys():
        raise ValueError("paired_run_keys_mismatch")
    regressions, improvements = [], 0
    for key, before in left.items():
        after = right[key]
        b, a = before["score"]["useful_auto"], after["score"]["useful_auto"]
        if b is None or a is None:
            raise ValueError("incomplete_usefulness_judgments")
        improvements += int(not b and a)
        if not b or a:
            continue
        category, explanation, follow_up = diagnose(after)
        regressions.append(
            {
                "case_id": key[0],
                "repeat_index": key[1],
                "question": after["run"]["question"],
                "baseline_run_id": before["run"]["run_id"],
                "candidate_run_id": after["run"]["run_id"],
                "baseline_answer": before["run"]["answer"],
                "candidate_answer": after["run"]["answer"],
                "candidate_error": after["run"]["error"],
                "candidate_tool_calls": [
                    x for x in after["run"]["trace"] if x["type"] == "tool_call"
                ],
                "candidate_context_recall": after["score"]["context_recall"],
                "candidate_judgment": after["score"]["judgment"],
                "category": category,
                "observed_explanation": explanation,
                "follow_up": follow_up,
            }
        )
    return {
        "provenance": "analysis of recorded automated judgments",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [baseline, candidate]
        },
        "paired_attempts": len(left),
        "regressions": len(regressions),
        "distinct_regressed_questions": len({r["case_id"] for r in regressions}),
        "improvements": improvements,
        "categories": dict(Counter(r["category"] for r in regressions)),
        "repeat_index_note": "Original logs use 0, 1, 2; the narrative displays repetitions 1, 2, 3.",
        "cases": regressions,
    }


def main() -> None:
    """Write a reproducible local analysis; leave source logs and labels untouched."""
    result = analyze(
        ROOT / "reports/runs/test_baseline.jsonl",
        ROOT / "reports/runs/test_selected.jsonl",
    )
    (ROOT / "reports/error_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# Разбор парных регрессий baseline → short_prompt",
        "",
        "Автор: Иван Лазичный. Разбор сохранённых журналов и исходных автоматических оценок. "
        "Ответы, метки полезности и измеренные расходы сохранены. "
        "Хеши источников, исходные ответы, цитаты, tool arguments и идентификаторы запусков сохранены в [error_analysis.json](error_analysis.json).",
        "",
        f"В {result['paired_attempts']} парах вопрос–повтор: {result['regressions']} регрессий по {result['distinct_regressed_questions']} вопросам "
        f"и {result['improvements']} улучшений. Одинаковые 70% полезных ответов скрывают разные ошибки. "
        "Номер повтора ниже — 1–3; в сырых JSONL индексы 0–2.",
        "",
        f"Отклонённые кандидаты с недостаточной исторической диагностикой — {result['categories'].get('validation_opaque', 0)}; "
        f"ошибочные отказы после поиска — {result['categories'].get('false_refusal', 0)}; "
        f"неподтверждённая связь фактов — {result['categories'].get('unsupported_relation', 0)}; "
        f"неверный факт при настоящих, но нерелевантных цитатах — {result['categories'].get('incorrect_fact_with_valid_quotes', 0)}. "
        "Наличие ошибки поиска в той же попытке не устанавливает причину ошибки валидации.",
        "",
        "| Вопрос / повтор | Наблюдаемый тип | Эталонные свидетельства в контексте |",
        "|---|---|---:|",
    ]
    for row in result["cases"]:
        lines.append(
            f"| {row['case_id']} / {row['repeat_index'] + 1} | {row['category']} | {row['candidate_context_recall']:.1f} |"
        )
    for row in result["cases"]:
        answer = row["candidate_answer"]
        lines.extend(
            [
                "",
                f"## {row['case_id']}, повтор {row['repeat_index'] + 1}",
                "",
                row["question"],
                "",
                "Baseline: " + row["baseline_answer"]["answer"],
                "",
                "Выбранный вариант: "
                + (
                    answer["answer"]
                    if answer
                    else f"ответ не принят, ошибка `{row['candidate_error']['code']}`."
                ),
                "",
                row["observed_explanation"],
                "",
                "Следующая проверка: " + row["follow_up"],
            ]
        )
        if row["candidate_judgment"]:
            lines.extend(
                [
                    "",
                    "Записанное объяснение модельного судьи: «"
                    + row["candidate_judgment"]["reason"]
                    + "».",
                ]
            )
        if answer and answer["citations"]:
            lines.extend(["", "Цитаты выбранного варианта:", ""])
            lines.extend(
                f"- `{c['chunk_id']}`: «{c['quote']}»" for c in answer["citations"]
            )
    lines.extend(
        [
            "",
            "## Что можно заключить",
            "",
            "Сокращение контекста и инструкции уменьшило известный расход API, но не устранило ошибок поиска и обоснования. "
            "По старым no_valid_candidate нельзя честно восстановить причину отклонения. Для следующих опытов нужна подробная безопасная диагностика. "
            "Предлагаемые исправления поиска и проверки цитат ещё не измерены этим сравнением: применять их как проверенное улучшение без нового dev-замера нельзя. "
            "Просмотр test-регрессий служит отчётности; последующую настройку следует проводить на dev, а для нового итогового вывода использовать новый независимый тест.",
        ]
    )
    (ROOT / "reports/error_analysis.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "paired_attempts",
                    "regressions",
                    "distinct_regressed_questions",
                    "improvements",
                    "categories",
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
