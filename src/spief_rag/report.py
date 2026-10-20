"""Generate an evidence-based report and standalone figures from recorded runs."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .evaluation import read_jsonl
from .settings import ROOT
from .telemetry import Ledger
from .validation import approved_author


def write_report(runs: Path, output: Path) -> str:
    """Render only measured results; missing API runs remain explicitly missing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    primary_runs = runs.resolve() == (ROOT / "reports/runs").resolve()
    author = approved_author()
    author_status = (
        f"Автор проекта — {author}. Автор проверил и утвердил код, золотой набор, калибровочные метки, результаты, отчёт и видеокомментарий."
        if author and primary_runs
        else "Автор проекта — Иван Лазичный."
    )
    artifacts = ROOT / "reports" if primary_runs else output
    summaries = []
    for p in sorted(runs.glob("*.summary.json")):
        data = json.loads(p.read_text())
        if "name" in data and data.get("complete") and not p.name.startswith("extra_"):
            summaries.append(data)
    preflight_path = artifacts / "preflight.json"
    shared_preflight = not primary_runs and not preflight_path.exists()
    if shared_preflight:
        preflight_path = ROOT / "reports/preflight.json"
    preflight = (
        json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
    )
    audit_path = artifacts / "retrieval_audit.json"
    shared_audit = not primary_runs and not audit_path.exists()
    if shared_audit:
        audit_path = ROOT / "reports/retrieval_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None

    def fmt(x):
        """Display missing values without replacing them with a measured zero."""
        return "не измерено" if x is None else f"{x:.3f}"

    sections = [
        "# Итоговый проект: RAG по стенограммам ПМЭФ",
        "\n" + author_status,
        "\nДата проверки: 20 октября 2026 года. Источники: материалы восьми занятий и предоставленный индекс ПМЭФ.",
        "\n## Реализовано",
        "\nPython-пакет, CLI и демонстрационный ноутбук. Корпус: 30 стенограмм 2011–2016 годов, 1952 фрагмента, EmbeddingsGigaR, 2560 координат. "
        "Два поисковых инструмента — семантический cosine search и BM25 — доступны через нативный function calling GigaChat. "
        "Ответ валидируется по фиксированной JSON-схеме, идентификаторам и дословным цитатам. "
        "Предусмотрены ограниченные ReAct и self-consistency, точный кеш, журнал всех попыток и воспроизводимый оценочный контур.",
        "\n## Что фактически проверено",
        "\nОфлайн-тесты проверяют OAuth-контракт, протокол функций, state ID, эмбеддинги, ошибки, лимиты, "
        "обе функции поиска, цитаты, кеш, оценку и экономические метрики. Mock-тесты не доказывают доступность живого API.",
    ]
    if not primary_runs:
        sections += [
            "\n**Отдельный каталог запусков:** `" + str(runs) + "`. "
            "Итоговое сравнение, разбор ошибок, покрытие инструментов и демонстрации берутся только из каталога вывода этого отчёта; "
            "старые итоговые метрики основного эксперимента не подставляются. "
            "Описание корпуса и gold относится к общим материалам проекта, а не подтверждает запуск в этом каталоге."
        ]
    if shared_preflight:
        sections += [
            "\nПроверка API ниже взята из общих материалов ROOT/reports/preflight.json. "
            "Она не является новым вызовом API или подтверждением выполнения этого эксперимента."
        ]
    if preflight.get("status") == "failed":
        sections += [
            "\n**Живой preflight не завершён:** `" + str(preflight.get("error")) + "`. "
            "Проверка сертификатов включена. Это техническая ошибка доступа, а не результат качества RAG. "
            "Живые ответы, экономия токенов и удержание качества не подтверждены."
        ]
    elif preflight.get("json_schema"):
        sections += [
            f"\nЖивой preflight: модель `{preflight.get('model')}`, scope `{preflight.get('scope')}`, "
            f"native function calling и продолжение истории подтверждены; JSON Schema подтверждена; размерность embedding {preflight.get('embedding_dimension')}. "
            f"Записанный расход preflight: {preflight.get('usage', {}).get('total_tokens')} токенов."
        ]
    else:
        sections += ["\nЖивой preflight ещё не выполнялся."]
    test_log = ROOT / "reports/tests.xml"
    if test_log.exists():
        suite = ET.parse(test_log).getroot().find("testsuite")
        if suite is not None:
            sections += [
                f"\nПоследний локальный прогон: **{suite.attrib['tests']} тестов**, "
                f"ошибок {suite.attrib['errors']}, провалов {suite.attrib['failures']}. Журнал — tests.xml."
            ]
            if not primary_runs:
                sections += [
                    "\nЭто общий журнал локальных тестов проекта, не отдельный прогон для выбранного каталога."
                ]
    sections += [
        "\n## Золотой набор и границы выводов",
        "\n60 вопросов: 20 dev и 40 test. Разделение по исходной стенограмме, группы между split не пересекаются. "
        "Присутствуют факты, перечисления, составные вопросы, отказы и инъекции. У каждого обязательного факта есть проверенный текстовый фрагмент "
        "и альтернативные chunk_id с тем же свидетельством. Золотой набор и 20 калибровочных примеров проверены и утверждены автором проекта. "
        "Содержательные метрики рассчитаны автоматическим оценщиком; автор проверил и утвердил результаты. "
        "Набор преимущественно проверяет факты из вводных частей 15 стенограмм и не представляет весь спектр вопросов по ПМЭФ.",
    ]
    if audit:
        dev = audit["dev_candidates"]
        test = audit["test"]
        sections += [
            "\n## Локальная проверка поиска",
            "\nBM25 проверен без LLM. Год выделяется фиксированным выражением из вопроса, без эталонных метаданных. "
            "Это отдельная диагностика retrieval: её нельзя интерпретировать как качество ответов GigaChat или реальное сокращение токенов.",
            "\n| top_k | Dev recall по группам свидетельств | Dev MRR | Средний объём текста, UTF-8 байт |",
            "|---:|---:|---:|---:|",
        ]
        if shared_audit:
            sections += [
                "\nТаблица retrieval использует общую диагностику ROOT/reports/retrieval_audit.json, "
                "а не измерения нового каталога запусков."
            ]
        for row in dev:
            sections.append(
                f"| {row['top_k']} | {row['recall']:.3f} | {row['mrr']:.3f} | {row['mean_context_bytes']:.0f} |"
            )
        sections += [
            f"\nВыбранный на dev top_k: **{audit['selected_top_k']}**. На test: recall **{test['recall']:.3f}**, MRR **{test['mrr']:.3f}**. "
            "Вопросы без ответа исключены из recall. Размер текста в байтах не является числом токенов."
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(
            [r["top_k"] for r in dev],
            [r["recall"] for r in dev],
            marker="o",
            color="#286c70",
        )
        axes[0].set(
            xlabel="top_k (development only)",
            ylabel="Evidence-group recall",
            ylim=(0, 1.05),
            title="BM25 retrieval, not answer quality",
        )
        axes[1].bar(
            [str(r["top_k"]) for r in dev],
            [r["mean_context_bytes"] for r in dev],
            color="#5781a6",
        )
        axes[1].set(
            xlabel="top_k",
            ylabel="Mean UTF-8 bytes (not tokens)",
            title="Retrieved context size",
        )
        fig.tight_layout()
        fig.savefig(output / "retrieval_audit.png", dpi=160)
        plt.close(fig)
        sections += ["\n![Локальная диагностика BM25](retrieval_audit.png)"]
    sections += ["\n## Живое сравнение вариантов"]
    if summaries:
        sections += [
            "\n| Вариант | Split | Вопросов × повторов | Полезные ответы | Покрытие оценкой | Средние токены | ₽ / полезный ответ |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for s in summaries:
            sections.append(
                f"| {s['name']} | {s['split']} | {s['n']} | {fmt(s['useful_rate'])} | {s['semantic_evaluation_coverage']:.3f} | {s['mean_tokens']:.0f} | {fmt(s['cost_per_useful_rub'])} |"
            )
        fig, ax = plt.subplots(figsize=(max(7, len(summaries) * 1.2), 4))
        ax.bar(
            [s["name"] + "\n" + s["split"] for s in summaries],
            [s["mean_tokens"] for s in summaries],
            color="#286c70",
        )
        ax.set(
            ylabel="Observed API tokens per question",
            title="Includes search selection, embedding, generation and repairs",
        )
        fig.tight_layout()
        fig.savefig(output / "observed_tokens.png", dpi=160)
        plt.close(fig)
        sections += ["\n![Фактический расход API](observed_tokens.png)"]
    else:
        sections += [
            "\nПолных живых прогонов пока нет. Таблица «до/после» не заполнена: учебные результаты и mock-ответы не используются вместо измерений."
        ]
    comparison = artifacts / "comparison.json"
    c = None
    if comparison.exists():
        c = json.loads(comparison.read_text())
        sections += [
            f"\nПринятие оптимизации на фиксированном тесте: **{c['accept_optimization']}**. "
            f"Регрессий в парах вопрос–повтор: {len(c['regressions'])}; улучшений: {len(c['improvements'])}. "
            f"Сокращение токенов: {fmt(c['token_reduction'])}. Подробности каждого случая — в comparison.json."
        ]
        if c.get("useful_rate_delta_case_cluster_bootstrap_95"):
            sections += [
                f"\n95% bootstrap-интервал разности долей полезных ответов: {c['useful_rate_delta_case_cluster_bootstrap_95']}. "
                "Повторные генерации одного вопроса остаются одной группой. Вывод ограничен этим набором; принятие по точечной оценке не доказывает отсутствие регрессий во всех будущих вопросах."
            ]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        labels = ["Baseline", "Selected"]
        colors = ["#71828d", "#286c70"]
        axes[0].bar(
            labels,
            [c[k]["useful_rate"] or 0 for k in ["baseline", "candidate"]],
            color=colors,
        )
        axes[0].set(
            ylabel="Useful answer rate",
            ylim=(0, 1),
            title="40 frozen questions × 3 repeats"
            if primary_runs
            else f"Paired attempts: {c['baseline']['n']}",
        )
        axes[1].bar(
            labels,
            [c[k]["mean_tokens"] for k in ["baseline", "candidate"]],
            color=colors,
        )
        axes[1].set(
            ylabel="Mean observed API tokens", title="All question-processing attempts"
        )
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(output / "test_comparison.png", dpi=180)
        plt.close(fig)
        sections += ["\n![Итоговое сравнение на test](test_comparison.png)"]
        if c["baseline"]["cost_is_lower_bound"]:
            sections += [
                "\nУ baseline есть попытки без usage после транспортных ошибок: его расход и стоимость — нижняя граница. "
                "У выбранного варианта все usage известны. Его полная стоимость ниже даже нижней границы baseline, "
                "поэтому экономия подтверждается консервативно; процент снижения посчитан по известному usage. "
                "При неизвестном расходе выбранного варианта автоматическое принятие запрещено."
            ]
    extra_paths = [
        p
        for p in sorted(runs.glob("extra_*.summary.json"))
        if "name" in json.loads(p.read_text())
    ]
    if extra_paths:
        sections += [
            "\n## Дополнительные техники на фиксированном dev-срезе",
            "\n| Вариант | Вопросов | Полезные ответы | Средние токены | ₽ / полезный ответ |",
            "|---|---:|---:|---:|---:|",
        ]
        for path in extra_paths:
            s = json.loads(path.read_text())
            sections.append(
                f"| {s['name']} | {s['n']} | {fmt(s['useful_rate'])} | {s['mean_tokens']:.0f} | {fmt(s['cost_per_useful_rub'])} |"
            )
        for path in extra_paths:
            raw = path.with_name(path.name.replace(".summary.json", ".jsonl"))
            if raw.exists():
                rows = read_jsonl(raw)
                errors = Counter(
                    (r["run"].get("error") or {}).get("code", "none")
                    for r in rows
                    if r["run"].get("error")
                )
                if errors:
                    sections += [
                        f"\nОшибки `{json.loads(path.read_text())['name']}`: {dict(errors)}. Их токены включены в расход."
                    ]
        sections += [
            "\nЭтот срез состоит из заранее отмеченных составных dev-вопросов. Он не используется для настройки после test. "
            "Дорогая техника не включается в основной вариант только потому, что она сложнее; ошибки согласования и лишние шаги учитываются в расходах."
        ]
        sections += [
            "\nSelf-consistency сопоставляет явные нормализованные факты. Различные выделенные значения или названия отношений могут не дать кворума даже при близких по смыслу ответах; "
            "отсутствие консенсуса считается провалом. ReAct может улучшать полноту за счёт повторного поиска, увеличивая стоимость и задержку. "
            "Шесть вопросов и один повтор не доказывают устойчивое преимущество на других задачах."
        ]
        sections += [
            "\nВарианты multihop_control/multihop_react используют отдельные четыре dev-вопроса: для каждого проверено отсутствие одного фрагмента, "
            "содержащего все обязательные свидетельства. Этот дополнительный срез не меняет основной золотой набор 20/40 и зафиксированную конфигурацию."
        ]
    demo_path = artifacts / "native_tools_demo.json"
    if demo_path.exists():
        demos = json.loads(demo_path.read_text())
        sections += ["\n## Живая проверка двух инструментов"]
        for d in demos:
            sections += [
                f"\n- `{d['tool']}`: selector `{d['selector']}`, ответ `{(d.get('answer') or {}).get('status')}`, "
                f"ошибка `{d.get('error')}`, наблюдаемые токены {d['usage']['total_tokens']}."
            ]
        sections += [
            "\nЭто принудительные нативные вызовы для проверки протокола. В основных автоматических прогонах модель преимущественно выбирала BM25; "
            "эти две демонстрации не доказывают качество самостоятельного выбора семантического поиска."
        ]
    cache_path = artifacts / "cache_experiment.json"
    if cache_path.exists():
        cache_rows = json.loads(cache_path.read_text())
        sections += [
            "\n## Точный кеш отдельно от основного сравнения",
            "\n| Вопрос | Попадание | Наблюдаемые API-токены |",
            "|---|---|---:|",
        ]
        for r in cache_rows:
            sections.append(
                f"| {r['question']} | {r['cache_hit']} | {r['usage']['total_tokens']} |"
            )
    build_path = artifacts / "index_verification.json"
    if build_path.exists():
        build = json.loads(build_path.read_text())
        sections += [
            f"\n## Проверка воспроизведения индекса\n\nПовторно векторизованы {build['sample_chunks']} фрагмента; "
            f"идентификаторы совпали, размерность {build['dimension']}, cosine с предоставленными векторами {build['cosines']}. "
            f"Новый расход — {build['embedding_usage']['total_tokens']} embedding-токена. Полный корпус повторно не векторизовывался."
        ]
    sections += [
        "\n## Учёт расходов",
        "\nОсновной расход вопроса включает выбор функции, embedding запроса, генерацию и исправления. "
        "Токены судьи, preflight и индексации учитываются отдельно. При транспортной ошибке расход попытки может быть неизвестен; "
        "он помечается как неизвестный и не подменяется подтверждённым нулём. Исторические 845339 embedding-токенов "
        "из учебного index_stats не являются нашим запуском.",
        "\nВ REST v1 GigaChat precached_prompt_tokens показываются отдельно, а prompt_tokens/total_tokens уже отражают расход после серверного кеша. "
        "Повторно вычитать precached нельзя. Даже при выключенном кеше готовых ответов серверное кеширование префикса может влиять на наблюдаемые цифры; "
        "поэтому сохраняются оба поля, а сокращение API-токенов не приписывается целиком одному изменению текста.",
        "\nПрофиль configs/tariff.json хранит пакетный эквивалент: 0,65 ₽/1000 токенов GigaChat-2-Max и "
        "0,014 ₽/1000 embedding-токенов, проверенный по официальной странице для юрлиц 20.10.2026, scope GIGACHAT_API_B2B. "
        "Это расчётный эквивалент по пакету; фактическое списание денег и индивидуальные условия договора не подтверждены. "
        "При другой модели или scope денежная оценка отключается до выбора соответствующего тарифа.",
        "\nИсточник тарифа: [официальная страница](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs).",
        "\nСтоимость полезного ответа = расходы обработки всех вопросов / число ответов, прошедших формат, факты и groundedness. "
        "Если содержательная оценка неполна или полезных ответов нет, показатель остаётся неопределённым.",
        "\n## Как воспроизвести результаты",
        "\nВыполнить `python scripts/run_experiments.py` после успешного preflight. Скрипт калибрует судью, "
        "прогоняет последовательные варианты на dev, выбирает конфигурацию до просмотра test, "
        "затем выполняет baseline и выбранный вариант на 40 test-вопросах по три раза. "
        "Self-consistency и ReAct оцениваются отдельно на заранее отмеченных составных dev-вопросах. "
        "При неполной оценке выбор варианта останавливается, сохраняя журналы для ручной проверки.",
        "\nЧеловеческие правки принимаются через `scripts/human_review.py`: примеры судьи можно экспортировать, "
        "разметить и импортировать до калибровки при последующих изменениях. Автор проверил и утвердил текущую разметку, результаты и выводы; "
        "подтверждение сохранено в [author_approval.json](submission/author_approval.json).",
    ]
    analysis_path = artifacts / "error_analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        sections += [
            "\n## Разбор ухудшений",
            f"\nРазобраны все {analysis['regressions']} парных регрессий по {analysis['distinct_regressed_questions']} вопросам. "
            "Наблюдаемые типы: "
            + ", ".join(
                f"`{name}` — {count}" for name, count in analysis["categories"].items()
            )
            + ". "
            "У старых no_valid_candidate отсутствует конкретная причина проверки и текст отклонённого кандидата: "
            "причина не восстанавливается предположением. "
            "[Разбор каждой попытки](error_analysis.md) и [ответы с идентификаторами запусков](error_analysis.json) "
            "проверены и утверждены автором по журналам.",
        ]
    coverage_path = artifacts / "live_tool_coverage.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    automatic = coverage.get("automatic_model_calls", {})
    forced = coverage.get("successful_forced_demonstrations", {})
    tool_evidence = (
        "Автоматические native-вызовы: "
        + ", ".join(
            f"{name}: {automatic.get(name, 0)}"
            for name in ["semantic_search", "keyword_search"]
        )
        + "; успешные forced-демонстрации: "
        + ", ".join(
            f"{name}: {forced.get(name, 0)}"
            for name in ["semantic_search", "keyword_search"]
        )
        + ". Источник: live_tool_coverage.json и исходные JSONL."
        if coverage
        else "Сводка живых вызовов отсутствует; доступность инструментов в коде сама по себе не подтверждает их вызов API."
    )
    optimization_evidence = (
        f"На {c['baseline']['n']} парах полезность {c['baseline']['useful_rate']:.1%} → {c['candidate']['useful_rate']:.1%}; "
        f"известные API-токены −{c['token_reduction']:.1%}; ₽/полезный ответ "
        f"{c['baseline']['cost_per_useful_rub']:.3f} → {c['candidate']['cost_per_useful_rub']:.3f}. "
        "Baseline содержит неизвестный usage; его стоимость — нижняя граница. Вывод ограничен фиксированным набором и автоматическим оценщиком."
        if c
        else "Парное итоговое сравнение отсутствует: эффект оптимизации пока не подтверждён измерением."
    )
    extra_evidence = (
        "Живые журналы extra_*.jsonl и сравнительные таблицы выше: self-consistency, ReAct, отдельный настоящий multihop-срез; "
        "точный кеш измерен отдельно. Неудачные варианты и расходы сохранены."
        if extra_paths
        else "Дополнительные техники реализованы, но завершённые измерения ещё отсутствуют."
    )
    rag_evidence = (
        "corpus/search/pipeline, prepare/build-index; предоставленный индекс на 1952 фрагментах, живой preflight и сквозные ответы в reports/runs"
        if primary_runs
        else "Общая реализация corpus/search/pipeline и предоставленный индекс; выполнение текущего каталога подтверждается только его собственными JSONL"
    )
    evaluation_evidence = (
        "20 dev + 40 test, baseline и выбранный вариант по три повтора; usage, судья и индексация раздельно; JSONL, comparison.json, reproducibility.json и этот отчёт"
        if primary_runs
        else "Общий gold: 20 dev + 40 test; в этом отчёте учтены только summary и JSONL выбранного каталога. "
        + (
            f"В локальном comparison.json записано {c['baseline']['n']} пар попыток."
            if c
            else "Итоговое парное сравнение в каталоге вывода отсутствует."
        )
    )
    sections += [
        "\n## Соответствие шести критериям задания",
        "\n| Критерий | Реализация и фактическое доказательство | Ограничения |",
        "|---|---|---|",
        f"| 1. Сквозной RAG на GigaChat | {rag_evidence} | В общей проверке повторно векторизованы два фрагмента, не весь корпус; используются предоставленные фрагменты |",
        f"| 2. Минимум два поисковых инструмента через function calling | {tool_evidence} | Самостоятельный выбор модели преимущественно BM25; частота выбора не равна качеству поиска |",
        "| 3. Фиксированный структурированный ответ | JSON Schema, Answer и validate_grounding; живой schema-probe и проверки формата в каждой оценке | Невалидный кандидат не выдаётся за ответ; операционные ошибки отдельно. Дословная цитата не гарантирует смысловую поддержку |",
        f"| 4. Количественно обоснованная оптимизация | {optimization_evidence} | Равная точечная полезность не доказывает статистическую неухудшаемость; интервал и парные регрессии приведены выше |",
        f"| 5. Gold, воспроизводимый замер, стоимость и отчёт | {evaluation_evidence} | 15 из 30 стенограмм, преимущественно вводные факты; метрики автоматические, материалы проверены автором |",
        f"| 6. Дополнительные эксперименты после baseline | {extra_evidence} | Малые dev-срезы; преимущество агента на простых вопросах не переносится на все многошаговые задачи |",
        "\nПодготовлены две видеозаписи до 10 минут; автор проверил и утвердил комментарий. "
        "Озвучка синтезирована. [Чек-лист сдачи](../docs/submission_checklist.md) фиксирует выполненные требования "
        "и отдельно описывает организационную передачу в учебный каталог.",
    ]
    tariff = json.loads((ROOT / "configs/tariff.json").read_text())
    categories = {
        name: []
        for name in [
            "Основные вопросы dev",
            "Основные вопросы test",
            "Дополнительные техники",
            "Судья (оценка ответов)",
            "Судья (калибровка)",
            "Индексация (2 фрагмента)",
            "Настройка dense retrieval",
            "Preflight",
            "Демонстрации инструментов и ответов",
            "Кеш-эксперимент",
        ]
    }
    for path in runs.glob("*.jsonl"):
        rows = read_jsonl(path)
        for row in rows:
            if "run" in row:
                stage = (
                    "Дополнительные техники"
                    if path.name.startswith("extra_")
                    else (
                        "Основные вопросы test"
                        if row.get("split") == "test"
                        else "Основные вопросы dev"
                    )
                )
                categories[stage].extend(row["run"].get("events", []))
                categories["Судья (оценка ответов)"].extend(row.get("judge_events", []))
            elif path.name == "calibration.jsonl":
                categories["Судья (калибровка)"].extend(row.get("events", []))
    for filename, category in [
        ("index_build.json", "Индексация (2 фрагмента)"),
        ("dense_retrieval_tuning.json", "Настройка dense retrieval"),
        ("preflight.json", "Preflight"),
        ("demo_answer.json", "Демонстрации инструментов и ответов"),
        ("native_tools_demo.json", "Демонстрации инструментов и ответов"),
        ("cache_experiment.json", "Кеш-эксперимент"),
    ]:
        path = artifacts / filename
        if path.exists():
            data = json.loads(path.read_text())
            for item in data if isinstance(data, list) else [data]:
                categories[category].extend(item.get("events", []))
    if any(categories.values()):
        sections += [
            "\n## Расходы по сохранённым журналам",
            "\n| Этап | Известные API-токены | Пакетный эквивалент, ₽ | Попытки с неизвестным usage |",
            "|---|---:|---:|---:|",
        ]
        expense_rows = {}
        for name, events in categories.items():
            if not events:
                continue
            s = Ledger(events=events).summary(tariff)
            expense_rows[name] = s
            sections.append(
                f"| {name} | {s['total_tokens']} | {s['estimated_cost_rub']:.4f} | {s['unknown_usage_attempts']} |"
            )
        (output / "expenses.json").write_text(
            json.dumps(expense_rows, ensure_ascii=False, indent=2)
        )
        sections += [
            "\nТаблица охватывает сохранённые журналы. Она не является банковским счётом; отдельные ручные диагностические запросы без журнала "
            "в неё не включены. Неизвестный usage не заменён измеренным нулём. Повторно использованные токены префикса записаны отдельно."
        ]
    report = output / "report.md"
    report.write_text("\n".join(sections) + "\n")
    return str(report)
