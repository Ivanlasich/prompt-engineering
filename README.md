# ПМЭФ: RAG-ассистент на GigaChat

Итоговый проект по восьми занятиям курса «Продвинутые техники промптинга и токенизации LLM».
GigaChat выбирает поиск по смыслу или BM25, приложение исполняет функцию, модель отвечает фиксированным JSON с проверяемыми цитатами.
Отдельный контур сравнивает качество, токены и стоимость полезного ответа.

Автор проекта — Иван Лазичный. Автор проверил и утвердил код, золотой набор, калибровочные метки, результаты, отчёт и видеокомментарий. Подтверждение автора сохранено в [author_approval.json](reports/submission/author_approval.json). Численные показатели рассчитаны оценочным контуром и проверены автором.

## Быстрый запуск

Python 3.12+, команды из корня репозитория. Проверенная среда — Python 3.12.14; зафиксированный `numpy` требует Python 3.12:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
spief-rag validate-gold
pytest -q
spief-rag inspect
```

Без `uv`: `python3.12 -m venv .venv`, затем `python -m pip install -r requirements.lock` и `python -m pip install -e .` внутри окружения.
`requirements.lock` фиксирует проверенное окружение; pyproject.toml описывает допустимые диапазоны.

Задать `GIGACHAT_CREDENTIALS` в окружении или скопировать `.env.example` в `.env.local` и заполнить локально.
Существующий `.env.local` не перезаписывать. Реальные env-файлы игнорируются Git; credential не принимается аргументом CLI и не попадает в результаты.
Для предоставленного учебного ключа подтверждён scope `GIGACHAT_API_B2B`. Для другого проекта scope указывается по его настройкам.

```bash
spief-rag preflight
spief-rag ask "Кто были сомодераторами IT-завтрака на ПМЭФ-2016 и какие организации они представляли?" --output reports/demo_answer.json
```

`preflight` делает реальные вызовы: список моделей, короткий ответ, безвредная forced-функция с продолжением истории, JSON Schema и один embedding.
При ошибке код возврата ненулевой; безопасные подробности — в `reports/preflight.json`.
По умолчанию `ask` и живая демонстрация ноутбука читают `configs/selected_config.json`: выбранный на dev вариант `short_prompt`. `--variant` явно переключает эксперимент; `--config` задаёт сохранённую конфигурацию.

Модель — `GigaChat-2-Max`, API — `https://api.giga.chat/v1`, OAuth — `https://ngw.devices.sberbank.ru:9443/api/v2/oauth`.
Прямой клиент httpx использует нативные `functions/function_call`, `functions_state_id` и `role=function`. Версии несовместимых SDK из занятий не смешиваются.
TLS проверяется; публичный корневой сертификат и его источник лежат в `certs/`.

## Данные и индексация

Архив `task/assets (2).zip` читается напрямую: 30 стенограмм ПМЭФ 2011–2016, 1952 фрагмента, EmbeddingsGigaR размерности 2560.
Распаковка других учебных assets и запуск ноутбуков курса не требуются.

```bash
spief-rag prepare --output data/index
spief-rag build-index --source data/index/chunks.jsonl --output data/rebuilt_index --limit 2
# Полная повторная индексация требует пустого каталога и расходует embedding-токены:
spief-rag build-index --source data/index/chunks.jsonl --output data/full_rebuilt_index
```

Команды не перезаписывают непустой каталог. Исходных отдельных стенограмм нет; canonical source — предоставленные фрагменты.
`build-index` не добавляет retrieval-инструкцию к документам. К поисковому запросу она добавляется как в занятии 2.
Для нового индекса: `Corpus.load("data/rebuilt_index")`, затем `SearchEngine(corpus, client)` и `Pipeline(...)`.

## Контракт и ошибки

```json
{
  "status": "ok",
  "answer": "Краткий ответ по источникам.",
  "citations": [{"chunk_id": "реальный-id", "quote": "Дословный фрагмент источника"}]
}
```

При нехватке сведений — `status=not_found`, объяснение и пустые citations. Метаданные добавляются кодом в `sources`.
API-ошибки, лимиты, обрезанная генерация и невалидные цитаты возвращаются как `error` вокруг ответа, не маскируются под `not_found`.
`ask` сохраняет безопасный журнал: answer/error/sources/evidence/trace/events/usage.

Базовый режим допускает один поисковый вызов, ReAct — до трёх. Имена и аргументы проверяются до исполнения.
Совпадение цитаты ещё не доказывает смысл ответа: для этого нужен содержательный оценщик.
Self-consistency создаёт три кандидата на одном контексте и ищет согласие минимум двух по нормализованным entity/attribute/value с подтверждениями. Отсутствие согласия — отдельная ошибка.

## Оценка и эксперименты

20 dev + 40 test, без пересечения исходных стенограмм. Разметка и ограничения — в `data/gold/README.md`.
20 примеров с утверждёнными автором метками калибруют судью; требуется минимум 80% согласия. `scripts/human_review.py` позволяет экспортировать и импортировать оценки примеров.

```bash
# Локальный BM25-аудит без API:
python scripts/retrieval_audit.py
# Отдельная диагностика semantic retrieval на dev, с API-эмбеддингами:
python scripts/tune_retrieval.py
# Новый полный протокол, расходующий API-токены:
python scripts/run_experiments.py --run-dir reports/runs/revalidated_20261020
```

Протокол: preflight → калибровка → пять вариантов на dev → выбор → baseline и выбранный вариант на test по три повтора → self-consistency и ReAct на составных dev-вопросах → отчёт.
При неполной содержательной оценке выбор останавливается. Если дешёвый вариант ухудшает качество, выбранным остаётся baseline; экономия заранее не обещается.
Benchmark фиксирует хеши корпуса, вопросов, промптов, схемы и реализации оценивания, настройки модели и происхождение калибровки судьи. Повтор совместимого запуска продолжает незаконченный файл; несовместимые данные отклоняются.

Исходные `reports/runs/*.jsonl` и их summary — замороженные журналы первого протокола (legacy). Их результаты сохраняются как исторические измерения; новая версия проверки воспроизводимости не дописывает эти файлы. Для нового полного эксперимента создаётся отдельный каталог, например `reports/runs/revalidated_20261020`; выбор конфигурации, калибровка, сравнение и отчёт этого запуска сохраняются внутри него. Эталоны и контрольные метки составлены по источникам, проверены и утверждены автором.

```bash
SPIEF_RUN_DIR=reports/runs/revalidated_20261020
mkdir -p "$SPIEF_RUN_DIR"
spief-rag calibrate --output "$SPIEF_RUN_DIR/calibration.jsonl"
for variant in baseline context short_prompt one_example compact; do
  spief-rag evaluate --variant "$variant" --split dev --calibration "$SPIEF_RUN_DIR/calibration.summary.json" --output "$SPIEF_RUN_DIR/dev_$variant.jsonl"
done
spief-rag select "$SPIEF_RUN_DIR/dev_baseline.summary.json" "$SPIEF_RUN_DIR/dev_context.summary.json" "$SPIEF_RUN_DIR/dev_short_prompt.summary.json" "$SPIEF_RUN_DIR/dev_one_example.summary.json" "$SPIEF_RUN_DIR/dev_compact.summary.json" --output "$SPIEF_RUN_DIR/selection.json"
spief-rag evaluate --variant baseline --split test --repeats 3 --calibration "$SPIEF_RUN_DIR/calibration.summary.json" --output "$SPIEF_RUN_DIR/test_baseline.jsonl"
spief-rag evaluate --config "$SPIEF_RUN_DIR/selected_config.json" --split test --repeats 3 --calibration "$SPIEF_RUN_DIR/calibration.summary.json" --output "$SPIEF_RUN_DIR/test_selected.jsonl"
spief-rag compare "$SPIEF_RUN_DIR/test_baseline.jsonl" "$SPIEF_RUN_DIR/test_selected.jsonl" --output "$SPIEF_RUN_DIR/comparison.json"
python scripts/cache_experiment.py
python scripts/demo_tools.py
spief-rag report --runs "$SPIEF_RUN_DIR" --output "$SPIEF_RUN_DIR/report"
```

`self_consistency` и `react` доступны через `--variant`. Для категории: `--categories multi`.
Кеш готовых ответов отключён в основном сравнении. Отдельный `ask --cache` учитывает точный вопрос, корпус, модель, промпт, схему и параметры, TTL — час.

## Метрики и расходы

- Поиск: recall по группам свидетельств, MRR, precision и покрытие контекстом.
- Ответ: JSON, источники, обязательные факты, запрещённые утверждения, отказы, groundedness и полнота.
- Результат: доля полезных ответов, парные регрессии, реальные токены, попытки, время и стоимость полезного ответа.

`useful_auto=None` — отсутствие содержательной оценки. При неполном покрытии стоимость полезного результата не вычисляется.
Индексация, preflight, судья и основной вопрос учитываются отдельно. Оплаченные ошибки не вычитаются.
При неизвестном расходе неответившей попытки денежная оценка — нижняя граница.
Предварительное ограничение использует консервативную границу UTF-8 bytes + резерв выхода, **не выдаёт её за токены модели**. Фактический расход берётся из API usage.
В GigaChat v1 `total_tokens` уже учитывает серверный кеш префикса; `precached_prompt_tokens` не вычитаются второй раз. Серверный кеш может влиять на сравнение даже при выключенном кеше готовых ответов.

Тарифный профиль `configs/tariff.json` соответствует Max/B2B, проверен 20.10.2026 по [официальной странице](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs).
Это пакетный эквивалент, не подтверждённое списание. Несовпадающий профиль не применяется.

## Демонстрация и сдача

- `notebooks/demo.ipynb` — локальная демонстрация и явно включаемые живые вызовы.
- `reports/report.md` — отчёт по фактическим измерениям.
- `docs/screencast.md` — сценарий записи до 10 минут.
- [Основной скринкаст](media/screencast.mp4) и [два поисковых инструмента](media/search-tools.mp4) — записи браузерной демонстрации с новыми API-вызовами. Русские субтитры включены в MP4 и сохранены рядом в `.vtt`.
- [Описание видео](media/README.md) и `media/recording-manifest.json` — озвучка, длительности, главы и воспроизведение. Авторский комментарий проверен и утверждён Иваном Лазичным; озвучка — Milena.
- [Новая платная проверка](reports/submission/README.md) — preflight, 20 калибровочных примеров и 12 диагностических вопросов под протоколом v2. Основной результат 70% → 70% получен на 40 вопросах с тремя повторами; отдельные 12 вопросов использованы для диагностики.
- [Разбор всех регрессий](reports/error_analysis.md) и [чек-лист сдачи](docs/submission_checklist.md).
- `docs/course_mapping.md` — связь реализации с занятиями.

Проект сдаётся в уже выделенный учебный каталог GitLab. Ключи и .venv туда не включать.
Сырые безопасные журналы в `reports/runs/` входят в материалы проекта: они содержат ответы, источники, usage и трассы, без ключей авторизации и внутренних рассуждений модели.
PDF содержит исторический дедлайн 22 сентября 2026; актуальный срок здесь не предполагается.

Локальный демонстрационный экран запускается командой `python scripts/demo_server.py --live` на `http://127.0.0.1:8765`. Ключ остаётся на сервере. Без `--live` доступны отчёты, а платные вызовы отключены. Текст и инструменты записи находятся в `docs/demo/` и `scripts/record_screencast.py`; дополнительные зависимости описаны в `media/README.md`.
