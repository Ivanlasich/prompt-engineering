"""Four supplemental dev challenges provably require multiple source fragments."""

import json
from collections import defaultdict

from spief_rag.corpus import Corpus
from spief_rag.settings import ROOT


def main():
    """Use only development source documents and check absence of a single-chunk answer."""
    corpus = Corpus.load()
    groups = defaultdict(list)
    for row in corpus.chunks:
        groups[row["source_id"]].append(row)
    sources = list(groups)
    cases = []

    def add(doc, question, reference, facts):
        """Attach independently verified literal evidence to each required fact."""
        source = sources[doc]
        annotated = []
        for i, (needle, pattern) in enumerate(facts):
            evidence = [r for r in groups[source] if needle in r["text"]]
            assert evidence, needle
            r = evidence[0]
            p = r["text"].index(needle)
            annotated.append(
                {
                    "id": f"fact-{i + 1}",
                    "needle": needle,
                    "patterns": [pattern],
                    "evidence_ids": [r["chunk_id"] for r in evidence],
                    "primary_chunk_id": r["chunk_id"],
                    "quote": r["text"][
                        max(0, p - 80) : min(len(r["text"]), p + len(needle) + 100)
                    ],
                }
            )
        assert not set.intersection(*(set(f["evidence_ids"]) for f in annotated)), (
            "one_chunk_would_answer"
        )
        cases.append(
            {
                "id": f"multihop-{len(cases) + 1:02d}",
                "split": "dev",
                "group_id": source,
                "source_ids": [source],
                "category": "multi",
                "question": question,
                "expected_status": "ok",
                "reference_answer": reference,
                "required_facts": annotated,
                "forbidden_patterns": [],
                "annotation_provenance": "author_reviewed",
                "single_chunk_can_cover_all_facts": False,
            }
        )

    add(
        0,
        "Какую должность в МЭА занимал Ласло Варро и какой ежегодный рост спроса на газ он прогнозировал на нефтегазовом брифинге ПМЭФ-2011?",
        "Директор МЭА по газовому, угольному и энергетическому рынкам; ежегодный рост спроса на газ 2,4%.",
        [
            (
                "Ласло Варро, директор МЭА по газовому, угольному и энергетическому рынкам",
                r"директор.*мэа|мэа.*директор",
            ),
            ("ежегодный рост спроса на газ будет составлять 2,4%", r"2[,.]4"),
        ],
    )
    add(
        2,
        "Какие организации проводили конференцию о российских стартапах ПМЭФ-2012 и сколько резидентов «Ингрии» уже получили статус резидента «Сколково», а сколько ожидалось до конца года по словам Ирины Калашниковой?",
        "Microsoft и технопарк «Ингрия»; три резидента уже получили статус, до конца года ожидали десять.",
        [
            (
                "организована компанией Microsoft и технопарком «Ингрия»",
                r"microsoft|майкрософт",
            ),
            ("технопарком «Ингрия»", r"ингри"),
            ("уже три резидента", r"три|3"),
            ("до конца года достигнет десяти", r"десят|10"),
        ],
    )
    add(
        4,
        "Какую организацию представляла Наталья Ханженкова на экологической панели ПМЭФ-2013 и на сколько процентов, по её словам, использование стеклобоя снижает потребление энергии для новых бутылок?",
        "ЕБРР, Европейский банк реконструкции и развития; снижение на 20–30%.",
        [
            (
                "Наталья Ханженкова, Европейский банк реконструкции и развития (ЕБРР)",
                r"ебрр|европейск.*банк",
            ),
            ("на 20—30% снижает потребление энергии", r"20.*30"),
        ],
    )
    add(
        6,
        "Кого Эско Ахо процитировал об инвестициях в знания на образовательной сессии ПМЭФ-2015 и какую компанию представляла Сари Балдауф как председатель совета директоров?",
        "Бенджамина Франклина; Сари Балдауф — председатель совета директоров Fortum.",
        [
            ("Benjamin Franklin", r"франклин|franklin"),
            ("Chair of the Board at Fortum", r"fortum|фортум"),
        ],
    )
    output = ROOT / "data/gold/multihop.jsonl"
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cases))
    print("Supplemental dev multi-fragment questions:", len(cases))


if __name__ == "__main__":
    main()
