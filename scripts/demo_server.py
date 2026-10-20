"""Serve a local presentation and explicitly enabled, metered GigaChat demo calls.

Only this process reads the ignored environment file. The browser receives public
answers, evidence and usage, never authorization headers or credentials.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from spief_rag.benchmark import load_tariff
from spief_rag.client import APIError, GigaChatClient
from spief_rag.contracts import Answer, public_schema, validate_grounding
from spief_rag.corpus import Corpus
from spief_rag.pipeline import Pipeline, parse_json
from spief_rag.prompts import answer_system
from spief_rag.search import SearchEngine, select_context, tool_specs
from spief_rag.settings import ROOT, Settings, get_experiment
from spief_rag.telemetry import Ledger, append_jsonl, run_identity
from spief_rag.validation import approved_author


def read_optional(path: Path, default=None):
    """Read a public JSON artifact if present, without reading arbitrary paths."""
    return json.loads(path.read_text()) if path.exists() else default


class DemoSession:
    """Bind a single local demo to bounded calls and append-only public logs."""

    def __init__(self, *, live: bool, output: Path, max_questions: int = 12):
        """Initialize the corpus offline; create the paid client only on opt-in."""
        self.live, self.output = live, output
        self.max_questions, self.count = max_questions, 0
        self.lock = threading.Lock()
        self.corpus = Corpus.load()
        self.config = get_experiment(
            "short_prompt", str(ROOT / "configs/selected_config.json")
        )
        self.client = GigaChatClient(Settings.from_env()) if live else None
        self.engine = SearchEngine(self.corpus, self.client)
        self.tariff = load_tariff(self.client) if self.client else None

    def project(self) -> dict:
        """Expose only the fixed public artifacts needed by the presentation."""
        return {
            "live": self.live,
            "corpus": {
                "documents": len({r["source_id"] for r in self.corpus.chunks}),
                "chunks": len(self.corpus.chunks),
            },
            "config": self.config.to_dict(),
            "comparison": read_optional(ROOT / "reports/comparison.json", {}),
            "extras": {
                n: read_optional(ROOT / f"reports/comparison_{n}.json", {})
                for n in ["self_consistency", "react", "multihop"]
            },
            "preflight": read_optional(ROOT / "reports/submission/preflight.json", {}),
            "tests": read_optional(ROOT / "reports/submission/offline_checks.json", {}),
            "live_summary": read_optional(
                ROOT / "reports/submission/validation.summary.json", {}
            ),
            "tool_coverage": read_optional(
                ROOT / "reports/live_tool_coverage.json", {}
            ),
            "author": approved_author() or "Иван Лазичный",
            "annotation_provenance": "checked and approved by author; metrics computed by automated benchmark",
        }

    def forced(self, question: str, name: str) -> dict:
        """Execute an actual forced native function call and validate its continuation."""
        started = time.perf_counter()
        ledger = Ledger()
        run = {
            **run_identity(),
            "question": question,
            "model": self.client.settings.model,
            "config": self.config.to_dict(),
            "corpus_version": self.corpus.version,
            "selector": "forced",
            "tool": name,
            "answer": None,
            "error": None,
            "sources": [],
            "evidence": [],
            "trace": [],
            "cache_hit": False,
        }
        try:
            history = [
                {"role": "system", "content": answer_system(self.config)},
                {"role": "user", "content": question},
            ]
            response = self.client.chat(
                history,
                ledger,
                functions=tool_specs(),
                function_call={"name": name},
                max_tokens=300,
            )
            call = response["message"].get("function_call")
            if not call or call["name"] != name:
                raise ValueError("wrong_forced_tool")
            run["trace"].append({"type": "tool_call", "step": 1, **call})
            raw = self.engine.execute(
                name, call["arguments"], ledger, top_k=self.config.top_k
            )
            hits = select_context(
                raw, deduplicate=True, byte_budget=self.config.context_byte_budget
            )
            run["evidence"] = hits
            run["trace"].append(
                {
                    "type": "retrieval",
                    "step": 1,
                    "tool": name,
                    "raw_hit_ids": [r["chunk_id"] for r in raw],
                    "context_ids": [r["chunk_id"] for r in hits],
                }
            )
            history.extend(
                [
                    response["message"],
                    {
                        "role": "function",
                        "name": name,
                        "content": json.dumps({"hits": hits}, ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": "Верни окончательный ответ по источникам в JSON.",
                    },
                ]
            )
            final = self.client.chat(
                history, ledger, schema=public_schema(), max_tokens=1200
            )
            answer = Answer.model_validate(parse_json(final["message"]["content"]))
            validate_grounding(answer, {r["chunk_id"]: r for r in hits})
            run["answer"] = answer.model_dump()
            cited = {c.chunk_id for c in answer.citations}
            run["sources"] = [
                {k: r[k] for k in ["chunk_id", "session", "time_start", "speakers"]}
                for r in hits
                if r["chunk_id"] in cited
            ]
            run["trace"].append(
                {"type": "validated_answer", "schema": True, "citations": True}
            )
        except (APIError, ValueError, TypeError) as exc:
            run["error"] = {"code": getattr(exc, "code", type(exc).__name__)}
        finally:
            run.update(
                events=ledger.events,
                usage=ledger.summary(self.tariff),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        return run

    def ask(self, question: str, mode: str) -> dict:
        """Make a bounded live request; reject offline and exhausted demo sessions."""
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > 1500
        ):
            raise ValueError("invalid_question")
        if mode not in {"auto", "semantic_search", "keyword_search"}:
            raise ValueError("invalid_mode")
        if not self.live:
            raise ValueError("live_disabled_start_with_--live")
        with self.lock:
            if self.count >= self.max_questions:
                raise ValueError("demo_question_limit_reached")
            self.count += 1
            if mode == "auto":
                run = Pipeline(self.engine, self.config, self.tariff).run(question)
                run["selector"] = "auto"
            else:
                run = self.forced(question, mode)
            run["demonstration"] = (
                "new paid API call; excluded from frozen quality comparison"
            )
            append_jsonl(self.output, run)
            return run

    def close(self) -> None:
        """Release the transport after recording or a normal server shutdown."""
        if self.client:
            self.client.close()


class DemoHandler(BaseHTTPRequestHandler):
    """Serve one presentation file and a small same-origin JSON API on loopback."""

    def __init__(self, *args, session: DemoSession, **kwargs):
        """Inject the shared session without exposing it through request parameters."""
        self.session = session
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Keep browser requests and typed questions out of console logs."""

    def send_content(self, content: bytes, mime: str, status: int = 200):
        """Return a non-cacheable local asset with a fixed content type."""
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, value: dict, status: int = 200):
        """Serialize only public application data as UTF-8 JSON."""
        self.send_content(
            json.dumps(value, ensure_ascii=False).encode(),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self):
        """Expose the presentation and its fixed public project summary."""
        path = urlparse(self.path).path
        if path == "/":
            self.send_content(
                (ROOT / "docs/demo/index.html").read_bytes(), "text/html; charset=utf-8"
            )
        elif path == "/api/project":
            self.send_json(self.session.project())
        else:
            self.send_json({"error": "not_found"}, 404)

    def do_POST(self):
        """Accept same-origin asks; no file paths or provider settings come from the browser."""
        origin = self.headers.get("Origin")
        expected = "http://" + self.headers.get("Host", "")
        if origin != expected:
            self.send_json({"error": "same_origin_required"}, 403)
            return
        if self.path != "/api/ask":
            self.send_json({"error": "not_found"}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 16000:
                raise ValueError("request_size_limit")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("request_must_be_object")
            self.send_json(
                self.session.ask(data.get("question"), data.get("mode", "auto"))
            )
        except (ValueError, TypeError, KeyError) as exc:
            self.send_json({"error": {"code": str(exc)[:100]}}, 400)


def make_server(*, live: bool, output: Path, port: int = 0):
    """Create a loopback-only server; the caller controls its lifetime."""
    session = DemoSession(live=live, output=output)
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(DemoHandler, session=session)
    )
    return server, session


def main():
    """Require --live for paid calls; the default presentation is read-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/submission/screencast_calls.jsonl",
    )
    args = parser.parse_args()
    server, session = make_server(live=args.live, output=args.output, port=args.port)
    print(f"http://127.0.0.1:{server.server_port} live={args.live}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()


if __name__ == "__main__":
    main()
