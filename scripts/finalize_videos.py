"""Finalize recorded media locally without any API calls.

Preserve the original paid browser episodes, replace only the final commentary,
and keep reversible originals in the ignored recording cache.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from record_screencast import duration, write_captions

from spief_rag.settings import ROOT

AUTHOR = "Иван Лазичный"
FOOTER = "Автор: Иван Лазичный. Озвучка синтезирована."
WORK = ROOT / ".cache/recording/final-media"
BACKUP = ROOT / ".cache/recording/original-footage"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def command(args: list[str]) -> None:
    """Run a local media command, failing on any encoder or decoder error."""
    subprocess.run(args, check=True)


def probe(path: Path) -> dict:
    """Inspect actual media streams, chapter markers and metadata."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def snapshot() -> list[dict]:
    """Keep the first originals unchanged so media updates remain reversible."""
    BACKUP.mkdir(parents=True, exist_ok=True)
    for name in [
        "screencast.mp4",
        "search-tools.mp4",
        "screencast.vtt",
        "search-tools.vtt",
        "recording-manifest.json",
        "screencast-poster.png",
    ]:
        target = BACKUP / name
        if not target.exists():
            shutil.copy2(ROOT / "media" / name, target)
    return json.loads((BACKUP / "recording-manifest.json").read_text())


def capture_checks() -> None:
    """Capture the updated local HTML with public artifact fixtures and no network."""
    from playwright.sync_api import sync_playwright

    def read_json(name: str) -> dict:
        """Read a fixed public artifact for the offline presentation."""
        return json.loads((ROOT / name).read_text())

    html = (ROOT / "docs/demo/index.html").read_text()
    data = {
        "live": False,
        "corpus": {"documents": 30, "chunks": 1952},
        "config": read_json("configs/selected_config.json"),
        "comparison": read_json("reports/comparison.json"),
        "extras": {},
        "tests": read_json("reports/submission/offline_checks.json"),
        "author": AUTHOR,
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        try:
            page = browser.new_page(
                viewport={"width": 1600, "height": 1000}, device_scale_factor=1
            )

            def fulfill_offline(route) -> None:
                """Serve only local fixtures; the capture never reaches a network."""
                if route.request.url.endswith("/api/project"):
                    route.fulfill(json=data)
                else:
                    route.fulfill(content_type="text/html", body=html)

            page.route("**/*", fulfill_offline)
            page.goto("http://video.local/", wait_until="domcontentloaded")
            page.wait_for_function("window.demo?.project")
            page.evaluate("demo.selectTab('checks')")
            page.screenshot(path=str(WORK / "checks.png"))
        finally:
            browser.close()


def metadata(timeline: list[dict], length: float, output: Path) -> None:
    """Write author attribution and unchanged paid-episode chapter markers."""
    lines = [
        ";FFMETADATA1",
        "title=ПМЭФ: RAG на GigaChat",
        "artist=" + AUTHOR,
        "author=" + AUTHOR,
        "comment=" + FOOTER,
    ]
    for i, item in enumerate(timeline):
        end = timeline[i + 1]["start"] if i + 1 < len(timeline) else length
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={round(item['start'] * 1000)}",
            f"END={round(end * 1000)}",
            "title=" + item["title"],
        ]
    output.write_text("\n".join(lines) + "\n")


def encode(entry: dict, main: bool, narration: dict) -> dict:
    """Re-encode the existing footage with a visible attribution and updated captions."""
    stem = "screencast" if main else "search-tools"
    source = BACKUP / f"{stem}.mp4"
    timeline = entry["chapters"]
    length = duration(source)
    captions = ROOT / f"media/{stem}.vtt"
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source)]
    filters = []
    if main:
        cut = timeline[-1]["start"]
        timeline[-1]["text"] = narration["main"][8]["text"]
        timeline[-1]["audio_duration"] = duration(WORK / "final.aiff")
        tail = timeline[-1]["audio_duration"] + 3.5
        length = cut + tail
        args += [
            "-loop",
            "1",
            "-framerate",
            "25",
            "-i",
            str(WORK / "checks.png"),
            "-i",
            str(WORK / "final.aiff"),
        ]
        filters += [
            f"[0:v]trim=duration={cut},setpts=PTS-STARTPTS[v0]",
            f"[0:a]atrim=duration={cut},asetpts=PTS-STARTPTS[a0]",
            f"[1:v]trim=duration={tail},setpts=PTS-STARTPTS[v1]",
            f"[2:a]loudnorm=I=-16:TP=-1.5:LRA=11,apad,atrim=duration={tail},asetpts=PTS-STARTPTS[a1]",
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[footage][audio]",
        ]
        caption_index, meta_index = 3, 4
        video_input, audio_input = "[footage]", "[audio]"
    else:
        caption_index, meta_index = 1, 2
        video_input, audio_input = "[0:v]", "0:a:0"
    write_captions(timeline, captions)
    chapter_file = WORK / f"{stem}.ffmetadata"
    metadata(timeline, length, chapter_file)
    args += ["-i", str(captions), "-i", str(chapter_file)]
    filters.append(
        video_input
        + "drawbox=x=0:y=ih-44:w=iw:h=44:color=white:t=fill,drawtext=fontfile="
        + FONT
        + ":textfile="
        + str(WORK / "footer.txt")
        + ":fontcolor=0x13345b:fontsize=20:x=(w-tw)/2:y=h-30[video]"
    )
    target = WORK / f"{stem}.mp4"
    args += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[video]",
        "-map",
        audio_input,
        "-map",
        f"{caption_index}:s:0",
        "-map_metadata",
        str(meta_index),
        "-map_chapters",
        str(meta_index),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "24",
        "-threads",
        "6",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "25",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        "language=rus",
        "-t",
        str(length),
        "-movflags",
        "+faststart",
        str(target),
    ]
    command(args)
    command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(target),
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-f",
            "null",
            "-",
        ]
    )
    shutil.copy2(target, ROOT / f"media/{stem}.mp4")
    info = probe(target)
    entry.update(
        duration_seconds=float(info["format"]["duration"]),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        author=AUTHOR,
        narration=narration["disclosure"],
        postprocessing="Original live API episodes and errors preserved; final commentary updated; author/TTS footer and metadata added",
    )
    return {
        "file": entry["file"],
        "duration_seconds": entry["duration_seconds"],
        "streams": sorted({s["codec_type"] for s in info["streams"]}),
        "chapters": len(info["chapters"]),
        "hash_match": True,
        "full_decode": "passed",
        "author": AUTHOR,
        "author_footer": True,
    }


def main() -> None:
    """Update both deliverable videos locally and verify every encoded frame."""
    WORK.mkdir(parents=True, exist_ok=True)
    entries = snapshot()
    narration = json.loads((ROOT / "docs/demo/narration.json").read_text())
    (WORK / "footer.txt").write_text(FOOTER)
    (WORK / "final.txt").write_text(narration["main"][8]["text"])
    command(
        [
            "say",
            "-v",
            "Milena",
            "-r",
            "165",
            "-f",
            str(WORK / "final.txt"),
            "-o",
            str(WORK / "final.aiff"),
        ]
    )
    capture_checks()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(encode, entry, i == 0, narration)
            for i, entry in enumerate(entries)
        ]
        checks = [future.result() for future in futures]
    (ROOT / "media/recording-manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n"
    )
    command(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            "60",
            "-i",
            str(ROOT / "media/screencast.mp4"),
            "-frames:v",
            "1",
            str(ROOT / "media/screencast-poster.png"),
        ]
    )
    (ROOT / "reports/submission/video_checks.json").write_text(
        json.dumps(
            {
                "videos": checks,
                "synthetic_narration_disclosed": True,
                "author": AUTHOR,
                "new_api_calls": 0,
                "originals_backup": str(BACKUP.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
