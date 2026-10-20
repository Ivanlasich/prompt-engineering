"""Check the exact Git delivery candidates without displaying credential values."""

import base64
import io
import json
import re
import subprocess
import zipfile

from dotenv import dotenv_values

from spief_rag.settings import ROOT


def suspicious_content(raw: bytes, credential: bytes) -> bool:
    """Detect the local key, encoded authorization pairs and JWT-shaped secrets."""
    if credential and credential in raw:
        return True
    if re.search(
        rb"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", raw
    ):
        return True
    for match in re.finditer(rb"[A-Za-z0-9+/]{80,}={0,2}", raw):
        try:
            decoded = base64.b64decode(match.group(), validate=True)
        except ValueError:
            continue
        if re.fullmatch(rb"[0-9a-fA-F-]{36}:[0-9a-fA-F-]{36}", decoded):
            return True
    return False


def main():
    """Fail before publishing on a secret or oversized blob; record only safe findings."""
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    paths = sorted(set(p.decode() for p in output.split(b"\0") if p))
    local = dotenv_values(ROOT / ".env.local")
    credential = (local.get("GIGACHAT_CREDENTIALS") or "").encode()
    findings, count, max_size = [], 0, 0
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            continue
        count += 1
        if path.name.startswith(".env") and path.name != ".env.example":
            findings.append({"path": name, "type": "environment_file"})
        raw = path.read_bytes()
        max_size = max(max_size, len(raw))
        if len(raw) >= 95_000_000:
            findings.append({"path": name, "type": "oversized_git_blob"})
        if path.suffix not in {".mp4", ".png", ".crt"} and suspicious_content(
            raw, credential
        ):
            findings.append({"path": name, "type": "possible_secret"})
        if path.suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                for member in archive.namelist():
                    if member.endswith(
                        (".json", ".jsonl", ".txt", ".py", ".ipynb")
                    ) and suspicious_content(archive.read(member), credential):
                        findings.append(
                            {
                                "path": name + ":" + member,
                                "type": "possible_archive_secret",
                            }
                        )
    result = {
        "checked_files": count,
        "largest_blob_bytes": max_size,
        "findings": findings,
        "credentials_in_report": False,
        "scope": "Git-tracked and non-ignored untracked delivery candidates",
    }
    (ROOT / "reports/submission/delivery_scan.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
