"""
Run provenance + safe persistence.

Every results.json should answer the four questions you need to trust or
reproduce a number months later:
  - git_sha / git_dirty : what code produced this, and was it fully committed?
  - schema_version      : how should a reader parse this file's format?
  - judge_config_hash   : were two results judged the same way (comparable)?

Plus atomic writes, so a killed job (SLURM timeout / OOM) never leaves a
half-written, corrupt results.json that downstream tooling reads as valid.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

# Bump when the results.json field layout or semantics change. Readers should
# dispatch on this instead of sniffing which fields exist.
SCHEMA_VERSION = 1

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def get_git_sha() -> str:
    """Commit SHA of the code that ran.

    Order: GIT_SHA env var → `<repo>/.git_sha` file (written by the mac's
    post-commit hook; survives rsync to a cluster node that has no .git) →
    live `git rev-parse HEAD` → "unknown". Cached: code can't change mid-process.

    A literal "unknown" env value is skipped (the sbatch exports GIT_SHA=unknown
    when its own `git` fails on the cluster) so we fall through to the file.
    """
    env = (os.environ.get("GIT_SHA") or "").strip()
    if env and env != "unknown":
        return env
    sha_file = _REPO_ROOT / ".git_sha"
    try:
        if sha_file.exists():
            val = sha_file.read_text().strip()
            if val:
                return val
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


@lru_cache(maxsize=1)
def is_git_dirty() -> bool:
    """True if the working tree had uncommitted changes — i.e. git_sha alone
    does NOT fully capture the code that ran. GIT_DIRTY env var overrides.
    """
    env = os.environ.get("GIT_DIRTY")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=_REPO_ROOT,
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return False
        return bool(out.stdout.strip())
    except Exception:
        return False


def judge_config_hash(judge_cfg: dict, evaluator_names: Iterable[str]) -> str:
    """Short stable hash of everything that determines a verdict.

    Two results with the same hash were judged identically → directly
    comparable. Covers judge model + temperature + max_tokens + which
    evaluators ran. (If you later version evaluator prompt templates, fold a
    template version into `judge_cfg` so prompt edits change the hash too.)
    """
    payload = {
        "judge_cfg": {
            k: judge_cfg.get(k)
            for k in ("model", "temperature", "max_tokens")
        },
        "evaluators": sorted(evaluator_names),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def provenance_fields(
    *, judge_cfg: dict | None = None,
    evaluator_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """The standard provenance block to splat into every result model."""
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": get_git_sha(),
        "git_dirty": is_git_dirty(),
    }
    if judge_cfg is not None and evaluator_names is not None:
        out["judge_config_hash"] = judge_config_hash(judge_cfg, evaluator_names)
    return out


# ---------------------------------------------------------------------------
# Atomic writes — write to a sibling .tmp then os.replace (atomic on POSIX),
# so a crash mid-write can never truncate the real file.
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes (used to fingerprint a referenced results.json
    so a stored pointer can detect later drift of its source)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def write_jsonl_atomic(path: str | Path, lines: Iterable[str]) -> None:
    """`lines` are already-serialized JSON strings (no trailing newline)."""
    atomic_write_text(path, "".join(line + "\n" for line in lines))
