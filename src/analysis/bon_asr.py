"""Best-of-N ASR(N) analysis.

Reads a `defense+evaluate` run's ``raw_results.jsonl`` (one row per BoN variant,
``id`` = ``<behavior>__bonK``, boolean ``asr`` = jailbroken), groups variants by
behavior, and computes attack-success-rate as a function of budget N:

    ASR(N) = mean_behaviors [ 1 - (1 - p_i)^N ]

where ``p_i`` is the fraction of behavior i's variants judged jailbroken — the
analytic any-success rate at budget N (matches BoN's "learn_p" bootstrap). Also
reports a log-log slope of ``log(1 - ASR(N))`` vs ``log(N)`` (BoN's ASR~N power
law). Reimplemented from other_repos/bon-jailbreaking/bon/utils/power_law_simple.py;
cite Hughes et al. (NEURIPS2025_69f3eb24). Stdlib-only, standalone-runnable.
"""
import json
import math
import re
from pathlib import Path
from typing import Optional

_VARIANT_RE = re.compile(r"^(.*)__bon(\d+)$")
_DEFAULT_NS = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def _base_id(vid: str):
    m = _VARIANT_RE.match(vid)
    return (m.group(1), int(m.group(2))) if m else (vid, 0)


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _lin_slope(xs: list, ys: list) -> Optional[float]:
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    return (n * sxy - sx * sy) / denom if denom else None


def load_variant_flags(results_dir) -> dict:
    """behavior_id -> list of (variant_k, flagged_bool), from raw_results.jsonl."""
    rr = Path(results_dir) / "raw_results.jsonl"
    by_behavior: dict = {}
    for line in rr.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        base, k = _base_id(str(row["id"]))
        by_behavior.setdefault(base, []).append((k, bool(row.get("asr"))))
    return by_behavior


def compute_asr_curve(results_dir, ns: Optional[list] = None) -> dict:
    by_behavior = load_variant_flags(results_dir)
    ns = ns or _DEFAULT_NS
    ps = [_mean([f for _, f in v]) for v in by_behavior.values() if v]
    n_behaviors = len(ps)
    max_variants = max((len(v) for v in by_behavior.values()), default=0)
    ns = [n for n in ns if n <= max_variants] or ([max_variants] if max_variants else [])
    asr_of_n = {n: _mean([1 - (1 - p) ** n for p in ps]) for n in ns}
    xs = [math.log(n) for n in ns if 0 < asr_of_n[n] < 1]
    ys = [math.log(1 - asr_of_n[n]) for n in ns if 0 < asr_of_n[n] < 1]
    slope = _lin_slope(xs, ys) if len(xs) >= 2 else None
    return {
        "results_dir": str(results_dir),
        "n_behaviors": n_behaviors,
        "max_variants_per_behavior": max_variants,
        "per_behavior_success_prob_mean": _mean(ps),
        "per_behavior_ever_succeeds_frac": _mean([1.0 if p > 0 else 0.0 for p in ps]),
        "asr_of_n": asr_of_n,
        "loglog_slope": slope,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Best-of-N ASR(N) from a defense+evaluate run dir")
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None, help="write JSON summary here too")
    a = ap.parse_args()
    summary = compute_asr_curve(a.results_dir)
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
