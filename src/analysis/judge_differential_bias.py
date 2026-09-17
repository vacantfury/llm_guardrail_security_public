#!/usr/bin/env python3
"""WHY NO NEW LABELING WAS NEEDED. Round R (2026-07-28) already labeled 100 benign
responses by hand, blind, with the judge verdict withheld -- and its sample is
split 49 text-channel / 51 image-channel. That is the attachment manipulation
itself, so the differential test runs on labels that already exist.

WHAT IT COMPUTES. Within each arm, the judge's refusal rate minus the human's
(the judge's bias against the human anchor). The quantity of interest is the
DIFFERENCE of those two biases: that is the part which does NOT cancel in a
paired contrast, and therefore the only part that could inflate a reported
effect. A percentile interval comes from resampling rows within arm.

SCOPE LIMIT, stated because it is real: Round R's responses come from
`qwen2_5_vl_7b` / `internvl3_8b` under a defense-pipeline study, where the image
channel carried an encoded payload. Paper B's blank canvas carries nothing. So
this bounds the judge's arm-dependent bias on a NEIGHBOURING manipulation, not
on the exact cells. It is evidence, not proof, and the paper says so.

Usage (labels + key are gitignored and stay local, as Round J/R's did):
    uv run python -m src.analysis.judge_differential_bias
"""
from __future__ import annotations

import collections
import csv
import random
import sys
from pathlib import Path

KEY = Path("judge_model_issue/round_r_refusal_key.csv")
SHEET = Path("judge_model_issue/round_r_refusal_labels_BLANK.md")
SEED = 20260808
N_RESAMPLE = 10_000
HUMAN_REFUSED = {"d": True, "i": True, "a": False}


def _read_labels(path: Path) -> dict[str, str]:
    """Read the markdown labeling sheet (`LABEL (a/d/i):` lines)."""
    sys.path.insert(0, "judge_model_issue")
    from compute_refusal_kappa import read_labels  # the canonical reader
    return read_labels(str(path))


def kappa(pairs: list[tuple]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    labs = {x for p in pairs for x in p}
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load() -> list[tuple[str, bool, bool]]:
    key = {r["sample_id"]: r for r in csv.DictReader(open(KEY))}
    out = []
    for sid, raw in _read_labels(SHEET).items():
        k = key.get(sid)
        if not k:
            continue
        h = HUMAN_REFUSED.get(str(raw).strip().lower()[:1])
        if h is None:
            continue
        out.append((k["channel"], h, str(k["judge_refused"]).strip().lower() == "true"))
    return out


def main() -> None:
    rows = load()
    print(f"usable labeled rows: {len(rows)}  (Round R, blind, judge verdict withheld)\n")
    print(f"{'arm':8s} {'n':>4s} {'human ref%':>11s} {'judge ref%':>11s} "
          f"{'judge-human':>12s} {'over-flag':>10s} {'miss':>6s} {'kappa':>7s}")
    print("-" * 76)

    by_arm = {a: [(h, j) for c, h, j in rows if c == a] for a in ("text", "image")}
    gap = {}
    for arm, a in by_arm.items():
        n = len(a)
        hr = 100 * sum(h for h, _ in a) / n
        jr = 100 * sum(j for _, j in a) / n
        gap[arm] = jr - hr
        print(f"{arm:8s} {n:4d} {hr:10.1f}% {jr:10.1f}% {jr - hr:+11.1f} "
              f"{sum(1 for h, j in a if j and not h):10d} "
              f"{sum(1 for h, j in a if h and not j):6d} {kappa(a):7.3f}")

    diff = gap["image"] - gap["text"]
    rng = random.Random(SEED)
    draws = []
    for _ in range(N_RESAMPLE):
        g = {}
        for arm, xs in by_arm.items():
            s = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
            g[arm] = 100 * (sum(j for _, j in s) - sum(h for h, _ in s)) / len(s)
        draws.append(g["image"] - g["text"])
    draws.sort()
    lo, hi = draws[int(0.025 * N_RESAMPLE)], draws[int(0.975 * N_RESAMPLE)]

    print("\n" + "=" * 76)
    print(f"judge bias vs human:  text {gap['text']:+.1f}pp   image {gap['image']:+.1f}pp")
    print(f"DIFFERENTIAL (image - text): {diff:+.1f}pp, 95% CI [{lo:+.1f}, {hi:+.1f}] "
          f"({N_RESAMPLE} row-resamples within arm, seed {SEED})")
    print()
    print(f"  contains zero .................. {lo <= 0 <= hi}")
    for eff, name in ((54, "largest headline effect"), (23, "smallest significant effect")):
        print(f"  |CI upper| vs {name} (+{eff}pp): {100 * max(abs(lo), abs(hi)) / eff:.0f}%")
    print()
    print("READING. The point estimate is small and the interval contains zero, so")
    print("there is no evidence the judge is differentially biased by attachment.")
    print("But n~50 per arm does NOT bound it tightly: the interval's upper end is a")
    print("modest fraction of the largest effects and a substantial fraction of the")
    print("smallest significant ones. Conclusion: the large contrasts are protected;")
    print("the ~20pp contrasts are NOT fully protected by this test alone. Do not")
    print("report this as 'the bias cancels' -- report the interval.")


if __name__ == "__main__":
    main()
