"""Verify a guard capture's logprobs decode to the SAME verdict the parser produced.

`guard_threshold.verdict_probability` matches on EXACT token text, so a guard
whose verdict word is split by the tokenizer derives a P(unsafe) that either
contradicts its own parsed verdict or resolves to nothing at all. Neither
failure is visible in the resulting curve — a silently-unresolved guard just
looks binary, and a mis-matched one just looks miscalibrated. This module is the
gate that makes the tier-2 (continuous) numbers trustworthy, and
`conf/analysis/guard_threshold.yaml` requires it be run before any guard's
continuous curve is reported.

It caught a real defect on the 2026-07-25 capture: Qwen3Guard's
``Controversial`` verdict tokenizes as ``Cont`` + ``roversial``, so the
spelled-out vocabulary entry matched nothing and 18% of that guard's records
fell back to the ordinal tier.

Two things are reported per guard:

* **coverage** — what fraction of records the vocabulary could resolve at all.
  Low coverage means a split/renamed verdict token, not an uncertain guard.
* **agreement** — of the resolved records, how often ``P(unsafe) > 0.5`` matches
  the parser's boolean. Bulk disagreement means the safe/unsafe lists are
  swapped or wrong; a sub-1% residue is expected (knife-edge records where the
  probability sits a hair either side of 0.50 while greedy decoding committed
  the other way).

The printed ``UNRESOLVED heads`` line is the diagnostic: it shows the literal
opening tokens of the records the vocabulary missed, which is how a new guard's
verdict vocabulary should be set in the first place.

Usage (from the repo root, on the cluster where the capture lives)::

    python -m src.analysis.verify_guard_logprobs outputs/autoattack_defense/guard_scores/<ts>

Exit status is 0 when every guard passes, 1 otherwise — so it can gate a
downstream sweep in a script.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from .guard_threshold import load_config, locate_verdict, verdict_probability

# A guard passes only if the vocabulary resolves nearly every record AND the
# resolved ones agree with the parser. Both bars are deliberately just under
# 100%: the expected residue is knife-edge records, not systematic error.
MIN_COVERAGE = 0.95
MIN_AGREEMENT = 0.98

# How many disagreeing records to print per guard. Enough to see the pattern
# (a swapped vocabulary shows up in the first one); not enough to bury it.
MAX_EXAMPLES = 5


class GuardCheck:
    """Coverage/agreement tally for one guard's captured records."""

    def __init__(self, guard: str) -> None:
        self.guard = guard
        self.n = 0
        self.n_mechanism_errors = 0
        self.n_with_logprobs = 0
        self.n_resolved = 0
        self.n_agree = 0
        self.matched_tokens: Counter = Counter()
        self.heads: Counter = Counter()
        self.unresolved_heads: Counter = Counter()
        self.examples: list[dict] = []

    @property
    def coverage(self) -> float:
        return self.n_resolved / self.n_with_logprobs if self.n_with_logprobs else 0.0

    @property
    def agreement(self) -> float:
        return self.n_agree / self.n_resolved if self.n_resolved else 0.0

    @property
    def ok(self) -> bool:
        return (
            self.n_resolved > 0
            and self.coverage >= MIN_COVERAGE
            and self.agreement >= MIN_AGREEMENT
        )

    def report(self) -> str:
        lines = [
            f"\n=== {self.guard} [{'OK' if self.ok else 'SUSPECT'}] ===",
            f"  records           : {self.n}   (mechanism errors: {self.n_mechanism_errors})",
            f"  with logprobs     : {self.n_with_logprobs}",
            f"  vocab resolved    : {self.n_resolved} ({100 * self.coverage:.1f}%)",
            f"  verdict agreement : {self.n_agree}/{self.n_resolved} "
            f"({100 * self.agreement:.2f}%)",
            f"  matched tokens    : {dict(self.matched_tokens.most_common(6))}",
            f"  completion heads  : {dict(self.heads.most_common(4))}",
        ]
        if self.unresolved_heads:
            lines.append(
                f"  UNRESOLVED heads  : {dict(self.unresolved_heads.most_common(4))}"
            )
        lines += [f"    DISAGREE {ex}" for ex in self.examples]
        return "\n".join(lines)


def _head(tokens: list[str], n: int = 3) -> str:
    """The first few tokens verbatim — this is what exposes a split verdict word."""
    return "|".join(tokens[:n])


def check_guard(path: Path, cfg: dict) -> GuardCheck:
    """Tally one guard's capture JSONL against the configured verdict vocabulary."""
    guard = path.stem
    check = GuardCheck(guard)

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            check.n += 1

            # A mechanism error is a dead endpoint, not a verdict. The capture
            # pass nulls the verdict for these; they must not be scored.
            if rec.get("mechanism_error"):
                check.n_mechanism_errors += 1
                continue

            logprobs = rec.get("logprobs")
            if not logprobs:
                continue
            check.n_with_logprobs += 1

            tokens = [(t.get("token") or "") for t in (logprobs.get("content") or [])]
            head = _head(tokens)
            check.heads[head] += 1

            # Ask the sweep's own locator which token it scores, rather than
            # re-deriving it here: a diagnostic that scans differently from the
            # code under test reports confident nonsense (an earlier copy of
            # this loop ignored the anchor and named a token from the middle of
            # the model's reasoning trace).
            found = locate_verdict(logprobs, guard, cfg)
            if found is not None:
                _idx, tok_text, tok_side = found
                check.matched_tokens[f"{tok_text.strip().lower()}->{tok_side}"] += 1

            p_unsafe = verdict_probability(logprobs, guard, cfg)
            if p_unsafe is None:
                check.unresolved_heads[head] += 1
                continue
            check.n_resolved += 1

            parsed: Optional[bool] = rec.get("verdict_unsafe")
            if parsed is None:
                continue
            if (p_unsafe > 0.5) == bool(parsed):
                check.n_agree += 1
            elif len(check.examples) < MAX_EXAMPLES:
                check.examples.append({
                    "id": rec.get("id"),
                    "parsed_unsafe": parsed,
                    "p_unsafe": round(p_unsafe, 4),
                    "head": head,
                    "raw": (rec.get("raw") or "")[:100].replace("\n", " "),
                })

    return check


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(__doc__)
        return 2

    scores_dir = Path(argv[0])
    files = sorted(scores_dir.glob("*.jsonl"))
    if not files:
        print(f"no capture JSONL found under {scores_dir}")
        return 2

    cfg = load_config()
    checks = [check_guard(path, cfg) for path in files]
    for check in checks:
        print(check.report())

    bad = [c.guard for c in checks if not c.ok]
    print(
        f"\n{len(checks) - len(bad)}/{len(checks)} guards OK"
        + (f" — SUSPECT: {', '.join(bad)}" if bad else "")
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
