"""Guard-threshold calibration sweep — a PAPER-AGNOSTIC analysis tool.

WHY THIS IS CHEAP (the load-bearing design idea)
------------------------------------------------
For a threshold tau, a behavior counts as broken iff

    guard PASSES it at tau   AND   the target's answer to it is harmful.

The second conjunct does not depend on the guard at all, so it can be reused
from runs already on disk: the no-defense floor records one judged response per
(attack, behavior). Sweeping therefore needs only a GUARD-ONLY forward pass to
capture per-prompt severity -- no target inference, no judge calls. The same
holds on the benign axis for over-refusal. A full 5-guard sweep is a handful of
cheap open-model passes, not a re-run of the experiment matrix.

TWO TIERS OF SEVERITY
---------------------
tier 1 (available with no new dependency): persist each guard's RAW output and
  map it to an ordinal level. Only Qwen3Guard-Gen is natively 3-way
  (Safe/Controversial/Unsafe), which the production path deliberately collapses
  to binary (`guard_utils.parse_qwen3guard`) -- so tier 1 already recovers a
  real 2-cut curve for that guard, and a single point for the binary ones.
tier 2 (needs plumbing): verdict-token logprobs give a continuous score and a
  dense curve for every guard. `llm_utils` v3.0.0 exposes no logprobs in any
  service, so tier 2 needs either a package bump or a guard-path-local direct
  call to the vLLM endpoint. This module is written so tier 2 drops in without
  changing callers: supply `score` instead of `raw`, and the sweep switches
  from ordinal cuts to the configured grid automatically.

Read-only over records produced by the capture pass; performs no model calls.
Tunables live in conf/analysis/guard_threshold.yaml -- nothing numeric here.

Usage
-----
    from src.analysis.guard_threshold import sweep_guard, load_config
    curve = sweep_guard(records, harmful_labels, benign_labels, cfg)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "conf" / "analysis" / "guard_threshold.yaml"


def load_config(path: Optional[Path] = None) -> dict:
    """Load the sweep tunables. All numbers used by this module come from here."""
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GuardRecord:
    """One guard verdict on one prompt.

    Severity is taken from the first of these that is present, most precise
    first:
      score:    P(unsafe), already reduced (tier 2).
      logprobs: the verbatim payload from `query_guard_scored` (tier 2) — we
                reduce it here, so the reduction can be revised without
                re-running any guard.
      raw:      the guard's verbatim completion (tier 1, ordinal).
    """
    id: str
    guard: str
    raw: Optional[str] = None
    score: Optional[float] = None
    logprobs: Optional[dict] = None
    chain: Optional[str] = None
    condition: Optional[str] = None

    def __post_init__(self) -> None:
        if self.raw is None and self.score is None and self.logprobs is None:
            raise ValueError(
                f"GuardRecord {self.id!r} carries no raw, score, or logprobs — "
                "nothing to threshold on.")


def _verdict_matcher(guard: str, cfg: dict):
    """Build this guard's token->side classifier from config.

    Returns `((safe_set, unsafe_set), side_fn)`, or None if the guard has no
    verdict vocabulary configured. Shared by `locate_verdict` (which token is
    the decision) and `verdict_probability` (how to bucket that token's
    alternatives) so the two can never disagree about what counts as a verdict.

    Match modes: `exact` requires the token text to BE a verdict word; `prefix`
    accepts any fragment that could only begin one of them, for guards whose
    tokenizer does not give the verdict a token of its own (see the config's
    `verdict_match` notes).
    """
    vocab = (cfg.get("verdict_tokens") or {}).get(guard)
    if not vocab:
        return None
    safe_set = {s.lower() for s in vocab.get("safe", [])}
    unsafe_set = {s.lower() for s in vocab.get("unsafe", [])}
    mode = (cfg.get("verdict_match") or {}).get(guard, "exact")

    def side(text: str) -> Optional[str]:
        """Which verdict this token indicates, or None if it is not a decision."""
        # Leading punctuation is stripped because a tokenizer may glue the
        # field's colon onto the verdict's first letter (':h' for "harmful").
        norm = text.strip().lstrip(":").strip().lower()
        if not norm:
            return None
        if mode == "prefix":
            is_safe = any(w.startswith(norm) for w in safe_set)
            is_unsafe = any(w.startswith(norm) for w in unsafe_set)
            # A fragment that could begin EITHER verdict decides nothing; keep
            # scanning rather than guessing which word it was going to become.
            if is_safe == is_unsafe:
                return None
            return "safe" if is_safe else "unsafe"
        if norm in safe_set:
            return "safe"
        if norm in unsafe_set:
            return "unsafe"
        return None

    return (safe_set, unsafe_set), side


def locate_verdict(
    logprobs: Optional[dict], guard: str, cfg: dict
) -> Optional[tuple[int, str, str]]:
    """Find the token that carries this guard's decision.

    Returns `(index, token_text, "safe"|"unsafe")`, or None when the verdict
    cannot be located — the caller then falls back to the ordinal tier rather
    than scoring some other token as if it were the decision.

    This is public so diagnostics report the SAME position the sweep scores.
    An independent re-implementation in the verifier previously scanned without
    the anchor and claimed GuardReasoner-VL matched "harmful" on 1199/1200
    records, for a guard that called 576 of them unharmful — a wrong answer
    from a tool whose whole job is to catch wrong answers.
    """
    if not logprobs:
        return None
    matcher = _verdict_matcher(guard, cfg)
    if matcher is None:
        return None
    _sets, side = matcher

    tokens = list(logprobs.get("content") or [])
    stop = {s.lower() for s in (cfg.get("verdict_stop") or {}).get(guard, [])}

    # REASONING guards state the verdict only AFTER a chain of thought that is
    # itself full of verdict words ("...would be harmful..."), so first-match
    # scanning reads the trace instead of the decision. An anchored guard names
    # the marker its verdict follows; we scan from the LAST occurrence of that
    # marker, which is the one inside the final result block.
    #
    # Measured on GuardReasoner-VL: unanchored matching agreed with the guard's
    # own parsed verdict 52% of the time — a coin flip — because the first
    # "harmful" token sat ~90 tokens deep in the trace. Anchoring is what makes
    # the number mean anything.
    anchors = {a.lower() for a in (cfg.get("verdict_anchor") or {}).get(guard, [])}
    if anchors:
        last = -1
        for i, tok in enumerate(tokens):
            if (tok.get("token") or "").strip().lower() in anchors:
                last = i
        if last < 0:
            # No anchor -> the verdict cannot be located. Return None so the
            # caller falls back to the ordinal tier rather than scoring a token
            # from the reasoning trace as if it were the decision.
            return None
        offset = last + 1
        tokens = tokens[offset:]
    else:
        offset = 0

    for i, tok in enumerate(tokens):
        text = tok.get("token") or ""
        # A guard may state SEVERAL verdicts (GuardReasoner-VL grades the
        # request and the response separately, one after the other). Stop at the
        # next field's marker so an unmatched verdict token can never let the
        # scan slide into a different field's answer — which is exactly what
        # happened before prefix mode: the request verdict ':h' matched nothing,
        # so the scan ran on and scored the RESPONSE verdict instead, producing
        # p_unsafe=0.0 on requests the guard had called harmful.
        if text.strip().lower() in stop:
            return None
        which = side(text)
        if which is None:
            continue
        return offset + i, text, which
    return None


def verdict_probability(
    logprobs: Optional[dict], guard: str, cfg: dict
) -> Optional[float]:
    """Reduce a captured logprobs payload to P(unsafe) at the verdict token.

    Locates the verdict with `locate_verdict`, then sums the probabilities of
    the unsafe alternatives among that token's `top_logprobs`.

    Renormalises over the safe+unsafe alternatives actually present, so the
    result is a genuine two-way decision probability rather than a raw softmax
    mass that silently loses weight to unrelated tokens. Returns None when the
    payload is missing or no verdict token is found — the caller then falls
    back to the ordinal tier, never to a fabricated number.
    """
    found = locate_verdict(logprobs, guard, cfg)
    if found is None:
        return None
    index, _text, which = found

    _sets, side = _verdict_matcher(guard, cfg)   # non-None: locate_verdict passed
    alts = (logprobs.get("content") or [])[index].get("top_logprobs") or []
    p_safe = p_unsafe = 0.0
    for alt in alts:
        a_side = side(alt.get("token") or "")
        if a_side is None:
            continue
        p = math.exp(alt["logprob"])
        if a_side == "unsafe":
            p_unsafe += p
        else:
            p_safe += p
    total = p_safe + p_unsafe
    if total <= 0:
        # The verdict token was emitted but no alternative was gradable —
        # fall back to the hard verdict it actually produced.
        return 1.0 if which == "unsafe" else 0.0
    return p_unsafe / total


def load_records(path: Path) -> list[GuardRecord]:
    """Read a capture-pass JSONL into GuardRecords (unknown keys ignored)."""
    out: list[GuardRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(GuardRecord(
                id=d["id"], guard=d["guard"], raw=d.get("raw"),
                score=d.get("score"), logprobs=d.get("logprobs"),
                chain=d.get("chain"), condition=d.get("condition")))
    return out


# --------------------------------------------------------------------------
# severity
# --------------------------------------------------------------------------
def severity_of(rec: GuardRecord, cfg: dict) -> float:
    """Map a record to a severity in [0, 1], higher = more unsafe.

    Continuous scores pass through. Ordinal raw verdicts map onto an evenly
    spaced ladder over the configured levels, so both tiers share one sweep.
    Anything unparseable FAILS CLOSED (maximum severity) — mirroring
    guard_utils' own contract, so a broken parse can never silently look safe.
    """
    if rec.score is not None:
        return float(rec.score)

    p = verdict_probability(rec.logprobs, rec.guard, cfg)
    if p is not None:
        return p

    levels: list[str] = cfg["severity"]["levels"]
    omap: dict = cfg["severity"]["ordinal_map"]
    table = omap.get(rec.guard) or omap["_binary_default"]

    text = (rec.raw or "")
    level = None
    for key, lvl in table.items():
        if key.lower() in text.lower():
            # Prefer the most severe match present, not the first seen.
            if level is None or levels.index(lvl) > levels.index(level):
                level = lvl
    if level is None:
        return 1.0  # fail closed
    return levels.index(level) / (len(levels) - 1)


def thresholds_for(records: Iterable[GuardRecord], cfg: dict) -> list[float]:
    """The cut points to sweep.

    Ordinal guards get exactly the cuts their levels can express (reporting a
    dense grid for a 3-way guard would fake resolution it does not have);
    continuous guards get the configured grid.
    """
    recs = list(records)
    if any(r.score is not None or r.logprobs is not None for r in recs):
        s = cfg["sweep"]
        n = int(round((s["grid_stop"] - s["grid_start"]) / s["grid_step"])) + 1
        # Round each cut: repeated addition of a float step accumulates error
        # (0.02 * 15 = 0.30000000000000004), which would make thresholds ugly
        # in reports and unreliable as dict keys for callers.
        return [round(s["grid_start"] + i * s["grid_step"], 10) for i in range(n)]
    levels = cfg["severity"]["levels"]
    return [i / (len(levels) - 1) for i in range(len(levels))]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def wilson(k: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval — the same interval the paper reports elsewhere."""
    if n == 0:
        return (0.0, 0.0)
    # two-sided z from the normal quantile (Acklam-free rational approximation
    # is overkill here; the paper only ever uses 0.95 and 0.99)
    z = {0.90: 1.6448536269, 0.95: 1.9599639845, 0.99: 2.5758293035}.get(
        round(confidence, 2))
    if z is None:
        raise ValueError(f"wilson: unsupported confidence {confidence!r}")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------
@dataclass
class OperatingPoint:
    threshold: float
    ensemble_asr: float
    over_refusal: float
    n_harmful: int
    n_benign: int
    asr_ci: tuple[float, float] = (0.0, 0.0)
    refusal_ci: tuple[float, float] = (0.0, 0.0)
    usable: bool = False


@dataclass
class GuardCurve:
    guard: str
    points: list[OperatingPoint] = field(default_factory=list)

    def pareto(self) -> list[OperatingPoint]:
        """Distinct non-dominated trade-offs, sorted by over-refusal.

        Deduplicated by (ASR, over-refusal): on a dense grid many adjacent
        thresholds land on exactly the same operating point, and since ties do
        not dominate one another they would all survive — reporting 39 "choices"
        that are really 3. We keep the LOWEST threshold achieving each distinct
        point, that being the least aggressive way to reach it.
        """
        out: list[OperatingPoint] = []
        for p in self.points:
            if not any(q is not p
                       and q.ensemble_asr <= p.ensemble_asr
                       and q.over_refusal <= p.over_refusal
                       and (q.ensemble_asr < p.ensemble_asr
                            or q.over_refusal < p.over_refusal)
                       for q in self.points):
                out.append(p)
        best: dict[tuple[float, float], OperatingPoint] = {}
        for p in sorted(out, key=lambda p: p.threshold):
            best.setdefault((p.ensemble_asr, p.over_refusal), p)
        return sorted(best.values(), key=lambda p: p.over_refusal)

    def best_usable(self, cfg: dict) -> Optional[OperatingPoint]:
        """Lowest-ASR point that stays under the configured over-refusal bar.

        Returns None when the guard has NO usable point — which is itself the
        paper's ceiling claim, stated as a measurement instead of an assertion.
        """
        usable = [p for p in self.points if p.usable]
        return min(usable, key=lambda p: p.ensemble_asr) if usable else None


def sweep_guard(
    records: Iterable[GuardRecord],
    harmful_labels: dict[tuple[str, str], bool],
    benign_labels: dict[str, bool],
    cfg: Optional[dict] = None,
) -> GuardCurve:
    """Sweep one guard's decision threshold.

    Args:
        records: guard verdicts from the capture pass. Harmful-side records
            must carry `chain` (which attack); benign-side records must not.
        harmful_labels: (chain, behavior_id) -> was the TARGET's answer
            harmful. Reused from the no-defense floor runs — this is what makes
            the sweep cheap.
        benign_labels: (benign_condition, behavior_id) -> did the pipeline
            refuse this benign input for reasons OTHER than the guard (the
            no-defense floor). Keyed by condition because each benign variant
            has its OWN floor (measured: 17% for the image render vs 38% for
            the text baseline) — one shared floor would mis-attribute the
            difference to the guard.
        cfg: parsed conf/analysis/guard_threshold.yaml.

    Returns:
        A GuardCurve with one OperatingPoint per swept threshold. Ensemble ASR
        is the OR-reduction over attacks (Eq. 1 in the paper): a behavior counts
        as broken if ANY attack both passes the guard at this threshold and
        yields a harmful answer.
    """
    cfg = cfg or load_config()
    recs = list(records)
    if not recs:
        raise ValueError("sweep_guard: no records")
    guards = {r.guard for r in recs}
    if len(guards) != 1:
        raise ValueError(f"sweep_guard: expected ONE guard, got {sorted(guards)}")

    harmful = [r for r in recs if r.chain is not None]
    benign = [r for r in recs if r.chain is None]

    # Key on CONDITION as well as chain. Every benign variant carries chain=None
    # and the same behavior ids, so a (guard, id, chain) key silently collapses
    # them: the last variant read would overwrite the others, and the sweep
    # would report one variant's severities under every variant's name. Observed
    # before this fix — the text variant's 75% block rate disappeared behind the
    # image variant's 0%, dragging over-refusal down to the benign floor and
    # contradicting the paper's own published benign cells.
    sev = {(r.guard, r.id, r.chain, r.condition): severity_of(r, cfg)
           for r in recs}
    if len(sev) != len(recs):
        raise ValueError(
            f"sweep_guard: {len(recs) - len(sev)} records share a "
            "(guard, id, chain, condition) key — severities would overwrite "
            "each other. Check the capture for duplicate cells.")

    behaviors = {r.id for r in harmful}
    min_n = cfg["sweep"]["min_cell_n"]
    conf = cfg["report"]["wilson_confidence"]
    bar = cfg["sweep"]["usable_over_refusal_max"]

    curve = GuardCurve(guard=next(iter(guards)))
    for tau in thresholds_for(recs, cfg):
        # ---- safety axis: OR over attacks, guard must PASS (severity < tau)
        broken = set()
        for r in harmful:
            if sev[(r.guard, r.id, r.chain, r.condition)] >= tau:
                continue                      # guard blocked it
            if harmful_labels.get((r.chain, r.id), False):
                broken.add(r.id)
        n_h = len(behaviors)
        asr = len(broken) / n_h if n_h else 0.0

        # ---- utility axis: benign refused by the guard OR already refused
        #
        # Averaged ACROSS benign variants (the paper reports over-refusal as
        # avg(text, image), and the two differ enough that picking one would
        # change the answer: WildGuard refuses 17% of the image-rendered benign
        # set and 81% of the text one). Each variant contributes equally
        # regardless of its size, matching the published mean-of-rates; with a
        # single variant present this reduces to the plain rate.
        per_variant: dict[str, list[int]] = {}
        for r in benign:
            cond = r.condition or "benign"
            blocked = (sev[(r.guard, r.id, None, r.condition)] >= tau
                       or benign_labels.get((cond, r.id), False))
            hit = per_variant.setdefault(cond, [0, 0])
            hit[0] += int(blocked)
            hit[1] += 1
        n_b = len(benign)
        refused = sum(k for k, _ in per_variant.values())
        orr = (sum(k / n for k, n in per_variant.values() if n) / len(per_variant)
               if per_variant else 0.0)

        curve.points.append(OperatingPoint(
            threshold=tau, ensemble_asr=asr, over_refusal=orr,
            n_harmful=n_h, n_benign=n_b,
            asr_ci=wilson(len(broken), n_h, conf) if n_h >= min_n else (0.0, 0.0),
            refusal_ci=wilson(refused, n_b, conf) if n_b >= min_n else (0.0, 0.0),
            usable=(orr <= bar)))
    return curve


def format_curve(curve: GuardCurve, cfg: Optional[dict] = None) -> str:
    """Human-readable sweep table, mirroring the paper's reporting style."""
    cfg = cfg or load_config()
    d = cfg["report"]["percent_decimals"]
    lines = [f"=== {curve.guard} — threshold sweep ==="]
    lines.append(f"{'tau':>6} {'ens ASR':>9} {'over-refusal':>14}  usable")
    for p in curve.points:
        lines.append(
            f"{p.threshold:6.2f} {100 * p.ensemble_asr:8.{d}f}% "
            f"{100 * p.over_refusal:13.{d}f}%  {'yes' if p.usable else 'no'}")
    best = curve.best_usable(cfg)
    lines.append(
        f"  best usable point: {100 * best.ensemble_asr:.{d}f}% ASR at "
        f"{100 * best.over_refusal:.{d}f}% over-refusal (tau={best.threshold:.2f})"
        if best else
        "  NO usable operating point at any threshold — the guard cannot be "
        "calibrated under the configured over-refusal bar.")
    return "\n".join(lines)
