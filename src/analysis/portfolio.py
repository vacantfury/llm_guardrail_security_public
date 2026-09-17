"""Portfolio ("best-of-all" / mixed) attack analysis.

The deployment-relevant question for the coverage-gap motivation (proposal
RQ1): faced with a *suite* of single-method attacks, does a defense still get
broken on some prompt? This is NOT a new query — it is an OR-reduction over the
per-prompt `asr` verdicts already stored in each `defense+evaluate` run's
raw_results.jsonl, joined on the stable prompt `id`.

For a fixed group (default: one model × one defense), a prompt is "broken by
the portfolio" if ANY attack variant in the group has asr=True for that id.

    portfolio_asr        = best-of-all (OR over attacks), per-prompt
    best_fixed_attack_asr= the single strongest attack alone
    complementarity_gap  = portfolio - best_fixed  (what the suite buys)

Per-id attribution (which attack broke each prompt) is written to
portfolio_rows.jsonl — that file is the coverage map.

No model/judge I/O; re-runnable from raw verdicts at any time, and the
attack_subset knob lets the same runs be re-sliced (text-only vs text+image
suite, etc.) for free.
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path
from typing import Any, Optional

from src.utils.experiment import get_new_experiment_data_dir
from src.utils.logger import get_logger
from src.utils.provenance import provenance_fields, sha256_file, write_json_atomic, write_jsonl_atomic

logger = get_logger(__name__)


# ------------------------------------------------------------------ helpers


def _expand_source_dirs(patterns: list[str]) -> list[Path]:
    """Expand globs and keep only dirs that actually hold a results.json."""
    seen: set[str] = set()
    out: list[Path] = []
    for pat in patterns:
        matches = glob.glob(pat) if any(c in pat for c in "*?[") else [pat]
        for m in sorted(matches):
            p = Path(m)
            if (p / "results.json").exists() and str(p) not in seen:
                seen.add(str(p))
                out.append(p)
            elif not (p / "results.json").exists():
                logger.warning(f"skip source (no results.json): {p}")
    return out


def _attack_label(res: dict) -> str:
    """A stable label for the attack variant a source dir represents.

    Within a (model, defense) group, the attack identity is the prompt-transform
    chain plus any decoy image. Two sources with the same chain+decoy are the
    same attack (e.g. reruns) and would be merged — usually you do not want that,
    so it is surfaced via a warning at group time.
    """
    chain = res.get("transformation_list") or []
    label = "+".join(chain) if chain else (res.get("encoding") or "unknown")
    decoy = res.get("companion_image_path")
    if decoy:
        label += f"[{Path(decoy).stem}]"
    return label


def _group_key(res: dict, group_by: list[str]) -> tuple[tuple[str, str], ...]:
    return tuple((k, str(res.get(k))) for k in group_by)


def _load_asr_by_id(source_dir: Path) -> dict[str, Optional[bool]]:
    """Per-id `asr` from raw_results.jsonl. None = excluded/unjudged (e.g.
    over-maxlen image, mechanism error / query-time overflow, upstream API
    filter). Only rows that are is_within_maxlen AND is_correctly_processed carry
    a real verdict; all excluded rows already have asr=None, so reading `asr`
    directly is sufficient (no need to re-filter here)."""
    out: dict[str, Optional[bool]] = {}
    rf = source_dir / "raw_results.jsonl"
    if not rf.exists():
        logger.warning(f"skip source (no raw_results.jsonl): {source_dir}")
        return out
    with open(rf) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["id"])] = row.get("asr")
    return out


# ------------------------------------------------------------------ core


def _analyze_group(
    sources: list[dict], denominator_policy: str,
) -> tuple[dict, list[dict]]:
    """Compute portfolio metrics + per-id attribution for one group.

    `sources` is a list of {label, asr_by_id} dicts (one per attack variant).
    Returns (group_metrics, attribution_rows)."""
    attacks = [s["label"] for s in sources]
    id_universe: set[str] = set()
    for s in sources:
        id_universe |= set(s["asr_by_id"].keys())

    evaluated_ids = {
        pid for pid in id_universe
        if any(s["asr_by_id"].get(pid) is not None for s in sources)
    }
    n_excluded_everywhere = len(id_universe) - len(evaluated_ids)

    if denominator_policy == "evaluated_union":
        denominator = len(evaluated_ids)
    else:  # full_source_set
        denominator = len(id_universe)

    # Per-attack success counts (same denominator → comparable to portfolio).
    per_attack_true: dict[str, int] = {a: 0 for a in attacks}
    n_broken = 0
    attribution: list[dict] = []
    for pid in sorted(id_universe):
        asr_by_attack = {s["label"]: s["asr_by_id"].get(pid) for s in sources}
        winning = [a for a, v in asr_by_attack.items() if v is True]
        for a in winning:
            per_attack_true[a] += 1
        broken = bool(winning)
        if broken:
            n_broken += 1
        attribution.append({
            "id": pid,
            "broken": broken,
            "winning_attacks": winning,
            "asr_by_attack": asr_by_attack,
        })

    def pct(n: int) -> float:
        return round(100.0 * n / denominator, 2) if denominator else 0.0

    per_attack_asr = {a: pct(per_attack_true[a]) for a in attacks}
    portfolio_asr = pct(n_broken)
    best_fixed = max(per_attack_asr.values()) if per_attack_asr else 0.0

    metrics = {
        "n_attacks": len(attacks),
        "attacks": attacks,
        "denominator": denominator,
        "n_broken": n_broken,
        "n_excluded_everywhere": n_excluded_everywhere,
        "portfolio_asr": portfolio_asr,
        "best_fixed_attack_asr": round(best_fixed, 2),
        "complementarity_gap": round(portfolio_asr - best_fixed, 2),
        "per_attack_asr": per_attack_asr,
    }
    return metrics, attribution


def run_analysis(task) -> dict[str, Any]:
    """Entry point for the `analyze` task mode. Writes results.json +
    portfolio_rows.jsonl under outputs/analyze/ and returns a summary dict."""
    t0 = time.time()
    if task.analysis != "portfolio_asr":
        raise ValueError(
            f"Unknown analysis {task.analysis!r}. Supported: 'portfolio_asr'.")

    source_dirs = _expand_source_dirs(task.source_dirs)
    if not source_dirs:
        raise ValueError(
            f"No source dirs with results.json matched {task.source_dirs!r}")

    warnings: list[str] = []

    # Load each source: results.json (grouping keys + label) + raw asr verdicts.
    loaded: list[dict] = []
    benchmarks: set[str] = set()
    for sd in source_dirs:
        res = json.loads((sd / "results.json").read_text())
        label = _attack_label(res)
        if task.attack_subset and label not in task.attack_subset:
            continue
        asr_by_id = _load_asr_by_id(sd)
        if not asr_by_id:
            warnings.append(f"empty verdicts: {sd}")
            continue
        if res.get("benchmark"):
            benchmarks.add(res["benchmark"])
        loaded.append({
            "dir": sd,
            "res": res,
            "label": label,
            "group": _group_key(res, task.group_by),
            "asr_by_id": asr_by_id,
        })

    if not loaded:
        raise ValueError(
            "No usable sources after filtering (attack_subset / empty verdicts).")
    if len(benchmarks) > 1:
        warnings.append(
            f"mixed benchmarks across sources: {sorted(benchmarks)} — "
            f"portfolio ids may not align; check group_by includes 'benchmark'.")

    # Group, then OR-reduce within each group.
    groups: dict[tuple, list[dict]] = {}
    for item in loaded:
        groups.setdefault(item["group"], []).append(item)

    group_results: list[dict] = []
    all_attribution: list[dict] = []
    total_ids = 0
    for gkey, items in sorted(groups.items()):
        # Merge-collision guard: two sources, same attack label in one group.
        labels = [it["label"] for it in items]
        dupes = {x for x in labels if labels.count(x) > 1}
        if dupes:
            warnings.append(
                f"group {dict(gkey)}: duplicate attack labels {sorted(dupes)} "
                f"(reruns?) — last-wins per id within a label is NOT applied; "
                f"both contribute to the OR. Dedup upstream if unintended.")
        sources = [{"label": it["label"], "asr_by_id": it["asr_by_id"]} for it in items]
        gmetrics, attribution = _analyze_group(sources, task.denominator_policy)
        group_key_dict = dict(gkey)
        total_ids += gmetrics["denominator"]
        group_results.append({"group_key": group_key_dict, **gmetrics})
        for row in attribution:
            all_attribution.append({**group_key_dict, **row})

    # Output dir + provenance.
    benchmark = sorted(benchmarks)[0] if len(benchmarks) == 1 else "mixed"
    out_dir = Path(get_new_experiment_data_dir(
        "outputs/analyze", dataset=benchmark, model=task.analysis))

    portfolio_values = [g["portfolio_asr"] for g in group_results]
    summary_metrics = {
        "mean_portfolio_asr": round(sum(portfolio_values) / len(portfolio_values), 2),
        "max_portfolio_asr": max(portfolio_values),
        "min_portfolio_asr": min(portfolio_values),
        "n_groups": float(len(group_results)),
    }

    from src.experiment.schemas import AnalyzeGroupResult, AnalyzeResult

    result = AnalyzeResult(
        analysis=task.analysis,
        source_dirs=[str(it["dir"]) for it in loaded],
        upstream_refs=[
            {"source_dir": str(it["dir"]),
             "results_sha256": sha256_file(it["dir"] / "results.json")}
            for it in loaded
        ],
        group_by=task.group_by,
        join_key=task.join_key,
        attack_subset=task.attack_subset,
        denominator_policy=task.denominator_policy,
        benchmark=benchmark,
        n_sources=len(loaded),
        n_groups=len(group_results),
        groups=[AnalyzeGroupResult(**g) for g in group_results],
        metrics=summary_metrics,
        primary_metric="mean_portfolio_asr",
        count=total_ids,
        elapsed_seconds=round(time.time() - t0, 2),
        output_dir=str(out_dir),
        status="success" if not warnings else "success_with_warnings",
        warnings=warnings,
        **provenance_fields(),
    )

    write_json_atomic(out_dir / "results.json", result.model_dump(exclude_none=True))
    write_jsonl_atomic(
        out_dir / "portfolio_rows.jsonl",
        [json.dumps(r) for r in all_attribution])
    logger.info(
        f"analyze[portfolio_asr]: {len(loaded)} sources → {len(group_results)} "
        f"groups; mean portfolio ASR={summary_metrics['mean_portfolio_asr']}% "
        f"→ {out_dir}")

    return {
        "status": "success",
        "output_dir": str(out_dir),
        "count": total_ids,
        "analysis": task.analysis,
        "n_sources": len(loaded),
        "n_groups": len(group_results),
        "mean_portfolio_asr": summary_metrics["mean_portfolio_asr"],
        "elapsed_seconds": round(time.time() - t0, 2),
    }
