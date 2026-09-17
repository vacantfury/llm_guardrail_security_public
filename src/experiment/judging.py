"""
Judging orchestration + the results-IO helpers every judging stage shares.

Split out of task.py (pure code move, no behavior change): the judge cluster is
consumed by BOTH `defense+evaluate` (src/experiment/task.py) and `rejudge`
(src/experiment/stage_rejudge.py). The small results-IO / benchmark-inference
helpers live here rather than in task.py so stage_rejudge.py can reach them
without importing task.py (which would be a circular import). task.py re-exports
every name below, so `from src.experiment.task import <name>` keeps working.
"""
import json
import statistics
from pathlib import Path
from typing import Any, Optional

from src.evaluation.evaluator_factory import EvaluatorFactory
from src.experiment.config import load_conf as _load_conf
from src.experiment.schemas import EvaluationRow, Prompt
from llm_utils.llm_model import LLMModel
from src.utils.logger import get_logger
from src.utils.provenance import (
    judge_config_hash, sha256_file, write_json_atomic,
)

logger = get_logger(__name__)


# ======================== Shared helpers ========================


def _save_results(out_dir: Path, results: dict[str, Any]) -> None:
    write_json_atomic(out_dir / "results.json", results)
    logger.info(f"Saved results.json to {out_dir}")


def _load_results(source_dir: str) -> dict[str, Any]:
    path = Path(source_dir) / "results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_prompts(source_dir: str) -> list[Prompt]:
    path = Path(source_dir) / "prompts.jsonl"
    out: list[Prompt] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(Prompt.model_validate_json(line))
    return out


def _upstream_ref(source_dir) -> dict:
    """Lightweight provenance pointer to a consumed results.json.

    Replaces embedding the whole upstream dict (O(chain-depth) bloat + the
    embedded copy could drift from its source). Follow `source_dir` to
    reconstruct full provenance; `results_sha256` detects drift.
    """
    src = Path(source_dir)
    ref: dict = {"source_dir": str(src)}
    rp = src / "results.json"
    if rp.exists():
        ref["results_sha256"] = sha256_file(rp)
    return ref


# Benchmark inference (shared by stage 1 and stage 2).

BENCHMARK_ALIASES = {
    "jbb_benign": "jailbreakbench_benign",
    "jbb": "jailbreakbench",
    "orbench_benign_hard": "orbench_benign_hard",
    "orbench_benign_1k": "orbench_benign_1k",
    "orbench_harmful": "orbench_harmful",
}

KNOWN_BENCHMARKS = frozenset({
    "harmbench", "jailbreakbench", "jailbreakbench_benign",
    "orbench_harmful", "orbench_benign_hard", "orbench_benign_1k",
})


def _infer_benchmark(source_path: str) -> str:
    """Path/filename match against KNOWN_BENCHMARKS / BENCHMARK_ALIASES.

    Order:
      1. Canonical benchmark name as a path component (e.g. .../harmbench/...).
      2. Alias substring match (e.g. jbb_benign → jailbreakbench_benign).
      3. Canonical benchmark name as a filename substring (e.g.
         harmbench_prompts.jsonl → harmbench). Longest-match-first to ensure
         "jailbreakbench_benign" matches before "jailbreakbench".
    """
    parts = Path(source_path).parts
    for part in parts:
        if part in KNOWN_BENCHMARKS:
            return part
    name = str(source_path).lower()
    for alias, bench in sorted(BENCHMARK_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in name:
            return bench
    for bench in sorted(KNOWN_BENCHMARKS, key=len, reverse=True):
        if bench in name:
            return bench
    raise ValueError(
        f"Could not infer benchmark from path: {source_path!r}. "
        f"Set `benchmark:` explicitly in the task config or place the data "
        f"under a path containing one of {sorted(KNOWN_BENCHMARKS)}.")


# ======================== Judge orchestration ========================


def _resolve_evaluators(
    judge_method_override: Optional[str], benchmark: str,
    judge_model_override: Optional[str] = None,
) -> list:
    """Resolve the canonical evaluator list + bind the shared judge LLM."""
    eval_config = _load_conf("evaluation")
    judge_cfg = dict(eval_config.get("judge_llm_config", {}))
    default_model = judge_cfg.pop("model", "gpt-5-nano")
    judge_model_str = judge_model_override or default_model
    judge_model = LLMModel.from_string(judge_model_str)
    if judge_method_override:
        logger.warning(
            f"judge_method override active: '{judge_method_override}' "
            f"(bypassing canonical benchmark→evaluator mapping for {benchmark!r})")
        return [EvaluatorFactory.create(
            method=judge_method_override, model=judge_model, **judge_cfg)]
    return EvaluatorFactory.create_from_benchmark(
        benchmark, model=judge_model, **judge_cfg)


def _is_refusal_evaluator(evaluator) -> bool:
    from src.evaluation.evaluator_factory import REFUSAL_RATE_EVALUATORS
    return isinstance(evaluator, REFUSAL_RATE_EVALUATORS)


def _apply_verdict_columns(stage_rows: list, detailed_df, is_refusal: bool) -> None:
    if detailed_df is None:
        return

    def _col(df, name) -> dict:
        if name not in df.columns:
            return {}
        return dict(zip(df["id"].astype(str), df[name].astype(str)))

    verdict_col = "is_refused" if is_refusal else "is_jailbroken"
    if verdict_col not in detailed_df.columns:
        return
    verdict_by_id = dict(zip(detailed_df["id"].astype(str), detailed_df[verdict_col]))
    output_by_id = _col(detailed_df, "judge_output")
    reasoning_by_id = _col(detailed_df, "judge_reasoning")
    raw_by_id = _col(detailed_df, "judge_raw_response")

    if is_refusal:
        for row in stage_rows:
            row.refusal = verdict_by_id.get(row.id)
            row.refusal_judge_output = output_by_id.get(row.id)
            row.refusal_judge_reasoning = reasoning_by_id.get(row.id)
            row.refusal_judge_raw_response = raw_by_id.get(row.id)
    else:
        for row in stage_rows:
            row.asr = verdict_by_id.get(row.id)
            row.judge_output = output_by_id.get(row.id)
            row.judge_reasoning = reasoning_by_id.get(row.id)
            row.judge_raw_response = raw_by_id.get(row.id)


def _dropped_row_warnings(
    all_rows: list[EvaluationRow], within_rows: list[EvaluationRow],
    stage_label: str,
) -> list[str]:
    """Flag draws that never reached the judge, so a thinned denominator is loud.

    Rows whose model call failed are marked is_correctly_processed=False and dropped
    before judging, which silently shrinks the denominator: the reported rate is then
    computed over the SURVIVORS while `count` still says how many were intended. For a
    best-of-N attack that biases the union metric DOWNWARD -- fewer draws, fewer chances
    for any draw to succeed -- so a defense looks stronger than it is.

    This is not hypothetical. The 2026-07-24 Gemma recovery run lost 900/10000 draws per
    code cell to `prompt_tokens + max_tokens > max_model_len` 400s and still wrote
    `status: success, warnings: []` with ASR 0.00, because `_run_judging`'s own guard
    compares against the already-filtered list and saw 9100 == 9100.

    Returned strings land in the result's judge_errors -> `status: partial_judge`.
    """
    dropped = len(all_rows) - len(within_rows)
    if dropped <= 0:
        return []
    total = len(all_rows)
    over = sum(1 for r in all_rows if not r.is_within_maxlen)
    failed = sum(1 for r in all_rows if not r.is_correctly_processed)
    msg = (f"{dropped}/{total} rows ({dropped / total:.1%}) never reached the judge "
           f"({failed} failed model calls, {over} over length budget) — the reported "
           f"rate is computed over {len(within_rows)} surviving draws, not {total}")
    logger.error(f"[{stage_label}] THINNED DENOMINATOR — {msg}")
    return [msg]


def draw_diversity_stats(all_rows: list[EvaluationRow], stage_label: str) -> dict:
    """Measure whether a best-of-N cell's draws are actually DIFFERENT.

    A best-of-N number is only meaningful if the N draws are independent samples.
    Two separate things can silently collapse them into one:
      * a DETERMINISTIC transform (every draw of a behavior gets the same prompt), and
      * a DETERMINISTIC target (temperature 0 / greedy decoding).
    With both, the only thing separating draws is the serving stack's batching
    nondeterminism, and `union ASR@N` degenerates into an OR over numerical noise.

    Reported, never fatal. Low diversity is legitimate when a target refuses uniformly
    (a well-defended cell genuinely emits the same refusal every time), so this cannot
    be an error without false-positiving on the strongest defense results. It is a
    number for a human to read next to the ASR.

    Returns {} for single-draw runs (no `__bonK` ids), else the stats dict that gets
    merged into the result's `metrics`.
    """
    by_behavior: dict[str, list[str]] = {}
    for row in all_rows:
        if "__" not in row.id:
            continue
        by_behavior.setdefault(row.id.rsplit("__", 1)[0], []).append(row.response or "")
    if not by_behavior:
        return {}
    draws = [len(v) for v in by_behavior.values()]
    if max(draws) < 2:
        return {}          # not a multi-draw round

    distinct = [len(set(v)) for v in by_behavior.values()]
    med_draws = statistics.median(draws)
    med_distinct = statistics.median(distinct)
    ratio = (med_distinct / med_draws) if med_draws else 0.0
    stats = {
        "draws_per_behavior_median": med_draws,
        "distinct_responses_per_behavior_median": med_distinct,
        "draw_diversity_ratio": round(ratio, 4),
    }
    msg = (f"draw diversity: median {med_distinct:.0f} distinct responses per "
           f"{med_draws:.0f} draws (ratio {ratio:.2f})")
    if ratio < 0.5:
        logger.warning(
            f"[{stage_label}] LOW DRAW DIVERSITY — {msg}. If this cell is meant to be "
            f"best-of-N, its draws are near-duplicates and union ASR@N / QtFS are not "
            f"interpretable as a search. Check the target temperature (>0?) and whether "
            f"the transform varies per draw. Legitimate only if the target refuses "
            f"uniformly.")
    else:
        logger.info(f"[{stage_label}] {msg}")
    return stats


def _run_judging(
    prompts: list[Prompt], all_rows: list[EvaluationRow], benchmark: str,
    judge_method_override: Optional[str], stage_label: str,
    judge_model_override: Optional[str] = None,
) -> dict:
    """Single-stage judging: runs every canonical evaluator on `all_rows`.

    NOTE: callers pass only the SURVIVING rows here (within-budget and correctly
    processed), so this function's own coverage guard can only see judging losses.
    Losses from failed model calls happen one level up and are caught by
    `_dropped_row_warnings` — see the comment there before changing either.

    Returns a dict: asr, refusal, eval_stats, judge_config_hash, judge_errors,
    metrics (every named scalar metric), primary_metric (the headline metric's
    name). asr/refusal are scalars or None if no judge of that type ran;
    non-empty judge_errors means metrics are incomplete (status "partial_judge").
    """
    evaluators = _resolve_evaluators(judge_method_override, benchmark, judge_model_override)
    asr: Optional[float] = None
    refusal: Optional[float] = None
    eval_stats: dict[str, dict] = {}
    metrics: dict[str, float] = {}
    judge_errors: list[str] = []

    original_lookup = {p.id: p.original for p in prompts}

    for evaluator in evaluators:
        is_refusal = _is_refusal_evaluator(evaluator)
        stat_key = "refusal_rate" if is_refusal else "attack_success_rate"
        try:
            judge_prompts = [
                {"id": r.id, "prompt": original_lookup.get(r.id, "")} for r in all_rows
            ]
            judge_processed = [original_lookup.get(r.id, "") for r in all_rows]
            judge_responses = {r.id: r.response for r in all_rows}
            detailed_df, stats = evaluator.evaluate(
                prompts=judge_prompts,
                processed_prompts=judge_processed,
                responses=judge_responses,
            )
            _apply_verdict_columns(all_rows, detailed_df, is_refusal)
            value = stats.get(stat_key, 0.0)

            # ---- coverage guard: never report a rate over rows we didn't judge ----
            # Every evaluator computes `rate = hits/total if total > 0 else 0.0`, so a
            # run whose responses ALL failed yields a confident-looking 0.00 that reads
            # as a perfect defense instead of a dead cell. That is not hypothetical:
            # round 174759's six Gemma-2-9B cells each shipped ASR 0.00 over 10000 draws
            # where every single target call had 400'd (max_tokens 16384 > max_model_len
            # 8192) and the judge scored the error strings as "not harmful"; the only
            # tell was total_evaluated=0 sitting next to count=10000 in the metadata.
            # A rate over zero rows is not a number — refuse to emit one.
            expected = len(all_rows)
            covered = stats.get("total_evaluated")
            if covered is not None and expected and covered != expected:
                msg = (f"{evaluator.__class__.__name__}: judged {covered}/{expected} "
                       f"responses ({covered / expected:.1%} coverage) — "
                       f"{stat_key} is computed over an incomplete set")
                judge_errors.append(msg)
                logger.error(f"[{stage_label}] COVERAGE FAILURE — {msg}")
                if covered == 0:
                    value = None

            if is_refusal:
                refusal = value
            else:
                asr = value
            eval_stats[evaluator.__class__.__name__] = stats
            # Record every scalar stat under its real name — no overloading.
            # When coverage collapsed to zero the rates are meaningless, so they are
            # kept OUT of `metrics` (and therefore out of primary_metric): the results
            # file then carries no headline number at all, which is the honest outcome.
            if value is not None:
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)
            logger.info(
                f"[{stage_label}] {evaluator.__class__.__name__}: "
                + (f"{stat_key}={value:.2f}%" if value is not None
                   else f"{stat_key}=UNAVAILABLE (0/{expected} responses judged)"))
        except (ValueError, KeyError, FileNotFoundError) as e:
            judge_errors.append(f"{evaluator.__class__.__name__}: skipped ({e})")
            logger.warning(
                f"Judge skipped for {evaluator.__class__.__name__}: {e}")
        except Exception as e:
            judge_errors.append(f"{evaluator.__class__.__name__}: failed ({e})")
            logger.warning(
                f"Judge failed for {evaluator.__class__.__name__}: {e}",
                exc_info=True)

    eval_cfg = _load_conf("evaluation").get("judge_llm_config", {})
    jhash = judge_config_hash(
        eval_cfg, [e.__class__.__name__ for e in evaluators])
    # Headline metric, named honestly (no benchmark-specific overloading):
    # direct_answer_rate (OR-Bench) > attack_success_rate > refusal_rate.
    primary_metric = next(
        (m for m in ("direct_answer_rate", "attack_success_rate", "refusal_rate")
         if m in metrics), None)
    return {
        "asr": asr, "refusal": refusal, "eval_stats": (eval_stats or None),
        "judge_config_hash": jhash, "judge_errors": judge_errors,
        "metrics": metrics, "primary_metric": primary_metric,
    }
