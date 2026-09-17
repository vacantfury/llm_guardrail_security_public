"""
Task runner for the destroyer-paper pipeline.

Modes:

  prompt_transform  — run a chain of PromptTransformations (attacks + image
                      renderers). Writes one subfolder per step under the
                      task output dir, each with its own cumulative
                      results.json.

  defense+evaluate  — defense + target-model query + judging, all in one
                      task. Consumes a specific subfolder from a
                      prompt_transform run.

  analyze           — pure post-processing over already-run results (no model
                      or judge I/O). Fans IN from many defense+evaluate dirs
                      and writes derived metrics (e.g. portfolio / best-of-all
                      ASR) under outputs/analyze/. See src/analysis/.
"""
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.evaluation.evaluator_factory import EvaluatorFactory
from src.defense import create_defense
from src.experiment.config import load_conf as _load_conf
from src.experiment.schemas import (
    DefenseEvaluateResult, EvaluationRow, Prompt, PromptTransformResult,
    PromptTransformStepResult, RawPrompt, RejudgeResult, TargetModelConfig,
    TransformationSpec,
)
from llm_utils import LLMServiceFactory
from llm_utils.base_llm_service import is_mechanism_error, strip_mechanism_error
from llm_utils.llm_model import LLMModel
from src.prompt_transformations import (
    Modality, create_transformation, resolve_transformation_name,
)
from src.utils.experiment import get_new_experiment_data_dir
from src.utils.logger import get_logger
from src.utils.provenance import (
    judge_config_hash, provenance_fields, sha256_file, write_json_atomic,
    write_jsonl_atomic,
)
from src.experiment.judging import (  # noqa: F401  (re-exported for back-compat)
    BENCHMARK_ALIASES, KNOWN_BENCHMARKS, _apply_verdict_columns,
    _dropped_row_warnings, draw_diversity_stats, _infer_benchmark, _is_refusal_evaluator,
    _load_prompts, _load_results, _resolve_evaluators, _run_judging,
    _save_results, _upstream_ref,
)
from src.experiment.stage_rejudge import (  # noqa: F401  (re-exported for back-compat)
    _load_eval_rows, _run_rejudge,
)

logger = get_logger(__name__)

# Max rendered pages (images) a single multimodal prompt may carry before it is
# excluded from the target query + ASR (is_within_maxlen=False) rather than
# truncated. A proxy for total content length, which is what actually drives
# vision-token usage against the served --max-model-len (20480 as of 2026-06-11;
# raised from 16384 to fit most code_attack ir_plain, but NOT to 32768 — that
# risks V100-32GB KV-cache OOM on Pixtral-12B). Any prompt that still overflows
# at query time is excluded via is_correctly_processed (not miscounted), so this
# page cap + the max-model-len are a best-effort reduction, not a guarantee.
# Tune down if the re-probe shows requests erroring below this page count.
# YAML home: conf/imaging/default.yaml `max_pages_per_prompt` (audit 2026-07-24);
# this constant is the fail-safe default when the YAML key is absent/unreadable.
MAX_PAGES_PER_PROMPT = 8


def _max_pages_per_prompt() -> int:
    """Resolve the page cap from conf/imaging/default.yaml, falling back to the
    module constant."""
    try:
        from src.experiment.config import load_conf
        return int(load_conf("imaging").get(
            "max_pages_per_prompt", MAX_PAGES_PER_PROMPT))
    except Exception:
        return MAX_PAGES_PER_PROMPT


# ======================== Shared helpers ========================


def _extract_companion_image_path(upstream: dict) -> Optional[str]:
    """First companion/decoy `image_path` anywhere in the upstream chain.

    Done once at write time so the decoy variant becomes a first-class field
    on results.json, instead of every reader walking
    `upstream.results_history[i].<step>.config.image_path`.
    """
    stack = [upstream]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        for rec in node.get("results_history", []) or []:
            if not isinstance(rec, dict):
                continue
            for step in rec.values():
                cfg = step.get("config", {}) if isinstance(step, dict) else {}
                if isinstance(cfg, dict) and cfg.get("image_path"):
                    return cfg["image_path"]
        if isinstance(node.get("upstream"), dict):
            stack.append(node["upstream"])
    return None


def _behavior_of(item) -> Optional[str]:
    """Strip a best-of-N draw suffix off an id: 'foo__bon37' -> 'foo'.

    Draw ids in this repo are '<behavior>__bon<k>'. Returns None for anything
    that doesn't carry an id, so callers can skip the check rather than guess.
    """
    pid = getattr(item, "id", None)
    if pid is None and isinstance(item, dict):
        pid = item.get("id")
    if not isinstance(pid, str):
        return None
    return re.sub(r"__bon\d+$", "", pid)


def _warn_if_single_behavior(kept: list, total: int, prompt_range) -> None:
    """Shout when a prompt_range slice collapses onto one or two behaviors.

    Best-of-N data is BEHAVIOR-MAJOR — 100 behaviors x 100 draws, so indices
    0-99 are all 100 draws of the FIRST behavior and nothing else. A contiguous
    `prompt_range: [0, 99]` therefore looks like a 100-prompt pilot and is
    actually a single-item study, which is not a detectable condition from the
    result numbers alone: it just reports a confident rate for one behavior.

    This exists because it already cost us a wrong conclusion. The P6 pilot
    (2026-07-30) ran [0, 99] = 100 draws of `korean_war_north_defensive`, one of
    HarmBench's mildest items, measured SelfDefend blocking 0/100, and was
    reported as "the shadow is blind to the code encoding". The full 10,000-draw
    run blocked 82%. Same defense, same target, same encoding.

    A pilot subset must SAMPLE ACROSS behaviors. With no stride support in
    `prompt_range`, use a span wide enough to cover many behaviors (e.g.
    [0, 1999] for 20) and read the behavior count this logs.
    """
    if not kept:
        return
    behaviors = {b for b in (_behavior_of(i) for i in kept) if b is not None}
    if not behaviors:
        return
    n = len(behaviors)
    logger.info(f"prompt_range {prompt_range}: kept {len(kept)}/{total} draws "
                f"spanning {n} distinct behavior(s)")
    if n <= 2 and len(kept) >= 20:
        logger.warning(
            f"⚠️  prompt_range {prompt_range} kept {len(kept)} draws but only "
            f"{n} distinct behavior(s): {sorted(behaviors)}. Best-of-N data is "
            f"behavior-major, so a contiguous head is ONE behavior's draw cloud, "
            f"NOT a sample of the benchmark. Any rate computed here describes "
            f"that one behavior. Widen the range to span behaviors before "
            f"drawing a conclusion from it.")


def _apply_prompt_range(items: list, prompt_range: Optional[list[int]]) -> list:
    if not prompt_range or len(prompt_range) != 2:
        return items
    try:
        start, end = int(prompt_range[0]), int(prompt_range[1])
    except (ValueError, TypeError):
        return items
    start = max(0, start)
    end = min(len(items) - 1, end)
    if start > end:
        return items
    kept = items[start : end + 1]
    _warn_if_single_behavior(kept, len(items), prompt_range)
    return kept


# ======================== prompt_transform ========================


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_raw_prompts(path: str) -> list[RawPrompt]:
    out: list[RawPrompt] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(RawPrompt.model_validate_json(line))
    return out


def _resolve_step_config(
    type_name: str, task_params: dict[str, Any],
) -> dict[str, Any]:
    """3-layer config merge for a single chain step.

    Legacy YAML files use the user-facing alias name (`set_theory.yaml`,
    `plain.yaml`), not the registry's canonical name (`llm_set_theory`,
    `ir_plain`). To bridge that, try the lookup with multiple candidate
    names per transformation:
        1. user-input `type_name` as-is (matches alias YAMLs)
        2. canonical type_name (matches the rare new YAMLs that use it)
        3. for `ir_*` renderers: also try the un-prefixed legacy name
           (`ir_plain` → `plain.yaml`)

    Only uses a YAML if it actually exists. Falls through to bare task_params
    for new transformations (deep_inception, code_attack, ECSO defenses)
    that have no YAML defaults — avoids polluting the wrapper with unrelated
    subsystem defaults.

    Classical-language configs live one level deeper
    (conf/text_encoding/classical_language/<lang>.yaml); they merge as
    default.yaml → <lang>.yaml → task_params, so encoder defaults like
    `model:` apply without every task passing them explicitly.
    """
    from src.experiment.config import CONF_DIR, _deep_merge, _load_yaml

    canonical, _ = resolve_transformation_name(type_name)
    is_image = canonical.startswith("ir_")

    candidates: list[str] = [type_name]
    if canonical not in candidates:
        candidates.append(canonical)
    if is_image and canonical.startswith("ir_"):
        unprefixed = canonical[len("ir_"):]
        if unprefixed not in candidates:
            candidates.append(unprefixed)

    subsystem = "imaging" if is_image else "text_encoding"

    for name in candidates:
        if (CONF_DIR / subsystem / f"{name}.yaml").exists():
            return _load_conf(
                subsystem, override_name=name,
                task_overrides=task_params or None,
            )

    if canonical == "llm_classical_language":
        lang = type_name[:-len("_literary")] if type_name.endswith("_literary") else type_name
        deep_path = CONF_DIR / subsystem / "classical_language" / f"{lang}.yaml"
        if deep_path.exists():
            merged = _load_conf(subsystem)          # text_encoding/default.yaml
            merged = _deep_merge(merged, _load_yaml(deep_path))
            if task_params:
                merged = _deep_merge(merged, task_params)
            return merged

    return dict(task_params or {})


def _run_prompt_transform(task) -> dict[str, Any]:
    """Stage 1: run a chain of PromptTransformations.

    Each step in task.transformation_list runs in order:
      - resolves config (3-layer YAML merge + task overrides)
      - instantiates the transformation via the factory
      - validates modality consistency vs. cumulative chain state
      - creates a subfolder named after type_name
      - applies the transformation (writes its own artifacts into the subfolder)
      - persists prompts.jsonl + cumulative results.json into that subfolder

    Input prompts come from either:
      - `task.source_file` — a raw dataset JSONL (chain starts fresh), OR
      - `task.source_transform_subdir` — a prior step subfolder (chain
        continues from that step; lets multiple downstream chains share an
        identical encoder output, avoiding LLM-encoder nondeterminism in
        paired ablations).
    """
    t_chain_start = time.time()

    upstream: dict = {}
    upstream_chain_names: list[str] = []
    if task.source_transform_subdir:
        source_dir = Path(task.source_transform_subdir)
        if not source_dir.exists():
            raise FileNotFoundError(
                f"source_transform_subdir does not exist: {source_dir}")
        upstream = _load_results(str(source_dir))
        if not upstream:
            raise ValueError(
                f"source_transform_subdir missing results.json: {source_dir}")
        is_multimodal = bool(upstream.get("is_multimodal", False))
        benchmark = (
            task.benchmark
            or upstream.get("benchmark")
            or _infer_benchmark(str(source_dir)))
        upstream_chain_names = list(upstream.get("transformation_list", []))
        dataset_path = upstream.get("dataset_path", "")
        prompts = _load_prompts(str(source_dir))
        prompts = _apply_prompt_range(prompts, task.prompt_range)
        logger.info(
            f"prompt_transform chaining from {source_dir} "
            f"(n_prompts={len(prompts)}, is_multimodal={is_multimodal}, "
            f"upstream_chain={'+'.join(upstream_chain_names)})")
    else:
        benchmark = task.benchmark or _infer_benchmark(task.source_file)
        dataset_path = task.source_file
        raw = _load_raw_prompts(task.source_file)
        raw = _apply_prompt_range(raw, task.prompt_range)
        logger.info(f"Loaded {len(raw)} raw prompts from {task.source_file}")
        prompts: list[Prompt] = [
            Prompt(id=r.id, encoding="", original=r.prompt, encoded=r.prompt)
            for r in raw
        ]
        is_multimodal = False

    new_chain_label = "_".join(
        s.type if isinstance(s, TransformationSpec) else s["type"]
        for s in task.transformation_list
    )
    full_chain_label = (
        "_".join(upstream_chain_names) + "_" + new_chain_label
        if upstream_chain_names else new_chain_label
    )[:80]
    task_dir = Path(get_new_experiment_data_dir(
        "outputs/prompt_transform", dataset=benchmark, model=full_chain_label,
    ))
    logger.info(f"prompt_transform task_dir: {task_dir}")

    transformation_list_names: list[str] = []
    results_history: list[dict[str, PromptTransformStepResult]] = []

    # Pre-construct all transformations so config errors (missing files, bad
    # params) raise BEFORE any expensive earlier step runs. The discarded
    # instances are cheap — only renderer __init__ I/O.
    for _pre_spec in task.transformation_list:
        _pre_type = (
            _pre_spec.type if isinstance(_pre_spec, TransformationSpec)
            else _pre_spec["type"])
        _pre_params = (
            _pre_spec.params if isinstance(_pre_spec, TransformationSpec)
            else _pre_spec.get("params", {}))
        create_transformation(_pre_type, **_resolve_step_config(_pre_type, _pre_params))

    for step_spec in task.transformation_list:
        step_type = (
            step_spec.type if isinstance(step_spec, TransformationSpec)
            else step_spec["type"])
        step_params = (
            step_spec.params if isinstance(step_spec, TransformationSpec)
            else step_spec.get("params", {}))

        merged_config = _resolve_step_config(step_type, step_params)
        transformation = create_transformation(step_type, **merged_config)
        type_name = transformation.type_name

        # Modality validation: can't run a text-only transformation after a
        # multimodal one (no way to "untransform" image back to text).
        if (transformation.input_modality == Modality.TEXT
                and is_multimodal):
            raise ValueError(
                f"Chain ordering invalid: step {type_name!r} expects text "
                f"input, but chain is already multimodal after previous "
                f"step(s). Reorder so image transformations come last.")

        step_dir = task_dir / type_name
        step_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Applying step: {type_name} ({len(prompts)} prompts)")
        t_step_start = time.time()
        prompts = transformation.apply(prompts, step_dir)
        elapsed_step = round(time.time() - t_step_start, 2)

        write_jsonl_atomic(
            step_dir / "prompts.jsonl",
            [p.model_dump_json() for p in prompts])

        if transformation.output_modality == Modality.MULTIMODAL:
            is_multimodal = True
        transformation_list_names.append(type_name)

        step_metrics = transformation.step_metrics()
        step_record = PromptTransformStepResult(
            config=transformation.get_config(),
            metrics=step_metrics,
            prompts_file="prompts.jsonl",
            images_dir=step_metrics.get("images_dir"),
            is_multimodal_after=is_multimodal,
            elapsed_seconds=elapsed_step,
            usage=transformation.get_usage(),
        )
        results_history.append({type_name: step_record})

        result = PromptTransformResult(
            task_id=task_dir.name,
            benchmark=benchmark,
            prompt_range=task.prompt_range,
            dataset_path=dataset_path,
            is_multimodal=is_multimodal,
            total_steps=len(upstream_chain_names) + len(results_history),
            latest_step=type_name,
            latest_step_dir=type_name,
            task_dir=str(task_dir),
            timestamp_last_step=_now_iso(),
            elapsed_seconds_total=round(time.time() - t_chain_start, 2),
            count=len(prompts),
            transformation_list=upstream_chain_names + transformation_list_names,
            results_history=results_history,
            upstream_ref=(
                _upstream_ref(task.source_transform_subdir)
                if task.source_transform_subdir else None),
            **provenance_fields(),
        )
        write_json_atomic(
            step_dir / "results.json",
            result.model_dump(exclude_none=True))
        logger.info(
            f"Step {type_name} done in {elapsed_step}s "
            f"(is_multimodal={is_multimodal})")

    latest_dir = task_dir / transformation_list_names[-1]
    return {
        "status": "success",
        "output_dir": str(latest_dir),
        "task_dir": str(task_dir),
        "transformation_list": transformation_list_names,
        "count": len(prompts),
        "is_multimodal": is_multimodal,
        "elapsed_seconds": round(time.time() - t_chain_start, 2),
    }


# ======================== defense+evaluate ========================


def _run_defense_evaluate(task) -> dict[str, Any]:
    """Stage 2: defense + target-model query + judging.

    Consumes a specific prompt_transform step subfolder pointed at by
    task.source_transform_subdir. Reads prompts.jsonl + the upstream
    PromptTransformResult to know is_multimodal, encoding, benchmark.
    """
    t0 = time.time()

    source_dir = Path(task.source_transform_subdir)
    if not source_dir.exists():
        raise FileNotFoundError(
            f"source_transform_subdir does not exist: {source_dir}")
    upstream = _load_results(str(source_dir))
    if not upstream:
        raise ValueError(
            f"source_transform_subdir is missing results.json: {source_dir}")

    is_multimodal = bool(upstream.get("is_multimodal", False))
    transformation_list = list(upstream.get("transformation_list") or [])
    encoding = upstream.get("latest_step") or (transformation_list[-1] if transformation_list else None)
    benchmark = task.benchmark or upstream.get("benchmark") or _infer_benchmark(str(source_dir))

    prompts = _load_prompts(str(source_dir))
    prompts = _apply_prompt_range(prompts, task.prompt_range)
    logger.info(
        f"defense+evaluate: defense={task.defense}, model={task.target_model}, "
        f"source={source_dir}, n_prompts={len(prompts)}, is_multimodal={is_multimodal}")

    # Output dir: outputs/defense+evaluate/<benchmark>/<model>_<defense>_<chain_label>_<ts>_<rand>/
    chain_label = "_".join(upstream.get("transformation_list", []))[:60]
    out_dir = Path(get_new_experiment_data_dir(
        "outputs/defense+evaluate", dataset=benchmark,
        model=f"{task.target_model}_{task.defense}_{chain_label}",
    ))

    # Create defense + target service.
    # 3-layer defense config merge:
    #   conf/defense/default.yaml → conf/defense/<defense>.yaml → task.defense_config
    # This makes conf/defense/<defense>.yaml the source of truth for each
    # defense's hyperparameters (e.g. SS's perturbation_model + temperature),
    # consistent with how text_encoding/imaging configs are loaded.
    try:
        merged_defense_config = _load_conf(
            "defense", override_name=task.defense,
            task_overrides=task.defense_config or None,
        )
    except FileNotFoundError:
        # No per-defense yaml exists → fall back to task.defense_config only.
        merged_defense_config = dict(task.defense_config or {})
    defense = create_defense(task.defense, **merged_defense_config)
    # Per-task target sampling overrides win over the model/global yaml defaults, and
    # the RECORDED config is the merged (effective) one — results.json must never claim
    # a temperature the run did not use.
    target_overrides = dict(task.target_model_config or {})
    target_service = LLMServiceFactory.create(task.target_model, **target_overrides)
    target_llm = LLMModel.from_string(task.target_model)
    target_model_config = {
        **LLMServiceFactory._load_model_defaults(target_llm),
        **target_overrides,
    }

    # Multi-image overflow guard: a paginated prompt can render to many images
    # and exceed the served context budget. Rather than drop pages (which
    # mutilates the attack and fakes a refusal), EXCLUDE over-budget prompts
    # from the query AND from ASR, and flag them (is_within_maxlen=False) so the
    # coverage loss is recorded honestly instead of silently degraded.
    def _num_images(p) -> int:
        return len(p.image_encoded) if p.image_encoded else 0

    max_pages = _max_pages_per_prompt()
    if is_multimodal:
        within = [p for p in prompts if _num_images(p) <= max_pages]
        over = [p for p in prompts if _num_images(p) > max_pages]
    else:
        within, over = list(prompts), []
    if over:
        logger.warning(
            f"{len(over)}/{len(prompts)} prompts exceed "
            f"max_pages_per_prompt={max_pages} — excluded from "
            f"query+ASR (is_within_maxlen=False): {[p.id for p in over][:10]}")

    # Run defense on within-budget prompts only.
    pairs = defense.query(
        prompts=within,
        target_service=target_service,
        is_multimodal=is_multimodal,
        source_dir=source_dir,
        system_message=task.system_message,
    )

    # Build EvaluationRow list with synthetic prompt_stage. num_images is
    # recorded for every prompt; over-budget rows carry an empty response and
    # is_within_maxlen=False, and are NOT judged. Rows whose target response is a
    # mechanism-error sentinel (the call failed to produce a valid output:
    # query-time context-overflow, network, timeout, rate-limit, failed batch
    # item) are flagged is_correctly_processed=False and likewise excluded from
    # judging/ASR — an untestable prompt must not count as a defeated attack. A
    # refusal is a *successful* call, so it stays is_correctly_processed=True.
    stage_label = f"{encoding}__{task.defense}" if encoding else task.defense
    npages_by_id = {p.id: _num_images(p) for p in prompts}
    all_rows: list[EvaluationRow] = []
    n_mechanism_err = 0
    for pid, resp in pairs:
        failed = is_mechanism_error(resp)
        if failed:
            n_mechanism_err += 1
        all_rows.append(EvaluationRow(
            id=pid, prompt_stage=stage_label,
            response=strip_mechanism_error(resp), asr=None,
            num_images=npages_by_id.get(pid, 0),
            is_within_maxlen=True, is_correctly_processed=not failed))
    if n_mechanism_err:
        logger.warning(
            f"{n_mechanism_err}/{len(pairs)} prompts hit a mechanism error "
            f"(is_correctly_processed=False) — excluded from query+ASR.")
    all_rows += [
        EvaluationRow(id=p.id, prompt_stage=stage_label, response="", asr=None,
                      num_images=_num_images(p), is_within_maxlen=False)
        for p in over
    ]

    # Write raw_results.jsonl (raw model responses preserved so judging can
    # be re-run later with a different judge config if needed).
    write_jsonl_atomic(
        out_dir / "raw_results.jsonl",
        [row.model_dump_json() for row in all_rows])

    # Judge only rows that were both within budget AND correctly processed
    # (over-budget and mechanism-error rows are excluded from ASR). Filtering
    # keeps object references, so verdict columns still land back in all_rows.
    within_rows = [
        r for r in all_rows if r.is_within_maxlen and r.is_correctly_processed]
    judged = _run_judging(
        prompts=within, all_rows=within_rows, benchmark=benchmark,
        judge_method_override=task.judge_method, stage_label=stage_label,
        judge_model_override=task.judge_model,
    )
    asr, refusal = judged["asr"], judged["refusal"]
    eval_stats, jhash, judge_errors = (
        judged["eval_stats"], judged["judge_config_hash"], judged["judge_errors"])
    judge_errors = list(judge_errors) + _dropped_row_warnings(
        all_rows, within_rows, stage_label)
    # Best-of-N validity: are the draws actually different? Reported, never fatal
    # (uniform refusals are a legitimate low-diversity case). See draw_diversity_stats.
    diversity_stats = draw_diversity_stats(all_rows, stage_label)
    judge_method_provenance = task.judge_method or benchmark
    # Record the judge that ACTUALLY scored. A self-contained classifier judge
    # (e.g. wildguard) IS the judge — recording the vestigial config-default LLM
    # there falsely claims e.g. gpt-5-nano ran. An LLM-rubric judge uses the judge
    # LLM, so record the per-task override or the evaluation default.
    from src.evaluation.evaluator_factory import SELF_CONTAINED_JUDGE_METHODS
    if task.judge_method in SELF_CONTAINED_JUDGE_METHODS:
        judge_model_used = task.judge_method
    else:
        judge_model_used = task.judge_model or _load_conf("evaluation").get(
            "judge_llm_config", {}).get("model")

    # Overwrite raw_results.jsonl now that verdict columns are filled.
    write_jsonl_atomic(
        out_dir / "raw_results.jsonl",
        [row.model_dump_json() for row in all_rows])

    elapsed = round(time.time() - t0, 2)

    result = DefenseEvaluateResult(
        source_transform_subdir=str(source_dir),
        campaign=task.campaign,
        upstream_ref=_upstream_ref(source_dir),
        is_multimodal=is_multimodal,
        defense=task.defense,
        defense_config=task.defense_config or {},
        target_model=task.target_model,
        target_model_config=TargetModelConfig(**target_model_config),
        model_family=target_llm.family,
        alignment_tier=target_llm.alignment_tier,
        system_message=task.system_message,
        benchmark=benchmark,
        encoding=encoding,
        transformation_list=transformation_list,
        companion_image_path=_extract_companion_image_path(upstream),
        prompt_range=task.prompt_range,
        judge_method=judge_method_provenance,
        judge_model=judge_model_used,
        count=len(all_rows),
        asr=asr,
        refusal_rate=refusal,
        metrics={**judged["metrics"], **diversity_stats},
        primary_metric=judged["primary_metric"],
        eval_stats=eval_stats,
        target_usage=target_service.get_usage(),
        defense_usage=defense.get_usage(),
        elapsed_seconds=elapsed,
        output_dir=str(out_dir),
        **provenance_fields(),
        judge_config_hash=jhash,
        status="success" if not judge_errors else "partial_judge",
        warnings=judge_errors,
    )
    _save_results(out_dir, result.model_dump(exclude_none=True))

    return {
        "status": "success",
        "output_dir": str(out_dir),
        "count": len(all_rows),
        "defense": task.defense,
        "target_model": task.target_model,
        "encoding": encoding,
        "judge_method": judge_method_provenance,
        "asr": asr,
        "refusal_rate": refusal,
        "usage": result.target_usage,
        "elapsed_seconds": elapsed,
    }


# ======================== adaptive_attack ========================

CHANNEL_RENDERERS = {
    "image_figstep": "ir_figstep",
    "image_flowchart": "ir_fc_flowchart",
    "image_low_contrast": "ir_low_contrast",
    "image_occluded": "ir_occluded",
}


def _run_adaptive_attack(task) -> dict[str, Any]:
    """For each behavior, runs a query budget: query the defense, read the
    refusal/block outcome, have an attacker LLM rewrite, repeat. The FINAL kept
    response per behavior is judged by the project judge, so ASR is comparable to
    a defense+evaluate cell. Writes a DefenseEvaluateResult so the existing
    ensemble/portfolio analysis reads it unchanged.

    With `attack_config.channels` listing more than one channel, the attacker
    also picks HOW each candidate is delivered (plain text or one of the
    CHANNEL_RENDERERS renders), so the search varies both channels at the same
    query budget.
    """
    from src.attacks.adaptive_pipeline_attack import (
        ATTACKER_SYSTEM, run_adaptive_attack)
    from src.defense.guard_utils import GUARD_REFUSAL_TEXT

    t0 = time.time()
    benchmark = task.benchmark or _infer_benchmark(task.source_file)
    raw = _apply_prompt_range(_load_raw_prompts(task.source_file), task.prompt_range)
    behaviors = [(r.id, r.prompt) for r in raw]
    logger.info(
        f"adaptive_attack: defense={task.defense}, target={task.target_model}, "
        f"attacker={task.attacker_model}, n={len(behaviors)}")

    # merge loop knobs: conf/adaptive_attack/default.yaml <- task.attack_config
    ac = _load_conf("adaptive_attack", task_overrides=task.attack_config or None)

    # build defense + services (mirror defense+evaluate)
    try:
        merged_defense_config = _load_conf(
            "defense", override_name=task.defense,
            task_overrides=task.defense_config or None)
    except FileNotFoundError:
        merged_defense_config = dict(task.defense_config or {})
    defense = create_defense(task.defense, **merged_defense_config)
    target_service = LLMServiceFactory.create(task.target_model)
    target_llm = LLMModel.from_string(task.target_model)
    attacker_service = LLMServiceFactory.create(task.attacker_model)
    # attacker MUST sample (temp 0 regenerates a failed candidate verbatim)
    attacker_service.temperature = float(ac.get("attacker_temperature", 0.9))
    attacker_service.max_tokens = int(ac.get("attacker_max_tokens", 800))

    out_dir = Path(get_new_experiment_data_dir(
        "outputs/defense+evaluate", dataset=benchmark,
        model=f"{task.target_model}_{task.defense}_adaptive"))

    channels = [str(c) for c in (ac.get("channels") or ["text"])]
    renderers: dict[str, Any] = {}

    def _renderer(channel: str):
        if channel not in renderers:
            type_name = CHANNEL_RENDERERS[channel]
            # keep_text: false matches how the published suite was rendered
            # (conf/experiment/.../render_n100.yaml) — the text channel becomes
            # the stock "check the image" instruction, so an image channel
            # actually hides the payload instead of duplicating it in text.
            renderers[channel] = create_transformation(
                type_name,
                **_resolve_step_config(type_name, {"keep_text": False}))
        return renderers[channel]

    unknown = [c for c in channels if c != "text" and c not in CHANNEL_RENDERERS]
    if unknown:
        raise ValueError(
            f"unknown adaptive-attack channel(s) {unknown}; expected 'text' or "
            f"one of {sorted(CHANNEL_RENDERERS)}")

    # pipeline_fn: candidates (text or rendered) -> defended responses.
    # One real defended query per candidate, whatever the channel — the budget
    # is per behavior per round, so text and image cost the attacker the same.
    round_idx = {"r": 0}

    def pipeline_fn(items: list[tuple[str, str, str]]) -> dict[str, str]:
        by_channel: dict[str, list[tuple[str, str]]] = {}
        for pid, text, ch in items:
            by_channel.setdefault(ch, []).append((pid, text))
        out: dict[str, str] = {}
        for ch, group in by_channel.items():
            ps = [Prompt(id=pid, encoding="", original=text, encoded=text)
                  for pid, text in group]
            if ch == "text":
                pairs = defense.query(
                    prompts=ps, target_service=target_service,
                    is_multimodal=False, source_dir=None, system_message=None)
            else:
                step_dir = out_dir / "adaptive_renders" / f"round{round_idx['r']}_{ch}"
                step_dir.mkdir(parents=True, exist_ok=True)
                rendered = _renderer(ch).apply(ps, step_dir)
                logger.info(
                    f"adaptive_attack round {round_idx['r']}: rendered "
                    f"{len(rendered)} candidates over channel {ch}")
                pairs = defense.query(
                    prompts=rendered, target_service=target_service,
                    is_multimodal=True, source_dir=step_dir, system_message=None)
            out.update({pid: strip_mechanism_error(resp) for pid, resp in pairs})
        round_idx["r"] += 1
        return out

    # attacker_fn: rewrite requests -> rewrites (one batched LLM call)
    def attacker_fn(reqs: list[tuple[str, str]]) -> dict[str, str]:
        convs = [(pid, [(text, None)]) for pid, text in reqs]
        out = attacker_service.batch_chat(
            conversations=convs, system_message=ATTACKER_SYSTEM, is_test=False)
        return {cid: (txt or "").strip() for cid, txt in out}

    round_counts: list[tuple[int, int, int]] = []
    traces = run_adaptive_attack(
        behaviors, pipeline_fn, attacker_fn,
        rounds=int(ac.get("rounds", 8)),
        refusal_markers=[m.lower() for m in ac.get("refusal_markers", [])],
        min_answer_chars=int(ac.get("min_answer_chars", 40)),
        block_text=GUARD_REFUSAL_TEXT,
        early_stop_fraction=float(ac.get("early_stop_fraction", 1.0)),
        round_log=lambda r, a, s: round_counts.append((r, a, s)),
        channels=channels)

    # judge the FINAL kept response per behavior, like any defense+evaluate cell
    stage_label = f"adaptive__{task.defense}"
    all_rows = [
        EvaluationRow(id=tr.prompt_id, prompt_stage=stage_label,
                      response=strip_mechanism_error(tr.final_response), asr=None,
                      num_images=0 if tr.final_channel == "text" else 1,
                      is_within_maxlen=True,
                      is_correctly_processed=not is_mechanism_error(tr.final_response))
        for tr in traces]
    write_jsonl_atomic(out_dir / "raw_results.jsonl",
                       [r.model_dump_json() for r in all_rows])
    # per-behavior attack traces (query budget used, candidate/response/channel history)
    write_jsonl_atomic(out_dir / "attack_traces.jsonl", [
        json.dumps({"id": tr.prompt_id, "rounds_used": tr.rounds_used,
                    "succeeded_heuristic": tr.succeeded_heuristic,
                    "final_text": tr.final_text,
                    "final_channel": tr.final_channel,
                    "blocked_flags": tr.blocked_flags,
                    "channels": tr.channels,
                    "candidates": tr.candidates}) for tr in traces])

    judge_prompts = [Prompt(id=tr.prompt_id, encoding="", original=tr.behavior,
                            encoded=tr.final_text) for tr in traces]
    within = [r for r in all_rows if r.is_correctly_processed]
    judged = _run_judging(
        prompts=judge_prompts, all_rows=within, benchmark=benchmark,
        judge_method_override=task.judge_method, stage_label=stage_label,
        judge_model_override=task.judge_model)

    # budget summary: cumulative heuristic-success by round (the sweep's data)
    n = len(traces)
    succ_by_round = {}
    for tr in traces:
        if tr.succeeded_heuristic:
            succ_by_round[tr.rounds_used] = succ_by_round.get(tr.rounds_used, 0) + 1
    cum, budget_curve = 0, []
    for r in range(1, int(ac.get("rounds", 8)) + 2):
        cum += succ_by_round.get(r, 0)
        budget_curve.append({"rounds": r, "cum_heuristic_success": cum,
                             "cum_frac": round(cum / n, 3) if n else 0.0})

    from src.evaluation.evaluator_factory import SELF_CONTAINED_JUDGE_METHODS
    if task.judge_method in SELF_CONTAINED_JUDGE_METHODS:
        judge_model_used = task.judge_method
    else:
        judge_model_used = task.judge_model or _load_conf("evaluation").get(
            "judge_llm_config", {}).get("model")
    write_jsonl_atomic(out_dir / "raw_results.jsonl",
                       [r.model_dump_json() for r in all_rows])

    elapsed = round(time.time() - t0, 2)
    result = DefenseEvaluateResult(
        source_transform_subdir=task.source_file,
        campaign=task.campaign,
        upstream_ref=None,
        is_multimodal=any(tr.final_channel != "text" for tr in traces),
        defense=f"adaptive_{task.defense}",
        defense_config={**(task.defense_config or {}),
                        "attacker_model": task.attacker_model,
                        "attack_config": {k: ac.get(k) for k in
                                          ("rounds", "attacker_temperature",
                                           "early_stop_fraction")},
                        "channels": channels,
                        "channel_final_counts": {
                            ch: sum(1 for tr in traces if tr.final_channel == ch)
                            for ch in channels},
                        "budget_curve": budget_curve},
        target_model=task.target_model,
        target_model_config=TargetModelConfig(
            **LLMServiceFactory._load_model_defaults(target_llm)),
        model_family=target_llm.family,
        alignment_tier=target_llm.alignment_tier,
        system_message=None,
        benchmark=benchmark,
        encoding="adaptive",
        transformation_list=["adaptive"],
        companion_image_path=None,
        prompt_range=task.prompt_range,
        judge_method=task.judge_method or benchmark,
        judge_model=judge_model_used,
        count=len(all_rows),
        asr=judged["asr"],
        refusal_rate=judged["refusal"],
        metrics=judged["metrics"],
        primary_metric=judged["primary_metric"],
        eval_stats=judged["eval_stats"],
        target_usage=target_service.get_usage(),
        defense_usage=defense.get_usage(),
        elapsed_seconds=elapsed,
        output_dir=str(out_dir),
        **provenance_fields(),
        judge_config_hash=judged["judge_config_hash"],
        status="success" if not judged["judge_errors"] else "partial_judge",
        warnings=list(judged["judge_errors"]),
    )
    _save_results(out_dir, result.model_dump(exclude_none=True))
    logger.info("adaptive_attack done: asr=%s, budget_curve=%s",
                judged["asr"], budget_curve[-1] if budget_curve else None)
    return {
        "status": "success", "output_dir": str(out_dir), "count": len(all_rows),
        "defense": f"adaptive_{task.defense}", "target_model": task.target_model,
        "encoding": "adaptive", "asr": judged["asr"],
        "refusal_rate": judged["refusal"], "elapsed_seconds": elapsed,
    }


# ======================== analyze ========================


def _run_analyze(task) -> dict[str, Any]:
    """`analyze` mode: pure post-processing over already-run results — no
    target/judge calls. Delegates to src/analysis (dispatched by task.analysis).
    """
    from src.analysis import run_analysis
    return run_analysis(task)


# ======================== Dispatcher ========================


TASK_MODES = {
    "prompt_transform": _run_prompt_transform,
    "defense+evaluate": _run_defense_evaluate,
    "analyze": _run_analyze,
    "rejudge": _run_rejudge,
    "adaptive_attack": _run_adaptive_attack,
}


def run_task(task) -> dict[str, Any]:
    """Run a task — dispatched by `task.mode`. Wires MLflow tracking."""
    from src.utils.mlflow_tracker import MLflowTracker  # lazy import
    from .schemas import (
        DefenseEvaluateTask, PromptTransformTask, TaskConfig,
    )

    # Accept legacy dict input.
    if isinstance(task, dict):
        from pydantic import TypeAdapter
        task = TypeAdapter(TaskConfig).validate_python(task)

    mode = task.mode
    if mode not in TASK_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Valid: {list(TASK_MODES)}")

    # MLflow tags.
    model_tag = ""
    encoding_tag = ""
    modality_tag = ""
    if isinstance(task, DefenseEvaluateTask):
        model_tag = task.target_model
        encoding_tag = task.defense
    elif isinstance(task, PromptTransformTask):
        from .schemas import TransformationSpec
        chain_types = [
            s.type if isinstance(s, TransformationSpec) else s["type"]
            for s in task.transformation_list
        ]
        encoding_tag = ",".join(chain_types)
        modality_tag = chain_types[-1] if chain_types else ""

    tracker = MLflowTracker()
    tracker.start_run(
        mode=mode, encoding=encoding_tag, model=model_tag, modality=modality_tag,
    )
    tracker.log_params(task.model_dump(exclude_none=True))

    try:
        result = TASK_MODES[mode](task)
        tracker.log_metrics(result)

        out_dir = result.get("output_dir")
        if out_dir:
            out_path = Path(out_dir)
            # Stamp mlflow_run_id into results.json for cross-ref.
            results_file = out_path / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                data["mlflow_run_id"] = tracker.run_id
                write_json_atomic(results_file, data)

            # MLflow is an index, not a second copy of the data: point at the
            # canonical output dir on disk instead of mirroring artifacts.
            tracker.log_output_dir(out_dir)

        return result
    except Exception:
        tracker.log_metrics({"status_failed": 1})
        raise
    finally:
        tracker.end_run()
