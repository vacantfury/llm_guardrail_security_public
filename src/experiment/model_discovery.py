"""
Model discovery for the experiment orchestrator.

Pure functions that answer "which models does this task reference?" — the
target model, the judge model(s) the evaluator will run, defense-owned second
models (guard / perturbation), and prompt-transform helper models — plus the
provider-based task classification built on top of them.

These are used by the orchestrator (`experiment.py`) to pre-discover
cluster-hosted models so vLLM servers can be started before tasks run, and by
the multi-cluster router (`multi_cluster.py`) to compute each task's split key.

Depends only on schemas / llm_utils / config — never on `experiment.py`, so
there is no import cycle. Every name here is re-exported from `experiment.py`
for backwards compatibility.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional


# ==================== Task Classification ====================

class TaskType(str, Enum):
    """Task type based on model requirements."""
    API_MODEL = "api_model"          # uses cloud API (GPT-4o, Gemini, Claude)
    CLUSTER_MODEL = "cluster_model"  # uses vLLM server on cluster GPU
    NO_MODEL = "no_model"            # no model query (imaging)


def _target_model_for_task(task) -> Optional[str]:
    """Return the model-string this task targets (None if the mode doesn't have one).

    Only defense+evaluate tasks have a top-level target model. prompt_transform
    chains may invoke an encoder LLM (via merged conf/text_encoding YAML) but
    those don't drive orchestrator-level cluster-server discovery — the
    factory handles them inline.
    """
    from .schemas import AdaptiveAttackTask, DefenseEvaluateTask
    if isinstance(task, (DefenseEvaluateTask, AdaptiveAttackTask)):
        return task.target_model
    return None


def _infer_task_type(task) -> TaskType:
    """Infer task type from a typed TaskConfig variant."""
    from .schemas import PromptTransformTask, RejudgeTask
    if isinstance(task, PromptTransformTask):
        # prompt_transform may make LLM calls inside encoders, but those go
        # through API-based encoder LLMs (gpt-4.1-mini etc.) — not cluster.
        return TaskType.NO_MODEL

    if isinstance(task, RejudgeTask):
        # No target; the judge may be cluster-served (WildGuard) or API (gpt-5-mini).
        from llm_utils import LLMModel, Provider
        try:
            m = LLMModel.from_string(task.judge_model)
            return (TaskType.CLUSTER_MODEL if m.provider == Provider.SLURM_CLUSTER
                    else TaskType.API_MODEL)
        except ValueError:
            return TaskType.NO_MODEL

    model_str = _target_model_for_task(task)
    if model_str:
        from llm_utils import LLMModel, Provider
        try:
            m = LLMModel.from_string(model_str)
            if m.provider == Provider.SLURM_CLUSTER:
                return TaskType.CLUSTER_MODEL
        except ValueError:
            pass
        return TaskType.API_MODEL

    return TaskType.NO_MODEL


def _resolve_model(model_str: str) -> Optional[Any]:
    """Resolve model string to LLMModel enum, or None."""
    from llm_utils import LLMModel
    try:
        return LLMModel.from_string(model_str)
    except ValueError:
        return None


# Pre-canonical-refactor design: ALL evaluators share one judge LLM config,
# read from conf/evaluation/default.yaml::judge_llm_config.model. So
# "judge_method" → judge model is constant across methods at any point in
# time — they all use whatever's in the YAML.
_KNOWN_JUDGE_METHODS = {"harmbench", "jailbreakbench", "refusal", "orbench"}


def _judge_model_for_method(judge_method: str) -> Optional[Any]:
    """Resolve a judge_method slug to the LLMModel currently configured to
    judge it (from conf/evaluation/default.yaml::judge_llm_config.model).

    Returns None for unknown methods or if the YAML can't be loaded. Used
    by the orchestrator to pre-discover cluster-hosted judges so vLLM
    servers can be started before tasks run. When the configured judge is
    an API model (e.g. gpt-5-nano), discovery correctly returns a
    Provider.OPENAI model → orchestrator skips cluster setup for judges.
    """
    if judge_method not in _KNOWN_JUDGE_METHODS:
        return None
    try:
        from .config import load_conf
        from llm_utils import LLMModel
        eval_config = load_conf("evaluation")
        judge_cfg = eval_config.get("judge_llm_config", {})
        model_str = judge_cfg.get("model")
        if not model_str:
            return None
        return LLMModel.from_string(model_str)
    except Exception:
        return None


def _collect_model_strings(obj) -> list:
    """Recursively collect string values stored under a "model" key.

    Used to discover cluster-hosted HELPER models referenced *inside* a
    transformation's params (an LLM encoder's `model`, or the variance-channel
    wrapper's paraphrase / attack-bank helper — possibly nested), so the
    orchestrator serves them. Walks dicts/lists; returns every non-empty string
    found under a `model` key.
    """
    found: list = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "model" and isinstance(v, str) and v:
                found.append(v)
            else:
                found.extend(_collect_model_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_collect_model_strings(item))
    return found


def _required_cluster_models_for_task(info: "TaskInfo") -> set:
    """Cluster-hosted (vLLM / Provider.SLURM_CLUSTER) models a task needs.

    Thin wrapper over `_referenced_models_for_task` filtered to the cluster
    provider — the historical name/behavior the scheduler relies on. To ask the
    same question for another provider (e.g. Bedrock, for multi-cluster routing),
    call `_referenced_models_for_task(info, {Provider.BEDROCK})`.
    """
    from llm_utils import Provider
    return _referenced_models_for_task(info, {Provider.SLURM_CLUSTER})


def _referenced_models_for_task(info: "TaskInfo", providers=None) -> set:
    """Return the set of LLMModels this task references, optionally filtered.

    `providers=None` returns every resolved model (any provider); passing a set
    of `Provider`s keeps only those. Covers the target model, the judge model(s)
    for the evaluator(s) that will run, AND any SECOND model a defense itself
    references (Round-3 guard amplifier's `guard_model`, or SemanticSmooth's
    `perturbation_model`), plus prompt-transform helper models. Judge discovery
    has two paths:

      1. Explicit override: task.judge_method set in YAML.
      2. Canonical inference: derive judge_method list from the benchmark
         slug (via EvaluatorFactory.judge_methods_for_benchmark). The
         benchmark itself comes from task.benchmark or the upstream
         source_transform_subdir path.

    Without path (2), removing `judge_method` from task YAMLs would silently
    bypass cluster judge discovery — tasks would hang on acquire_endpoint().

    Defense-config scan is DELIBERATELY narrow: only the known second-model
    keys (`guard_model`, `perturbation_model`, modality_complete's
    `amplifier_model` / `recover_model` / `gate_model` / `decode_model`, and
    the P6 self-check defenses' `shadow_model` / `filter_model`) are looked
    up — defense_config is a free-form dict and we don't want to guess at
    arbitrary keys. EXTEND THIS TUPLE whenever a new defense adds another
    second-model config key; forgetting to is a silent-until-runtime failure
    (the task dies with "No vLLM server was ever started for <model>", as the
    P6 pilot did for selfdefend's shadow_model on 2026-07-30).

    Budget note: a defense+evaluate task with a cluster target AND a cluster
    guard AND a cluster judge needs orchestrator + 3 vLLM servers = 4 SLURM
    jobs (well under MAX_SUBMIT_JOBS_PER_USER=8, but each cluster-hosted
    second model directly consumes num_cluster_jobs — account for it when
    sizing a round with multiple concurrent cluster-guard tasks).
    """
    from .schemas import (
        AdaptiveAttackTask, DefenseEvaluateTask, PromptTransformTask, RejudgeTask,
        TransformationSpec,
    )
    from .task import _infer_benchmark
    from src.evaluation.evaluator_factory import judge_methods_for_benchmark

    task = info.task
    required: set = set()

    def _keep(m) -> bool:
        return m is not None and (providers is None or m.provider in providers)

    # rejudge: no target/defense to serve — only the judge, which may itself be
    # cluster-served (e.g. WildGuard). Unlike defense+evaluate's benchmark-based
    # judge discovery (which reads the YAML default), the rejudge judge is the
    # task's OWN judge_model, so resolve it directly.
    if isinstance(task, RejudgeTask):
        judge = _resolve_model(task.judge_model)
        if _keep(judge):
            required.add(judge)
        return required

    # adaptive_attack: needs the TARGET, the ATTACKER LLM, the defense's guard,
    # and the judge — all potentially cluster-served. Missing any one hangs the
    # run on acquire_endpoint() (the silent-until-runtime failure this module
    # exists to prevent).
    if isinstance(task, AdaptiveAttackTask):
        for name in (task.target_model, task.attacker_model, task.judge_model):
            m = _resolve_model(name) if name else None
            if _keep(m):
                required.add(m)
        try:
            from .config import load_conf as _load_conf
            dcfg = _load_conf("defense", override_name=task.defense,
                              task_overrides=task.defense_config or None)
        except Exception:
            dcfg = dict(task.defense_config or {})
        for key in ("guard_model", "amplifier_model", "recover_model",
                    "gate_model", "decode_model"):
            m = _resolve_model(dcfg[key]) if dcfg.get(key) else None
            if _keep(m):
                required.add(m)
        if not task.judge_model:
            try:
                methods = judge_methods_for_benchmark(
                    task.benchmark or _infer_benchmark(task.source_file))
            except ValueError:
                methods = []
            for method in methods:
                jm = _judge_model_for_method(method)
                if _keep(jm):
                    required.add(jm)
        return required

    # Target model (only meaningful for defense+evaluate)
    target = _target_model_for_task(task)
    if target:
        m = _resolve_model(target)
        if _keep(m):
            required.add(m)

    # Judge model(s) for defense+evaluate
    if isinstance(task, DefenseEvaluateTask):
        # Explicit per-task judge_model override wins (mirrors the rejudge path
        # above). The judge_method → conf/evaluation default resolution below
        # IGNORES a per-task judge_model, so a cluster judge selected only via
        # judge_model (e.g. `judge_model: wildguard`) would go UNSERVED and the
        # judge step would hang on acquire_endpoint() (2026-07-16 wildguard-judge
        # hang). Resolve it directly here so the vLLM server actually starts.
        if task.judge_model:
            judge = _resolve_model(task.judge_model)
            if _keep(judge):
                required.add(judge)
        elif task.judge_method:
            judge = _judge_model_for_method(task.judge_method)
            if _keep(judge):
                required.add(judge)
        else:
            try:
                benchmark = task.benchmark or _infer_benchmark(
                    task.source_transform_subdir)
                methods = judge_methods_for_benchmark(benchmark)
            except ValueError:
                methods = []
            for method in methods:
                judge = _judge_model_for_method(method)
                if _keep(judge):
                    required.add(judge)

        try:
            from .config import load_conf as _load_conf
            defense_cfg = _load_conf(
                "defense", override_name=task.defense,
                task_overrides=task.defense_config or None,
            )
        except Exception:
            defense_cfg = dict(task.defense_config or {})

        for key in ("guard_model", "perturbation_model", "amplifier_model",
                    "recover_model", "gate_model", "decode_model",
                    # P6 self-check defenses (2026-07-30): selfdefend's separate
                    # shadow screener, and llm_self_defense's optional separate
                    # filter (None = screen with the target, needing no server).
                    "shadow_model", "filter_model"):
            model_str = defense_cfg.get(key)
            if model_str:
                m = _resolve_model(model_str)
                if _keep(m):
                    required.add(m)

    # Prompt-transform tasks may reference a HELPER model INSIDE a
    # transformation's params (an LLM encoder's `model`, or the variance-channel
    # wrapper's paraphrase / attack-bank helper). Historically transforms used
    # API helpers so this went unnoticed; a free cluster-served helper needs a
    # vLLM server too, else the transform errors with "No ClusterModelServerManager
    # registered". Scan every transformation spec's params for `model` values.
    if isinstance(task, PromptTransformTask):
        for spec in task.transformation_list:
            params = (spec.params if isinstance(spec, TransformationSpec)
                      else spec.get("params") if isinstance(spec, dict) else None)
            for model_str in _collect_model_strings(params):
                m = _resolve_model(model_str)
                if _keep(m):
                    required.add(m)

    return required
