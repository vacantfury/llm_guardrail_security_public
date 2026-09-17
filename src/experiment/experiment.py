"""
Experiment orchestrator for Encoding × Modality jailbreaking.

Two independent parameters control execution:

  num_main_job_threads  — thread pool size inside the orchestrator process.
      Tasks arrive in queue order; each occupies a thread, runs, and
      releases the thread on completion so the next queued task can start.
      Implemented via asyncio.Semaphore.

Cluster models: vLLM servers are started before task execution and shut
down afterward.  Tasks using cluster models get endpoints via
LLMServiceFactory.

Usage:
    exp = Experiment(conf_dir, num_main_job_threads=5, num_cluster_jobs=8)
    exp.add_tasks(mixed_tasks)
    results = exp.run()
"""
from __future__ import annotations

import asyncio
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .task import run_task
from .constants import MAX_SUBMIT_JOBS_PER_USER
# Task classification + model discovery moved to model_discovery.py. Re-imported
# here (not re-implemented) so every name that was importable from
# src.experiment.experiment stays importable — src/experiment/multi_cluster.py
# and temporary_scripts/re_evaluate_all.py import from this module by name.
from .model_discovery import (  # noqa: F401
    TaskType,
    _target_model_for_task,
    _infer_task_type,
    _resolve_model,
    _KNOWN_JUDGE_METHODS,
    _judge_model_for_method,
    _collect_model_strings,
    _required_cluster_models_for_task,
    _referenced_models_for_task,
)
from llm_utils import AccountFatalError
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Config Loading ====================

def load_preset(name: str, conf_dir: Optional[Path] = None) -> "ExperimentPreset":
    """Load and validate an experiment preset from conf/experiment/<name>.yaml.

    Returns a typed ExperimentPreset; pydantic validation fails loud at this
    boundary so YAML typos surface with field paths, not deep in a runner.
    """
    from .schemas import ExperimentPreset

    if conf_dir is None:
        conf_dir = Path(__file__).resolve().parent.parent.parent / "conf"

    path = conf_dir / "experiment" / f"{name}.yaml"
    if not path.exists():
        available = sorted(
            p.stem for p in (conf_dir / "experiment").glob("*.yaml")
        )
        raise FileNotFoundError(
            f"Preset '{name}' not found at {path}\n"
            f"Available presets: {', '.join(available)}"
        )
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return ExperimentPreset.model_validate(data)


@dataclass
class TaskInfo:
    """Task metadata for scheduling. Holds a typed TaskConfig."""
    index: int
    task: Any            # TaskConfig (discriminated union); kept as Any to avoid forward-ref
    name: str

    @property
    def mode(self) -> str:
        return self.task.mode

    @property
    def task_type(self) -> TaskType:
        return _infer_task_type(self.task)

    @property
    def config(self) -> dict:
        """Plain dict view of the task (for code paths that still want it)."""
        return self.task.model_dump()


def _get_task_name(task, index: int) -> str:
    """Generate a descriptive name for a typed TaskConfig."""
    from .schemas import DefenseEvaluateTask, PromptTransformTask, TransformationSpec
    parts = [task.mode]
    if isinstance(task, PromptTransformTask):
        chain = [
            s.type if isinstance(s, TransformationSpec) else s.get("type", "?")
            for s in task.transformation_list
        ]
        parts.append("+".join(chain))
    elif isinstance(task, DefenseEvaluateTask):
        parts.append(task.defense)
        parts.append(task.target_model)
    parts.append(str(index))
    return "_".join(parts)


# ==================== Orchestrator ====================

class Experiment:
    """
    Experiment orchestrator with thread-pool scheduling and vLLM server lifecycle.
    
    The orchestrator maintains a thread pool (sized by num_main_job_threads).
    Tasks arrive in queue order; each occupies a thread from the pool, runs,
    and releases the thread on completion so the next queued task can proceed.
    
    Usage:
        exp = Experiment(conf_dir, num_main_job_threads=5, num_cluster_jobs=8)
        exp.add_tasks(mixed_tasks)
        results = exp.run()
    """

    def __init__(self, conf_dir: Optional[Path] = None,
                 num_main_job_threads: int = 5,
                 num_cluster_jobs: int = MAX_SUBMIT_JOBS_PER_USER):
        """
        Args:
            conf_dir: Path to conf/ directory.
            num_main_job_threads: Thread pool size — max concurrent tasks
                within this orchestrator process.
            num_cluster_jobs: Total SLURM job budget (orchestrator + vLLM
                servers).  Capped at MAX_SUBMIT_JOBS_PER_USER (= 8).
        """
        if conf_dir is None:
            conf_dir = Path(__file__).resolve().parent.parent.parent / "conf"
        self.conf_dir = conf_dir
        self.tasks: deque[TaskInfo] = deque()
        self.results: list[dict[str, Any]] = []
        self.num_main_job_threads = num_main_job_threads
        self.num_cluster_jobs = min(num_cluster_jobs, MAX_SUBMIT_JOBS_PER_USER)
        self._task_counter = 0
        self._server_manager = None
        # Per-model remaining-task counter; populated in _run_async() and
        # decremented in _run_task_safe() so we can mid-run-teardown a model's
        # vLLM servers as soon as the last task needing it completes.
        self._model_remaining: dict = {}

    def add_task(self, task_def) -> None:
        """Add a task to the queue.

        Accepts either a plain dict (validated via the Pydantic TaskConfig
        discriminated union) or an already-validated TaskConfig instance.
        Typed validation fails loud at this boundary — typos in YAML fields
        raise ValidationError with field paths, instead of failing deep
        inside a runner.
        """
        from .schemas import (
            DefenseEvaluateTask, PromptTransformTask,
        )
        from pydantic import TypeAdapter
        from .schemas import TaskConfig as _TaskConfigAlias

        if isinstance(task_def, (PromptTransformTask, DefenseEvaluateTask)):
            task = task_def
        else:
            # TypeAdapter handles the Annotated[Union[...], Discriminator(...)]
            task = TypeAdapter(_TaskConfigAlias).validate_python(task_def)

        info = TaskInfo(
            index=self._task_counter,
            task=task,
            name=_get_task_name(task, self._task_counter),
        )
        self.tasks.append(info)
        self._task_counter += 1
        logger.debug(
            f"Added task '{info.name}' (type={info.task_type.value}, "
            f"total: {len(self.tasks)})")

    def add_tasks(self, task_defs: list) -> None:
        """Add multiple task definitions to the queue."""
        for task_def in task_defs:
            self.add_task(task_def)

    # ==================== Cluster Server Management ====================

    def _find_cluster_models(self, tasks: list[TaskInfo]) -> set:
        """Scan tasks for unique cluster models that need vLLM servers.

        Covers BOTH the task's target model AND the judge model implied by
        judge_method (each evaluator hard-binds its canonical judge via its
        own constants.JUDGE_MODEL). Without judge inclusion, an `evaluate`
        task with an API target + cluster-hosted judge silently hangs on
        acquire_endpoint() because the judge's vLLM server was never started.
        """
        cluster_models: set = set()
        for task in tasks:
            cluster_models |= _required_cluster_models_for_task(task)
        return cluster_models

    def _context_budget_preflight(self, tasks: list[TaskInfo]) -> None:
        """Fail fast if any task asks a cluster-served model for more tokens than
        its vLLM server will accept.

        Checks the EFFECTIVE per-task budget (`target_model_config.max_tokens`,
        already merged from conf/llm/default.yaml -> the model file -> the
        preset's override), not the model file alone — the 227537 value came
        from the preset.
        """
        from llm_utils import LLMModel, Provider
        offenders = []
        for t in tasks:
            # TARGET only. `target_model_config.max_tokens` is the target's
            # budget; judges and defense-side models carry their own and would
            # false-positive against a small-context guard model.
            cfg = getattr(t.task, "target_model_config", None)
            target_name = getattr(t.task, "target_model", None)
            req_max = getattr(cfg, "max_tokens", None)
            if not req_max or not target_name:
                continue
            try:
                m = LLMModel.from_string(target_name)
            except Exception:
                continue
            if m.provider is not Provider.SLURM_CLUSTER:
                continue
            mml = (self._load_cluster_config(m) or {}).get("max_model_len")
            if mml and int(req_max) > int(mml):
                offenders.append((m.model_id, int(req_max), int(mml)))
        if not offenders:
            return
        lines = "\n".join(
            f"  - {mid}: max_tokens={rq} > max_model_len={mml}"
            for mid, rq, mml in sorted(set(offenders)))
        raise RuntimeError(
            "Context-budget preflight FAILED — vLLM would reject every request "
            "with HTTP 400 and the run would report a fabricated ASR of 0.0:\n"
            f"{lines}\n"
            "Fix: raise `max_model_len` for the model (conf/llm/<model>.yaml::"
            "cluster, or its profiles.<cluster> block) above the requested "
            "max_tokens, or lower the task's target_model_config.max_tokens. "
            "Keep max_model_len consistent across targets you intend to COMPARE "
            "— an unequal generation budget is not a comparable measurement.")

    def _arch_ceiling_preflight(self, model_configs: dict) -> None:
        """Fail fast if any cluster-served model asks vLLM for a `max_model_len`
        above its architectural context ceiling.

        vLLM refuses to START under that condition, so the server job dies on
        launch and every task depending on it hangs until its endpoint timeout.
        Distinct from `_context_budget_preflight`, which compares a task's
        requested `max_tokens` against the server's `max_model_len` — this
        compares that `max_model_len` against what the architecture can serve
        at all (`max_position_embeddings`, via the llm_utils registry row).

        Covers EVERY cluster-served model in the run — target, judge, and guard
        alike — because any of them is a vLLM server that can fail to start.

        This check was written as a Pydantic validator on `LLMConfig`
        (schemas.py) but never ran: nothing constructs `LLMConfig`, and
        `_load_cluster_config` returns a plain dict. Moved here 2026-08-02 so
        it runs on the live path, before the first sbatch.
        """
        from .schemas import _arch_row_for
        offenders = []
        for model, cfg in model_configs.items():
            mml = cfg.get("max_model_len")
            if not mml:
                continue
            row = _arch_row_for(model.model_id)
            ceiling = getattr(row, "max_context_len", None) if row else None
            if ceiling and int(mml) > int(ceiling):
                offenders.append((model.model_id, int(mml), int(ceiling)))
        if not offenders:
            return
        lines = "\n".join(
            f"  - {mid}: max_model_len={mml} > architectural ceiling {ceil}"
            for mid, mml, ceil in sorted(set(offenders)))
        raise RuntimeError(
            "Architecture-ceiling preflight FAILED — vLLM would refuse to start "
            "these servers, and every task needing them would hang until its "
            "endpoint timeout:\n"
            f"{lines}\n"
            "Fix: lower `max_model_len` in conf/llm/<model>.yaml::cluster (or "
            "its profiles.<cluster> block) to at most the ceiling shown. The "
            "ceiling comes from the model's upstream config.json "
            "max_position_embeddings and cannot be raised. Nothing was "
            "submitted.")

    def _bedrock_preflight(self, tasks: list[TaskInfo]) -> None:
        """Fail fast BEFORE launching anything if the run needs Bedrock but its
        creds are dead."""
        from llm_utils import Provider
        needs_bedrock = any(
            _referenced_models_for_task(t, {Provider.BEDROCK}) for t in tasks)
        if not needs_bedrock:
            return
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError:
            logger.warning("Bedrock preflight skipped (boto3 not importable); "
                           "per-call credential checks still apply.")
            return
        try:
            boto3.client("sts").get_caller_identity()
            logger.info("Bedrock preflight: creds live.")
        except (ClientError, BotoCoreError) as e:
            raise RuntimeError(
                "Bedrock preflight FAILED — this run needs AWS Bedrock but the "
                f"creds are not usable ({type(e).__name__}: {e}). On xc the "
                "AWS creds expire every few hours and only the box operator "
                "can re-mint them — ask him to re-mint on the box, then rerun. "
                "Nothing was submitted."
            ) from e

    def _load_cluster_config(self, model):
        """
        Load cluster server config for a model.

        Layers: conf/clusters/_defaults.yaml → the CLUSTER_PROFILE's file
        (+ its `extends` chain) → the model-specific conf/llm/<model>.yaml
        `cluster` block (applied LAST — per-model wins, so e.g. a model's
        `time_limit: 01:00:00` backfill override survives) → that block's
        `profiles.<CLUSTER_PROFILE>` sub-block, applied last of all.

        CLUSTER_PROFILE (env var, set by the per-cluster orchestrator sbatch —
        e.g. `CLUSTER_PROFILE=aicr` from run_experiment_aicr.sbatch) selects
        conf/clusters/<profile>.yaml. Unset → `nurc` (the default cluster).
        NURC and AICR are peers over _defaults.yaml; a typo'd profile fails loud.
        """
        import os
        from .config import load_conf, load_cluster_profile, _deep_merge, CONF_DIR

        profile = os.environ.get("CLUSTER_PROFILE", "").strip() or "nurc"
        base = load_cluster_profile(profile, CONF_DIR)

        # Per-model cluster overrides (conf/llm/<model>.yaml::cluster). default.yaml
        # no longer carries a `cluster:` block, so this returns just the model's block.
        per_model = load_conf(
            "llm", section="cluster",
            match_field="model.model", match_value=model.model_id)

        # `profiles` is a routing table, never a server param — pull it out before
        # merging so it can never reach the sbatch writer as a stray key.
        per_model = dict(per_model or {})
        profile_overrides = (per_model.pop("profiles", None) or {}).get(profile, {})

        merged = _deep_merge(base, per_model)
        if profile_overrides:
            merged = _deep_merge(merged, profile_overrides)
            logger.info(
                f"Cluster config [{model.model_id}]: applied profiles.{profile} "
                f"overrides {sorted(profile_overrides)}")
        return merged

    def _count_existing_gpu_jobs(self) -> int:
        """Count this user's existing GPU-holding SLURM jobs.

        Counting *every* job (the old behaviour) was wrong in both directions:
        it charged CPU-only jobs — including this orchestrator and unrelated
        `mj` runs — against a GPU quota (which crippled legitimate submissions
        on AICR), while letting a 5th GPU server through on NURC where only 4
        fit. A server left PENDING deadlocks the run, because the orchestrator
        waits for one endpoint per model.

        Excludes the orchestrator's own job (via SLURM_JOB_ID) and any job that
        requests no GRES.
        """
        import os
        import subprocess

        user = os.environ.get("USER")
        if not user:
            return 0
        # BOTH TRES fields are needed, because the two clusters request GPUs
        # differently and each populates only its own field (measured 2026-07-25):
        #   NURC  `--gres=gpu:N` -> tres-per-node = "gres/gpu:1", tres-per-job = N/A
        #   AICR  `--gpus=N`     -> tres-per-node = "N/A",        tres-per-job = "gres/gpu:1"
        # The short `%b` code is tres-per-node ALONE, so keying on it counted zero
        # GPU jobs on AICR. `tres-alloc` is not a substitute: it is empty while a
        # job is still PENDING, and a pending server holds quota just as a running
        # one does.
        try:
            result = subprocess.run(
                ["squeue", "-u", user, "-h",
                 "-O", "JobID:.24,tres-per-node:.40,tres-per-job:.40"],
                capture_output=True, text=True, timeout=15)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("squeue unavailable; skipping pre-flight budget check.")
            return 0
        if result.returncode != 0:
            return 0
        own_job = os.environ.get("SLURM_JOB_ID")
        n = 0
        for line in result.stdout.splitlines():
            fields = line.split()
            if not fields:
                continue
            job_id, rest = fields[0], " ".join(fields[1:])
            if own_job and job_id == own_job:
                continue
            if "gpu" in rest.lower():
                n += 1
        return n

    @staticmethod
    def _pick_port_block(base_port: int, count: int, span: int = 400) -> int:
        """Pick a port block this orchestrator can own alone.

        Two mechanisms, because either alone is insufficient:

        1. SPREAD by SLURM job id. Concurrent orchestrators get deterministically
           different starting offsets, so they do not all begin at the YAML base.
           A pure free-port probe cannot do this on its own — the orchestrator
           never binds the port (its vLLM server does, ~30s later), so two
           orchestrators probing at the same moment both see the base as free.
        2. PROBE upward from there for a block with nothing already listening,
           which catches a collision the spread happens to land on.

        The probe is exact on a single-node cluster (servers share the
        orchestrator's node) and merely conservative elsewhere: a port busy on
        this node but free on the server's node only pushes the block higher,
        which costs nothing. Returns `base_port` unchanged if `span` is
        exhausted — same behavior as before this existed.
        """
        import os
        import socket

        stride = max(count, 4)
        try:
            job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
        except ValueError:
            job_id = 0
        start_base = base_port + (job_id % 32) * stride

        def block_free(start: int) -> bool:
            for p in range(start, start + count):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.bind(("0.0.0.0", p))
                except OSError:
                    return False
                finally:
                    s.close()
            return True

        for start in range(start_base, start_base + span):
            if block_free(start):
                if start != base_port:
                    logger.info(
                        f"Port block {start}-{start + count - 1} (YAML base "
                        f"{base_port}, SLURM_JOB_ID={job_id or 'unset'}) — "
                        f"blocks are spread per job so concurrent orchestrators "
                        f"on a single-node cluster cannot share a port.")
                return start
        logger.warning(
            f"No free {count}-port block in {start_base}-{start_base + span}; "
            f"falling back to the YAML base {base_port}. If servers now fail to "
            f"bind, check `ss -ltn` on the node for stale vLLM processes.")
        return base_port

    def _wait_for_servers_parallel(self, models: list) -> set:
        """Wait for the first server of each model to come up, in parallel.

        Returns the set of models whose first server failed to start —
        the caller skips tasks for those models upfront so they don't hang
        waiting for an endpoint that will never exist.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        failed: set = set()
        if not models:
            return failed
        with ThreadPoolExecutor(
            max_workers=len(models), thread_name_prefix="server-wait"
        ) as ex:
            futures = {
                ex.submit(self._server_manager.wait_for_first_server, m): m
                for m in models
            }
            for fut in as_completed(futures):
                model = futures[fut]
                try:
                    endpoint = fut.result()
                    logger.info(f"  → {model.model_id}: first server ready at {endpoint}")
                except RuntimeError as e:
                    logger.error(f"  ✗ {model.model_id}: server failed — {e}")
                    failed.add(model)
        return failed

    def _setup_cluster_servers(self, cluster_models: set) -> None:
        """Submit vLLM servers for all cluster models and return immediately.

        Fire-and-forget: this method does NOT wait for any server to become
        healthy. Tasks block on `acquire_endpoint(model)` when they need a
        not-yet-ready model, and the manager's monitor thread populates
        the pool as servers come up.

        This is what gives us "dispatch as servers come up" behavior: fast
        judges' tasks run as soon as that judge is ready, without being
        gated by slow/queued judges. A slow judge (e.g. queue-blocked on
        QoS) only delays the tasks that depend on it, not the whole run.

        Budget enforcement is two-layered (still done synchronously here):
          1. Pre-flight against MAX_SUBMIT_JOBS_PER_USER minus the user's
             existing SLURM jobs.
          2. Cap num_instances per model to never exceed each model's YAML
             request — capping never *promotes* a model's count.

        Port assignment uses a cumulative offset over a stable model order
        (sorted by model_id) — prevents collisions with mixed num_instances
        and keeps assignments reproducible across runs.
        """
        from llm_utils.cluster_server_manager import ClusterModelServerManager
        from llm_utils import LLMServiceFactory

        self._server_manager = ClusterModelServerManager()

        # Stable order: sort by model_id so port assignment is reproducible
        # and `enumerate(set)`'s non-determinism doesn't bite us.
        cluster_models_sorted = sorted(cluster_models, key=lambda m: m.model_id)
        model_configs = {m: self._load_cluster_config(m) for m in cluster_models_sorted}

        # Before any budget math or submission: would vLLM even start?
        self._arch_ceiling_preflight(model_configs)

        # Per-cluster GPU-server cap (conf/clusters/<profile>.yaml::max_gpu_jobs,
        # falling back to ::budget). This is a GPU quota, NOT a job quota — see
        # _count_existing_gpu_jobs. MAX_SUBMIT_JOBS_PER_USER is the fail-safe.
        import os
        from .config import load_cluster_profile
        profile = os.environ.get("CLUSTER_PROFILE", "").strip() or "nurc"
        try:
            _prof = load_cluster_profile(profile, self.conf_dir)
            gpu_cap = int(_prof.get("max_gpu_jobs",
                                    _prof.get("budget", MAX_SUBMIT_JOBS_PER_USER)))
        except Exception:
            gpu_cap = MAX_SUBMIT_JOBS_PER_USER

        # The orchestrator itself holds no GPU (it runs on a CPU partition), so
        # it is NOT charged against the GPU cap. `num_cluster_jobs` keeps its
        # historical meaning — orchestrator + servers — hence the -1 here.
        servers_wanted = self.num_cluster_jobs - 1
        existing = self._count_existing_gpu_jobs()
        available_slots = min(servers_wanted, gpu_cap - existing)
        logger.info(
            f"Pre-flight [{profile}]: gpu_cap={gpu_cap}, existing GPU jobs={existing}, "
            f"servers_wanted={servers_wanted} -> available_slots={available_slots}")
        if available_slots < 1:
            raise RuntimeError(
                f"No GPU slots available after pre-flight on '{profile}': "
                f"existing GPU jobs={existing}, gpu_cap={gpu_cap}, "
                f"num_cluster_jobs={self.num_cluster_jobs}. "
                f"Aborting release (a PENDING server would deadlock the run).")

        # Cap num_instances proportionally if over budget — but never promote.
        requested = {m: cfg.get("num_instances", 1) for m, cfg in model_configs.items()}
        total_requested = sum(requested.values())
        if total_requested > available_slots:
            n_models = len(model_configs)
            per_model_cap = max(available_slots // n_models, 1)
            logger.warning(
                f"Requested {total_requested} vLLM instances exceeds "
                f"{available_slots} slots; capping each model to "
                f"<= {per_model_cap} (never promotes above YAML request).")
            for m, cfg in model_configs.items():
                new = min(requested[m], per_model_cap)
                if new != requested[m]:
                    logger.info(f"  {m.model_id}: {requested[m]} -> {new} instance(s)")
                cfg["num_instances"] = new

        total_instances = sum(c["num_instances"] for c in model_configs.values())
        base_port = self._pick_port_block(
            next(iter(model_configs.values()))["port"], total_instances)
        next_port = base_port
        for m in cluster_models_sorted:
            cfg = model_configs[m]
            cfg["port"] = next_port
            next_port += cfg["num_instances"]
            logger.info(
                f"Starting {cfg['num_instances']} vLLM server(s) for "
                f"{m.model_id} on port(s) {cfg['port']}-"
                f"{cfg['port'] + cfg['num_instances'] - 1}")
            self._server_manager.start_server(m, cfg)

        # Register with factory so tasks auto-get endpoints when they call
        # acquire_endpoint(). No upfront blocking wait — tasks dispatch
        # immediately; slow servers only block their own dependent tasks.
        LLMServiceFactory.set_server_manager(self._server_manager)
        logger.info(
            "Server submission complete. Tasks will start dispatching "
            "immediately; per-task acquire_endpoint() will wait per-server.")

    def _teardown_cluster_servers(self) -> None:
        """Shut down all vLLM servers and clear factory-side singleton state."""
        if self._server_manager:
            logger.info("Shutting down cluster vLLM servers...")
            self._server_manager.shutdown_all()
            self._server_manager = None
        # Clear the factory's class-level singleton so subsequent Experiment
        # runs in the same process don't see a stale (shut-down) manager.
        from llm_utils import LLMServiceFactory
        LLMServiceFactory.clear_server_manager()

    def _maybe_release_model(self, model) -> None:
        """Decrement remaining-task count for a model; scancel its servers when 0.

        Lets the orchestrator free a heavy model's GPU allocation mid-experiment
        as soon as the last task needing it finishes — instead of holding
        4× A100s pinned for the entire experiment.

        Escape hatch: set env DISABLE_MIDRUN_TEARDOWN=1 to skip this entirely and
        hold every server until the final teardown. The remaining-count
        optimization can race on a mixed-benchmark matrix — a model whose count
        hits 0 is freed while a still-queued task (a different benchmark's tail,
        e.g. benign ORBench cells after the harmful block) still needs it,
        failing that task with a vanished endpoint. Use the flag when GPUs are
        not the binding constraint (small runs comfortably under budget).
        Observed 2026-07-18: 16 benign cells lost to qwen teardown after the
        110 harmful cells finished.
        """
        import os
        if os.environ.get("DISABLE_MIDRUN_TEARDOWN"):
            return
        if not self._model_remaining or model not in self._model_remaining:
            return
        remaining = self._model_remaining[model] - 1
        self._model_remaining[model] = remaining
        if remaining <= 0 and self._server_manager is not None:
            logger.info(
                f"All tasks for {model.model_id} complete; "
                f"shutting down its vLLM servers to free GPUs.")
            try:
                self._server_manager.shutdown_model(model)
            except Exception as e:
                logger.warning(f"shutdown_model({model.model_id}) failed: {e}")

    # ==================== Async Execution ====================

    async def _run_task_safe(
        self, task: TaskInfo, semaphore: asyncio.Semaphore,
        queue_name: str, total: int,
    ) -> dict[str, Any]:
        """Execute a single task with semaphore-based concurrency control.

        No upfront skip — if the task's required cluster model failed to
        start, the failure surfaces inside the runner when it calls
        acquire_endpoint(), which short-circuits on all-jobs-failed. The
        runner's try/except catches and reports the failure cleanly.

        Equivalent outcome to the old upfront-skip, but doesn't require
        the orchestrator to know about failed models in advance — which is
        what unblocks "dispatch as servers come up" (some tasks dispatch
        before slow servers are even known to have failed or succeeded).
        """
        # Resolve cluster-model dependencies once so we can decrement
        # remaining counts on completion (mid-run teardown).
        required_cluster = _required_cluster_models_for_task(task)

        async with semaphore:
            logger.info(f"\n{'~'*70}")
            logger.info(f"[{queue_name}] Executing Task {task.index+1}/{total}: {task.name}")
            logger.info(f"{'~'*70}\n")

            t0 = time.time()
            try:
                # Run synchronous task in thread pool
                result = await asyncio.to_thread(run_task, task.task)
                elapsed = time.time() - t0
                result["task_name"] = task.name
                result["original_index"] = task.index
                result["status"] = result.get("status", "success")
                result["elapsed_seconds"] = round(elapsed, 1)

                logger.info(f"[{queue_name}] Completed: {task.name} ({elapsed:.1f}s)")
                return result
            except AccountFatalError as e:
                # Account-global failure (invalid key / no credits): every
                # remaining task on this provider would fail identically, so
                # re-raise to ABORT the whole run fast with the actionable
                # message — do NOT record a degenerate 'failed' cell and let the
                # rest of the matrix grind on. (Per-model/other errors below stay
                # task-local.)
                logger.error(
                    f"[{queue_name}] ABORTING RUN — account-fatal on {task.name}: {e}")
                raise
            except Exception as e:
                elapsed = time.time() - t0
                err_str = str(e)
                err_summary = err_str[:500] + "..." if len(err_str) > 500 else err_str
                tb = traceback.format_exc()
                tb_lines = tb.splitlines()
                tb_short = "\n".join(tb_lines[:20] + (["  ... (truncated)"] if len(tb_lines) > 20 else []))
                logger.error(f"[{queue_name}] Failed: {task.name} ({elapsed:.1f}s) — {err_summary}")
                logger.error(tb_short)
                return {
                    "task_name": task.name,
                    "original_index": task.index,
                    "status": "failed",
                    "error": err_summary,
                    "traceback": tb_short,
                    "elapsed_seconds": round(elapsed, 1),
                }
            finally:
                # Mid-run teardown: decrement remaining count and shut down
                # this model's servers if it was the last task needing them.
                for m in required_cluster:
                    self._maybe_release_model(m)

    async def _run_async(self) -> list[dict[str, Any]]:
        """
        Execute all tasks with thread-pool scheduling.
        
        Tasks share one semaphore (num_main_job_threads) acting as a thread
        pool: each task acquires a slot in queue order, runs, and releases.
        Cluster models: vLLM servers are started before tasks run,
        and shut down after all tasks complete (or on error).
        """
        if not self.tasks:
            logger.warning("No tasks to execute")
            return []

        tasks_to_run = list(self.tasks)
        self.tasks.clear()
        total = len(tasks_to_run)

        # Build per-model remaining-task counts (covers target AND judge).
        # Used by _maybe_release_model() for mid-run server teardown.
        self._model_remaining = {}
        for task in tasks_to_run:
            for m in _required_cluster_models_for_task(task):
                self._model_remaining[m] = self._model_remaining.get(m, 0) + 1

        # Fail fast on dead Bedrock creds before submitting any server / running
        # any task (raises RuntimeError with the re-mint fix; no-op if no Bedrock).
        self._bedrock_preflight(tasks_to_run)

        self._context_budget_preflight(tasks_to_run)

        try:
            # Submit vLLM servers for cluster models. Non-blocking: tasks
            # dispatch immediately and per-task acquire_endpoint() handles
            # the wait for each model independently.
            cluster_models = self._find_cluster_models(tasks_to_run)
            if cluster_models:
                model_names = sorted(m.model_id for m in cluster_models)
                logger.info(f"Cluster models detected: {model_names}")
                self._setup_cluster_servers(cluster_models)

            logger.info(f"\n{'='*70}")
            logger.info(f"EXPERIMENT: {total} tasks (thread pool: {self.num_main_job_threads}, "
                        f"cluster jobs: {self.num_cluster_jobs})")
            logger.info(f"{'='*70}\n")

            # Thread pool: tasks acquire a slot in order, run, release.
            # Tasks needing a slow/failed cluster model will block per-task
            # on acquire_endpoint() (or fail-fast on its all-failed short-
            # circuit), while tasks needing already-ready models proceed.
            sem = asyncio.Semaphore(self.num_main_job_threads)
            coroutines = [
                self._run_task_safe(task, sem, task.mode.upper(), total)
                for task in tasks_to_run
            ]

            self.results = list(await asyncio.gather(*coroutines))

        finally:
            # Always shut down cluster servers, even on error
            self._teardown_cluster_servers()
            self._model_remaining = {}

        self._print_summary()
        return self.results

    def run(self) -> list[dict[str, Any]]:
        """Execute tasks (sync entry point)."""
        return asyncio.run(self._run_async())

    def _print_summary(self):
        """Print experiment summary."""
        logger.info(f"\n{'='*70}")
        logger.info("EXPERIMENT SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total tasks executed: {len(self.results)}")

        successful = sum(1 for r in self.results if r.get("status") == "success")
        failed = len(self.results) - successful

        logger.info(f"Successful: {successful}")
        if failed > 0:
            logger.info(f"Failed: {failed}")
            for r in self.results:
                if r.get("status") == "failed":
                    logger.info(f"  ✗ {r.get('task_name', 'unknown')}: {r.get('error', 'unknown')}")

        # Per-task timing
        for r in self.results:
            elapsed = r.get("elapsed_seconds", 0)
            status = "✓" if r.get("status") == "success" else "✗"
            logger.info(f"  {status} {r.get('task_name', 'unknown'):40s} {elapsed:>7.1f}s")

        # Aggregate statistics
        total_prompts = sum(
            r.get('count', r.get('num_prompts', 0)) for r in self.results
            if isinstance(r, dict)
        )
        if total_prompts:
            logger.info(f"Total prompts processed: {total_prompts}")

        total_elapsed = sum(r.get("elapsed_seconds", 0) for r in self.results)
        logger.info(f"Total task time: {total_elapsed:.1f}s")
        logger.info(f"{'='*70}\n")

    def __repr__(self) -> str:
        return (f"Experiment(tasks={len(self.tasks)}, "
                f"threads={self.num_main_job_threads}, "
                f"cluster_jobs={self.num_cluster_jobs})")


# ==================== Convenience ====================

def run_experiment_from_preset(preset_name: str, conf_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """Run an experiment from a preset name."""
    if conf_dir is None:
        conf_dir = Path(__file__).resolve().parent.parent.parent / "conf"
    preset = load_preset(preset_name, conf_dir)

    # Namespace outputs by paper line, inferred from the preset subdir
    # (e.g. 'autoattack_defense/experiment' -> outputs/autoattack_defense/...).
    from src.utils.experiment import set_current_paper
    set_current_paper(preset_name.split("/")[0] if "/" in preset_name else None)

    exp = Experiment(
        conf_dir,
        num_main_job_threads=preset.num_main_job_threads,
        num_cluster_jobs=preset.num_cluster_jobs,
    )
    exp.add_tasks(preset.tasks)
    return exp.run()
