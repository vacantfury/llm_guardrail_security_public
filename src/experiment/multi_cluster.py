"""Split ONE experiment preset across an ordered pool of SLURM clusters
(AICR first, then NURC), submitting each cluster its own sub-preset over ssh.

Design (why this is a thin pre-submit layer, not a runtime change):
    Every SLURM call in ``ClusterModelServerManager`` (sbatch/squeue/scontrol)
    is a LOCAL subprocess — the orchestrator assumes it runs on the target
    cluster's login node, and "which cluster" is purely where the process runs
    plus the ``CLUSTER_PROFILE`` env var. Since the code is synced to BOTH
    clusters, each cluster runs the existing single-cluster orchestrator
    natively against its own local SLURM. So multi-cluster needs no runtime
    federation: we only (a) split the task matrix here, then (b) ssh each
    cluster its own sub-preset + ``sbatch`` wrapper. Zero edits to the
    orchestrator runtime.

Split key = the full set of *cluster-served* models each task needs
    (target INTERSECT judge INTERSECT guard), computed by reusing the pipeline's
    own ``_required_cluster_models_for_task``. API-hosted models drop out (they
    are not vLLM servers), so the split is target-only when the judge is an API
    model (e.g. gpt-5-nano) and target+judge when the judge is itself served.
    A cell is atomic: it runs on ONE cluster, which serves every model it
    touches — a pipeline is never split across clusters.

Placement = greedy, pool-ordered, capability-filtered. Pack whole tasks onto the
    first CAPABLE cluster whose remaining server budget fits them (preset order),
    overflow the rest to the next. "Capable" = holds the credentials the task
    needs (routing policy, 2026-07-18): a task's Bedrock models require a
    ``bedrock`` cluster (xc), its non-Bedrock API models require an ``api_keys``
    cluster (aicr/nurc), GPU-served models run anywhere. Because the pool lists
    xc THIRD, GPU work fills aicr->nurc->xc — xc's GPUs are the THIRD tier, a
    normal option once aicr/nurc are capped or degraded (not an emergency one).
    A model shared across a split is served once per cluster (accepted
    duplication). A ``pins`` map forces every task needing a given model onto a
    named cluster. A task needing both Bedrock and non-Bedrock API creds is
    unsatisfiable (no cluster has both) and reported as a clear leftover.

DRY-RUN by default: :func:`dispatch` writes the sub-presets locally and returns
    the plan + exact ssh commands, submitting nothing. Actual submission happens
    only under ``submit=True`` (and only after the code is synced to both
    clusters). This module never self-initiates a cluster run.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class DispatchError(Exception):
    """Raised when a preset cannot be split across the configured pool."""


# ==================== Pool config ====================

@dataclass(frozen=True)
class ClusterSpec:
    """One cluster in the ordered dispatch pool.

    budget = max concurrent vLLM SERVERS the orchestrator may hold on this cluster
    (its GPU-QOS concurrent-job ceiling; the orchestrator runs on a separate CPU
    partition/QOS and does NOT count against it)."""
    name: str
    ssh: str                 # ssh alias/target in ~/.ssh/config (private)
    repo: str                # repo path on the cluster (private; may start with ~)
    sbatch: str              # sbatch wrapper, from conf/clusters/<name>.yaml
    budget: int              # max concurrent vLLM servers, from conf/clusters/<name>.yaml
    max_submit: int          # QOS submit cap, from conf/clusters/<name>.yaml
    bedrock: bool = False    # can invoke AWS Bedrock (xc only)
    api_keys: bool = True    # has injected non-Bedrock API keys (xc flagged false by routing preference)


def load_pool(path: Path, conf_dir: Path) -> tuple[list[ClusterSpec], dict[str, str]]:
    """Load the ordered cluster pool + optional model pins.

    The pool file (path, gitignored) carries ONLY the private connection + order
    + pins: `clusters: [{name, ssh, repo}, ...]` and `pins`. Each cluster's public
    server/limit config (sbatch, budget, max_submit) is read from the committed
    conf/clusters/<name>.yaml via config.load_cluster_profile.

    Returns (clusters, pins) where pins maps a model_id -> cluster name.
    """
    from .config import load_cluster_profile
    from .constants import MAX_SUBMIT_JOBS_PER_USER

    if not path.exists():
        raise DispatchError(
            f"cluster pool config not found: {path}\n"
            f"Copy conf/cluster_pool.example.yaml to {path} and fill in your "
            f"cluster ssh aliases / repo paths (the real file is gitignored)."
        )
    data = yaml.safe_load(path.read_text()) or {}
    raw_clusters = data.get("clusters") or []
    if not raw_clusters:
        raise DispatchError(f"{path} has no 'clusters' list.")

    clusters: list[ClusterSpec] = []
    for i, c in enumerate(raw_clusters):
        missing = {"name", "ssh", "repo"} - set(c)
        if missing:
            raise DispatchError(
                f"cluster #{i} in {path} is missing keys: {sorted(missing)}")
        name = str(c["name"])
        try:
            prof = load_cluster_profile(name, conf_dir)
        except (FileNotFoundError, ValueError) as e:
            raise DispatchError(
                f"cluster '{name}' in {path} has no committed "
                f"conf/clusters/{name}.yaml: {e}")
        sbatch = prof.get("sbatch")
        budget = prof.get("budget")
        if not sbatch or budget is None:
            raise DispatchError(
                f"conf/clusters/{name}.yaml must define 'sbatch' and 'budget'.")
        clusters.append(ClusterSpec(
            name=name,
            ssh=str(c["ssh"]),
            repo=str(c["repo"]),
            sbatch=str(sbatch),
            budget=int(budget),
            max_submit=int(prof.get("max_submit", MAX_SUBMIT_JOBS_PER_USER)),
            bedrock=bool(prof.get("bedrock", False)),
            api_keys=bool(prof.get("api_keys", True)),
        ))

    names = [c.name for c in clusters]
    if len(set(names)) != len(names):
        raise DispatchError(f"duplicate cluster names in {path}: {names}")

    pins = {str(k): str(v) for k, v in (data.get("pins") or {}).items()}
    for model, cname in pins.items():
        if cname not in names:
            raise DispatchError(
                f"pin '{model}' -> '{cname}' references an unknown cluster "
                f"(known: {names}).")
    return clusters, pins


# ==================== Split plan ====================

@dataclass(frozen=True)
class TaskNeed:
    """What one task needs, for routing. `gpu_models` are the vLLM-served model
    ids (each consumes a server slot / budget); the two booleans are capability
    demands that gate which cluster can run it (see ClusterSpec)."""
    idx: int
    gpu_models: frozenset          # SLURM_CLUSTER model ids needing a vLLM server
    needs_bedrock: bool = False    # references ≥1 Bedrock model → bedrock cluster only
    needs_other_api: bool = False  # references ≥1 non-Bedrock API model → api_keys cluster


def _cluster_can_run(cluster: ClusterSpec, need: TaskNeed) -> bool:
    """Capability gate: does this cluster hold the credentials the task needs?
    (Budget is checked separately — this is the hard yes/no on keys.)"""
    if need.needs_bedrock and not cluster.bedrock:
        return False
    if need.needs_other_api and not cluster.api_keys:
        return False
    return True


@dataclass
class ClusterAssignment:
    cluster: ClusterSpec
    task_indices: list[int]
    server_models: set[str]         # model_ids that will be served on this cluster

    @property
    def num_cluster_jobs(self) -> int:
        """orchestrator (1) + one vLLM job per distinct served model, capped at
        this cluster's max_submit."""
        return min(len(self.server_models) + 1, self.cluster.max_submit)

    @property
    def active(self) -> bool:
        return bool(self.task_indices)


@dataclass
class SplitPlan:
    assignments: list[ClusterAssignment]
    leftover: list[int]                 # task indices that fit no cluster
    total_cluster_models: set            # distinct served models across the whole preset
    leftover_reasons: dict = field(default_factory=dict)  # idx -> why it couldn't place


def plan_split(
    task_needs: list[TaskNeed],
    clusters: list[ClusterSpec],
    pins: dict[str, str],
) -> SplitPlan:
    """Assign tasks to clusters in POOL ORDER, honoring per-cluster capability.

    task_needs: one TaskNeed per task. Pure function — no I/O, no pipeline import
    — so the routing is unit-testable with fabricated inputs.
    """
    assign: dict[str, list[int]] = {c.name: [] for c in clusters}
    servers: dict[str, set[str]] = {c.name: set() for c in clusters}
    leftover_reasons: dict[int, str] = {}

    # Phase 1 — pin-forced tasks. A pin names the cluster for every task needing
    # that model. Two models pinned to DIFFERENT clusters, or a pin to a cluster
    # that lacks the task's credentials, is a user error surfaced loudly.
    pending: list[TaskNeed] = []
    for need in task_needs:
        pinned = {pins[m] for m in need.gpu_models if m in pins}
        if len(pinned) > 1:
            raise DispatchError(
                f"task #{need.idx} needs models pinned to different clusters "
                f"{sorted(pinned)}; a single pipeline cannot be split across "
                f"clusters. Fix the pins so its models share one cluster.")
        if pinned:
            cname = next(iter(pinned))
            by_name = {c.name: c for c in clusters}
            if not _cluster_can_run(by_name[cname], need):
                raise DispatchError(
                    f"task #{need.idx} is pinned to '{cname}' but that cluster "
                    f"lacks the credentials it needs (needs_bedrock="
                    f"{need.needs_bedrock}, needs_other_api={need.needs_other_api}). "
                    f"Repin to a capable cluster.")
            assign[cname].append(need.idx)
            servers[cname] |= need.gpu_models
        else:
            pending.append(need)

    # Pins may already overflow a cluster's budget — surface that clearly.
    for c in clusters:
        if len(servers[c.name]) > c.budget:
            raise DispatchError(
                f"pins force {len(servers[c.name])} servers onto '{c.name}' "
                f"(budget {c.budget}): {sorted(servers[c.name])}. "
                f"Raise its budget or repin.")

    # Phase 2 — capability-filtered greedy fill, pool order (preset order within
    # each cluster). For each still-pending task, walk clusters in pool order and
    # take the first that (a) CAN run it (credentials) and (b) has server budget.
    for need in list(pending):
        capable = [c for c in clusters if _cluster_can_run(c, need)]
        if not capable:
            both = need.needs_bedrock and need.needs_other_api
            leftover_reasons[need.idx] = (
                "needs both Bedrock AND non-Bedrock API creds — no single "
                "cluster has both (run the Bedrock and API stages separately)"
                if both else
                f"no cluster satisfies its credentials (needs_bedrock="
                f"{need.needs_bedrock}, needs_other_api={need.needs_other_api})")
            continue
        placed = False
        for c in capable:
            new = need.gpu_models - servers[c.name]
            if len(servers[c.name]) + len(new) <= c.budget:
                assign[c.name].append(need.idx)
                servers[c.name] |= need.gpu_models
                placed = True
                break
        if not placed:
            leftover_reasons[need.idx] = (
                f"needs {len(need.gpu_models)} server(s) but no capable cluster "
                f"has room: "
                + ", ".join(f"{c.name}(budget {c.budget}, "
                            f"used {len(servers[c.name])})" for c in capable))

    leftover = sorted(leftover_reasons)
    assignments = [
        ClusterAssignment(
            cluster=c,
            task_indices=sorted(assign[c.name]),
            server_models=servers[c.name],
        )
        for c in clusters
    ]
    total: set = set()
    for need in task_needs:
        total |= need.gpu_models
    return SplitPlan(assignments=assignments, leftover=leftover,
                     total_cluster_models=total,
                     leftover_reasons=leftover_reasons)


# ==================== Pipeline coupling ====================

def _model_key(m) -> str:
    """Stable identity for an LLMModel (its model_id)."""
    return getattr(m, "model_id", None) or getattr(m, "name", None) or str(m)


def compute_task_needs(preset) -> list[TaskNeed]:
    """Per task, its routing needs: GPU-served model ids + Bedrock / other-API
    capability demands."""
    from .experiment import (
        _referenced_models_for_task, TaskInfo, _get_task_name,
    )
    from llm_utils import Provider

    out: list[TaskNeed] = []
    for i, task in enumerate(preset.tasks):
        info = TaskInfo(index=i, task=task, name=_get_task_name(task, i))
        all_models = _referenced_models_for_task(info, None)
        gpu = frozenset(
            _model_key(m) for m in all_models if m.provider == Provider.SLURM_CLUSTER)
        needs_bedrock = any(m.provider == Provider.BEDROCK for m in all_models)
        # "other API" = anything needing a key that isn't Bedrock and isn't a
        # GPU-served or in-process-local model (OpenAI/Anthropic/Google/DeepSeek/
        # Z.AI/xAI/Moonshot). These need the injected API keys → aicr/nurc.
        needs_other_api = any(
            m.provider not in (Provider.SLURM_CLUSTER, Provider.BEDROCK, Provider.LOCAL)
            for m in all_models)
        out.append(TaskNeed(idx=i, gpu_models=gpu,
                            needs_bedrock=needs_bedrock,
                            needs_other_api=needs_other_api))
    return out


# Back-compat alias: old name returned (idx, gpu_models) tuples. Kept so any
# external caller / test importing it still resolves; new code uses
# compute_task_needs (richer, routing-aware).
def compute_task_models(preset) -> list[tuple[int, frozenset]]:
    return [(n.idx, n.gpu_models) for n in compute_task_needs(preset)]


# ==================== Sub-preset rendering ====================

def subpreset_name(preset_name: str, cluster_name: str) -> str:
    """Sub-preset name, kept under the ORIGINAL paper subdir so output
    namespacing (paper = first path component) is preserved.

    'autoattack_defense/experiment' + 'aicr'
        -> 'autoattack_defense/_mc_experiment_aicr'
    'test' + 'nurc' -> '_mc_test_nurc'
    """
    if "/" in preset_name:
        paper, base = preset_name.split("/", 1)
        return f"{paper}/_mc_{base.replace('/', '_')}_{cluster_name}"
    return f"_mc_{preset_name}_{cluster_name}"


def render_subpreset(preset, assignment: ClusterAssignment) -> dict:
    """Build the sub-preset dict for one cluster (its assigned tasks only)."""
    tasks = [preset.tasks[i].model_dump(mode="json")
             for i in assignment.task_indices]
    return {
        "num_main_job_threads": preset.num_main_job_threads,
        "num_cluster_jobs": assignment.num_cluster_jobs,
        "tasks": tasks,
    }


def write_subpreset(conf_dir: Path, name: str, data: dict) -> Path:
    """Write a sub-preset YAML under conf/experiment/<name>.yaml locally."""
    path = conf_dir / "experiment" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# GENERATED by dispatch.py (multi-cluster split) — do not edit by hand.\n"
        f"# Regenerate with: python dispatch.py <preset>\n"
    )
    path.write_text(header + yaml.safe_dump(data, sort_keys=False, width=120))
    return path


# ==================== ssh command construction ====================

def _remote_yaml_path(cluster: ClusterSpec, name: str) -> str:
    return f"{cluster.repo}/conf/experiment/{name}.yaml"


def dispatch_commands(cluster: ClusterSpec, name: str) -> list[str]:
    """The exact ssh command sequence to place + submit one cluster's sub-preset.

    Returned as human-readable strings (also what --submit executes). Remote
    paths are NOT single-quoted so a leading ~ expands on the remote shell.
    """
    remote_yaml = _remote_yaml_path(cluster, name)
    remote_dir = remote_yaml.rsplit("/", 1)[0]
    return [
        f"ssh {cluster.ssh} 'mkdir -p {remote_dir}'",
        f"cat conf/experiment/{name}.yaml | ssh {cluster.ssh} 'cat > {remote_yaml}'",
        f"ssh {cluster.ssh} 'cd {cluster.repo} && sbatch {cluster.sbatch} {name}'",
    ]


def _run_dispatch(cluster: ClusterSpec, name: str, conf_dir: Path) -> str:
    """Actually place the sub-preset on the cluster and submit it. Returns the
    sbatch stdout (job id line). Only called under submit=True."""
    remote_yaml = _remote_yaml_path(cluster, name)
    remote_dir = remote_yaml.rsplit("/", 1)[0]
    local_yaml = conf_dir / "experiment" / f"{name}.yaml"

    subprocess.run(["ssh", cluster.ssh, f"mkdir -p {remote_dir}"],
                   check=True, timeout=60)
    with open(local_yaml, "rb") as fh:
        subprocess.run(["ssh", cluster.ssh, f"cat > {remote_yaml}"],
                       stdin=fh, check=True, timeout=60)
    result = subprocess.run(
        ["ssh", cluster.ssh, f"cd {cluster.repo} && sbatch {cluster.sbatch} {name}"],
        capture_output=True, text=True, check=True, timeout=120)
    return result.stdout.strip()


# ==================== Orchestration ====================

@dataclass
class DispatchResult:
    preset_name: str
    plan: SplitPlan
    subpresets: dict[str, str]           # cluster name -> sub-preset name
    written: dict[str, Path]             # cluster name -> local sub-preset path
    commands: dict[str, list[str]]       # cluster name -> ssh command list
    submitted: dict[str, str] = field(default_factory=dict)  # cluster -> sbatch stdout
    health_notes: list = field(default_factory=list)  # human lines: dropped clusters / disabled caps (empty = all healthy)


def dispatch(
    preset_name: str,
    pool_path: Path,
    conf_dir: Path,
    submit: bool = False,
    probe: bool = True,
) -> DispatchResult:
    """Plan the split, write sub-presets locally, and (only if submit=True)
    ssh them to each cluster and sbatch. Dry-run otherwise.

    When ``probe`` (default), a runtime health preflight (cluster_health.py) runs
    BEFORE routing: unreachable / SLURM-down clusters are dropped and dead
    capabilities (e.g. xc's expired Bedrock creds) are flipped off, so the pure
    router only places work where it can actually run — and a needed capability
    that is dead everywhere aborts early with an actionable fix instead of failing
    a launched matrix cell-by-cell. Pass ``probe=False`` to skip the ssh probes.
    """
    from .experiment import load_preset

    clusters, pins = load_pool(pool_path, conf_dir)
    preset = load_preset(preset_name, conf_dir)
    task_needs = compute_task_needs(preset)

    health_notes: list = []
    if probe:
        from .cluster_health import apply_health, probe_pool
        need_bedrock = any(n.needs_bedrock for n in task_needs)
        healths = probe_pool(clusters, need_bedrock)
        clusters, health_notes = apply_health(clusters, healths)
        # Early, actionable abort if a needed capability is now dead everywhere.
        if not clusters:
            raise DispatchError(
                "cluster health preflight: every pool cluster is unreachable / "
                "SLURM-down —\n  " + "\n  ".join(health_notes))
        if need_bedrock and not any(c.bedrock for c in clusters):
            raise DispatchError(
                "cluster health preflight: this run needs Bedrock but no cluster "
                "can invoke it right now —\n  " + "\n  ".join(health_notes)
                + "\nFix: have the box operator re-mint the AWS creds on the "
                "box, then rerun. (Pass probe=False / --no-probe to bypass.)")

    plan = plan_split(task_needs, clusters, pins)

    if plan.leftover:
        detail = "; ".join(
            f"task #{i}: {plan.leftover_reasons.get(i, 'unplaceable')}"
            for i in plan.leftover)
        raise DispatchError(
            f"{len(plan.leftover)} task(s) could not be dispatched — {detail}. "
            f"Fix by: raising a cluster budget, reducing a task's model fan-out, "
            f"or splitting a Bedrock+API task into separate stages.")

    subpresets: dict[str, str] = {}
    written: dict[str, Path] = {}
    commands: dict[str, list[str]] = {}
    submitted: dict[str, str] = {}

    for a in plan.assignments:
        if not a.active:
            continue
        name = subpreset_name(preset_name, a.cluster.name)
        subpresets[a.cluster.name] = name
        written[a.cluster.name] = write_subpreset(
            conf_dir, name, render_subpreset(preset, a))
        commands[a.cluster.name] = dispatch_commands(a.cluster, name)
        if submit:
            submitted[a.cluster.name] = _run_dispatch(a.cluster, name, conf_dir)

    return DispatchResult(
        preset_name=preset_name, plan=plan, subpresets=subpresets,
        written=written, commands=commands, submitted=submitted,
        health_notes=health_notes)
