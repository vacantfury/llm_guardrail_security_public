"""
Pydantic schemas for the destroyer-paper pipeline (post-refactor).

Modes:

  Stage 1 — prompt_transform
      Runs a chain of PromptTransformations (encoders + image renderers)
      one-by-one. Each step writes its own subfolder under the task's
      output dir, plus a cumulative results.json carrying the chain's
      full history.

  Stage 2 — defense+evaluate
      Takes a specific subfolder from a prompt_transform output, hands the
      prompts to a Defense (which owns the target-model interaction loop),
      runs the canonical judge, writes raw_results.jsonl + results.json.

  Stage 3 — analyze
      Pure post-processing: fans IN from many defense+evaluate dirs and
      OR-reduces their per-prompt verdicts into derived metrics (e.g.
      portfolio / best-of-all ASR). No model or judge I/O.
"""
from typing import Annotated, Any, List, Literal, Optional, Union
from pydantic import (
    BaseModel, ConfigDict, Discriminator, Field, field_validator, model_validator)

from src.utils.provenance import SCHEMA_VERSION


# ====================================================================
# Inter-stage Prompt records
# ====================================================================


class RawPrompt(BaseModel):
    """Source prompt loaded from harmful/benign dataset JSONL."""
    id: str
    category: str
    source: str           # "advbench" | "harmbench" | "jailbreakbench"
    prompt: str


class Prompt(BaseModel):
    """Carrier across prompt_transform steps.

    `original` is the raw harmful query — never modified by the chain.
    Judges read it as ground-truth intent. `encoded` is the current text
    after the chain so far. `image_encoded` is set by image transformations.

    `image_original` / `encoding` are kept for back-compat with older
    prompts.jsonl files; the new pipeline doesn't write `image_original`
    and treats `encoding` as a free-text label (chain executor sets it to
    the last text transformation's type_name)."""
    id: str
    encoding: str = ""
    original: str
    encoded: str
    image_original: Optional[str] = None
    # List of rendered-image paths (relative to the prompt_transform step dir).
    # The paginated renderer emits one path per page; short prompts → 1 path.
    image_encoded: Optional[List[str]] = None

    @field_validator("image_encoded", mode="before")
    @classmethod
    def _coerce_image_encoded(cls, v):
        """Back-compat: archived prompts.jsonl stored `image_encoded` as a
        single path string; the paginated renderer now emits a list. Coerce a
        bare string → 1-element list so old data still loads."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return list(v)


class EvaluationRow(BaseModel):
    """One row in raw_results.jsonl (long format)."""
    id: str
    prompt_stage: str
    response: str
    asr: Optional[bool] = None
    # Multi-image overflow bookkeeping (paginated renderer). `num_images` is the
    # page count of the rendered prompt; `is_within_maxlen` is False when that
    # exceeds MAX_PAGES_PER_PROMPT, in which case the prompt is EXCLUDED from the
    # target query and from ASR (asr stays None) rather than silently truncated.
    num_images: int = 0
    is_within_maxlen: bool = True
    # False when the target call did not produce a valid model output (mechanism
    # error: context-overflow at query time, network/timeout, rate-limit
    # exhaustion, failed batch item). Such rows are EXCLUDED from the query's ASR
    # denominator — an untestable prompt must not count as a defeated attack.
    # Distinct from a refusal, which is a successful call (is_correctly_processed
    # stays True). Distinct from is_within_maxlen, which is a render-time page gate.
    is_correctly_processed: bool = True
    judge_output: Optional[str] = None
    judge_reasoning: Optional[str] = None
    judge_raw_response: Optional[str] = None
    refusal: Optional[bool] = None
    refusal_judge_output: Optional[str] = None
    refusal_judge_reasoning: Optional[str] = None
    refusal_judge_raw_response: Optional[str] = None


class Judgment(BaseModel):
    """ASR judgment for a single response."""
    id: str
    model: str
    encoding: str
    prompt_stage: str
    judge_output: str
    judge_reasoning: Optional[str] = None
    is_jailbroken: bool


# ====================================================================
# prompt_transform results
# ====================================================================


class PromptTransformStepResult(BaseModel):
    """A single step record inside the chain's results_history list."""
    config: dict                       # transformation's resolved kwargs
    metrics: dict = Field(default_factory=dict)
    prompts_file: str                  # relative or absolute path to prompts.jsonl
    images_dir: Optional[str] = None
    is_multimodal_after: bool
    elapsed_seconds: float
    usage: Optional[dict] = None


class PromptTransformResult(BaseModel):
    """Top-level results.json for every step subfolder of a prompt_transform task.

    Each step subfolder writes its own copy of this object, with
    `results_history` truncated to that step. The final step's results.json
    is the chain's complete record.
    """
    mode: Literal["prompt_transform"] = "prompt_transform"
    task_id: str
    benchmark: str
    prompt_range: Optional[list[int]] = None
    dataset_path: str

    is_multimodal: bool                # cumulative state AFTER latest step
    total_steps: int
    latest_step: str                   # type_name of the most recent step
    latest_step_dir: str               # relative path under task_dir
    task_dir: str
    timestamp_last_step: str           # ISO-8601
    elapsed_seconds_total: float
    count: int
    mlflow_run_id: Optional[str] = None

    # Provenance — see src/utils/provenance.py.
    schema_version: int = SCHEMA_VERSION
    git_sha: str = ""
    git_dirty: bool = False
    status: str = "success"

    # Configured plan — full chain including any upstream steps, in order.
    transformation_list: list[str]

    # Per-step log — only steps run THIS task. Steps from an upstream chain
    # (when source_transform_subdir was set) live in `upstream` instead.
    results_history: list[dict[str, PromptTransformStepResult]]

    # Provenance to the step this task chained from, if any. `upstream_ref`
    # ({source_dir, results_sha256}) is the current form — a pointer, not an
    # embedded copy (embedding bloated O(chain-depth) and could drift from its
    # source). `upstream` is legacy (old data embedded the full dict); new
    # writes leave it None and set `upstream_ref`.
    upstream: Optional[dict] = None
    upstream_ref: Optional[dict] = None


# ====================================================================
# defense+evaluate results
# ====================================================================


class TargetModelConfig(BaseModel):
    """Resolved LLM inference params for the target model."""
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    stop_sequences: list[str]
    seed: int
    n_completions: int
    stream: bool


class DefenseEvaluateResult(BaseModel):
    """Complete record of a defense+evaluate task — written to results.json."""
    mode: Literal["defense+evaluate"] = "defense+evaluate"
    campaign: Optional[str] = None       # optional run-campaign label, copied from the task

    # Provenance — which prompt_transform step subfolder this consumed.
    source_transform_subdir: str
    upstream: Optional[dict] = None      # legacy embedded copy (old data only)
    upstream_ref: Optional[dict] = None  # {source_dir, results_sha256} pointer — current form
    is_multimodal: bool              # copied from upstream's PromptTransformResult

    # Defense.
    defense: str
    defense_config: dict = Field(default_factory=dict)

    # Target model.
    target_model: str
    target_model_config: TargetModelConfig
    model_family: Optional[str] = None        # first-class, from the registry
    alignment_tier: Optional[str] = None      # first-class, from the registry
    system_message: Optional[str] = None

    # Benchmark + encoding (inferred or override).
    benchmark: str
    encoding: Optional[str] = None
    transformation_list: list[str] = Field(default_factory=list)
    # First-class decoy/companion image path — no more digging into
    # upstream.results_history[i].<step>.config.image_path to recover it.
    companion_image_path: Optional[str] = None
    prompt_range: Optional[list[int]] = None

    # Judging.
    judge_method: str
    judge_model: Optional[str] = None

    # Results.
    count: int
    asr: Optional[float] = None          # HarmBench-style ASR only — never overloaded
    refusal_rate: Optional[float] = None
    # Metric contract: every named scalar metric lives in `metrics`, and
    # `primary_metric` names which key is this benchmark's headline number
    # (e.g. "attack_success_rate", "refusal_rate", "direct_answer_rate") — so a
    # reader never has to know that, say, OR-Bench's headline hides in `asr`.
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric: Optional[str] = None
    eval_stats: Optional[dict[str, dict]] = None

    # Usage.
    target_usage: dict
    defense_usage: Optional[dict] = None

    elapsed_seconds: float
    output_dir: str
    mlflow_run_id: Optional[str] = None

    # Provenance — see src/utils/provenance.py.
    schema_version: int = SCHEMA_VERSION
    git_sha: str = ""
    git_dirty: bool = False
    judge_config_hash: Optional[str] = None
    status: str = "success"
    warnings: list[str] = Field(default_factory=list)


# ====================================================================
# analyze results — pure post-processing over already-run results
# ====================================================================


class AnalyzeGroupResult(BaseModel):
    """One (group_by) cell of an analyze run — e.g. one (model, defense).

    All rates are percentages over `denominator`, so portfolio / best-fixed /
    per-attack are directly comparable. `complementarity_gap` =
    portfolio_asr - best_fixed_attack_asr (how much the *suite* buys over the
    single strongest attack)."""
    group_key: dict[str, str]
    n_attacks: int                          # distinct attack variants joined
    attacks: list[str]
    denominator: int
    n_broken: int                           # ids broken by >= 1 attack
    n_excluded_everywhere: int              # ids with asr=None under every attack
    portfolio_asr: float                    # best-of-all (OR over attacks), %
    best_fixed_attack_asr: float            # max single-attack ASR, %
    complementarity_gap: float              # portfolio - best_fixed, pp
    per_attack_asr: dict[str, float]        # attack label -> ASR % (same denom)


class AnalyzeResult(BaseModel):
    """Top-level results.json for an `analyze` task — written to outputs/analyze/.

    No model/judge I/O: a pure OR-reduction over per-prompt `asr` verdicts in
    each source dir's raw_results.jsonl, joined on `join_key` (`id`). Per-id
    attribution (which attack broke each prompt) lives alongside in
    portfolio_rows.jsonl — that file IS the coverage map.
    """
    mode: Literal["analyze"] = "analyze"
    analysis: str

    # Provenance — the full list of consumed source dirs (analyze is the only
    # mode that fans IN from many upstreams), each as the standard
    # {source_dir, results_sha256} pointer.
    source_dirs: list[str]
    upstream_refs: list[dict] = Field(default_factory=list)

    group_by: list[str]
    join_key: str
    attack_subset: Optional[list[str]] = None
    denominator_policy: str
    benchmark: Optional[str] = None

    n_sources: int
    n_groups: int
    groups: list[AnalyzeGroupResult]

    metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric: Optional[str] = None
    count: int

    elapsed_seconds: float
    output_dir: str
    mlflow_run_id: Optional[str] = None

    # Provenance — see src/utils/provenance.py.
    schema_version: int = SCHEMA_VERSION
    git_sha: str = ""
    git_dirty: bool = False
    status: str = "success"
    warnings: list[str] = Field(default_factory=list)


class RejudgeResult(BaseModel):
    """results.json for a `rejudge` run — a fresh judging of a prior
    defense+evaluate dir's stored responses under a NEW judge, without
    re-querying the target. The source dir and its original verdicts are left
    untouched, so a judge switch never clobbers the historical numbers.

    Field names mirror DefenseEvaluateResult where they overlap so the existing
    results-doc / analysis tooling reads a rejudge dir the same way.
    """
    model_config = {"extra": "allow"}   # absorb provenance_fields() (git_sha, timestamp, …)

    mode: Literal["rejudge"] = "rejudge"
    rejudge_of: str                       # the source defense+evaluate dir
    upstream_ref: Optional[dict] = None   # {source_dir, results_sha256} pointer
    campaign: Optional[str] = None
    # carried through from the source result so downstream tables/joins keep working
    target_model: Optional[str] = None
    defense: Optional[str] = None
    # Inherited verbatim from the source cell. A rejudge never queries the target, so these
    # describe the run that PRODUCED the responses, not this one. They are carried because a
    # rejudge dir is otherwise unable to answer "at what temperature / against which guard?" —
    # which made temperature unrecoverable without walking upstream_ref and reading MLflow
    # (cost two separate verifications on 2026-08-08). Optional so old dirs still parse.
    target_model_config: Optional[dict] = None
    defense_config: dict = Field(default_factory=dict)
    is_multimodal: bool = False
    benchmark: Optional[str] = None
    encoding: Optional[str] = None
    transformation_list: list[str] = Field(default_factory=list)
    prompt_range: Optional[list[int]] = None
    # the fresh judging under the new judge
    judge_model: Optional[str] = None
    judge_method: Optional[str] = None
    count: int = 0
    asr: Optional[float] = None
    refusal_rate: Optional[float] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric: Optional[str] = None
    eval_stats: Optional[dict] = None
    judge_config_hash: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    output_dir: Optional[str] = None
    status: str = "success"
    warnings: list[str] = Field(default_factory=list)


# ====================================================================
# Task configs — what experiment.yaml's `tasks:` list contains
# ====================================================================


class _TaskBase(BaseModel):
    """Fields shared by every task mode."""
    benchmark: Optional[str] = None
    prompt_range: Optional[list[int]] = None
    campaign: Optional[str] = None      # optional run-campaign label (e.g. "paper_c_round1") — indexed by scripts/run_index.py, logged as an MLflow param


class TransformationSpec(BaseModel):
    """One entry in PromptTransformTask.transformation_list.

    `extra="forbid"` is LOAD-BEARING, not tidiness. Renderer arguments must be
    nested under `params:`; written as siblings of `type:` they used to be
    silently DROPPED, and the step then ran on its renderer defaults with no
    error anywhere. That cost a real mislabel: `presence_controls_stage1.yaml`
    wrote `image_path: .../mountain.png` as a sibling, so the run rendered the
    DEFAULT rabbit instead, and the paper reported that arm as the mountain
    decoy for a day (caught 2026-08-07 by checking rendered image dimensions
    against the config). Forbidding extras turns that class of typo into a
    startup ValidationError.
    """
    model_config = ConfigDict(extra="forbid")

    type: str                                  # transformation type_name
    params: dict[str, Any] = Field(default_factory=dict)


class PromptTransformTask(_TaskBase):
    """Stage 1: run a chain of PromptTransformations.

    Each step in `transformation_list` runs in order. The chain executor
    creates one subfolder per step (named by type_name) and writes a
    cumulative results.json into every subfolder.

    Either `source_file` (raw dataset) OR `source_transform_subdir` (a
    prior prompt_transform step) supplies the input prompts. Chaining from
    a prior step lets multiple downstream chains share an identical text
    encoding — critical for paired ablations where encoder-LLM
    nondeterminism would otherwise confound results.
    """
    mode: Literal["prompt_transform"]
    source_file: Optional[str] = None
    source_transform_subdir: Optional[str] = None
    transformation_list: list[TransformationSpec]

    @model_validator(mode="after")
    def _validate(self):
        if not self.transformation_list:
            raise ValueError("transformation_list must contain at least one step")
        if (self.source_file is None) == (self.source_transform_subdir is None):
            raise ValueError(
                "exactly one of source_file (raw dataset) or "
                "source_transform_subdir (prior step) must be set")
        return self


class DefenseEvaluateTask(_TaskBase):
    """Stage 2: defense + target-model query + judging, combined.

    `source_transform_subdir` points at a specific step subfolder of a
    prompt_transform run. The Defense reads that step's prompts.jsonl, runs
    its query-loop against the target model, and the result is judged with
    the canonical evaluator for the benchmark.
    """
    mode: Literal["defense+evaluate"]
    source_transform_subdir: str
    defense: str = "no_defense"
    defense_config: dict[str, Any] = Field(default_factory=dict)
    target_model: str
    target_model_config: dict[str, Any] = Field(default_factory=dict)
    judge_model: Optional[str] = None  # per-task judge LLM override; lets parallel papers use different judges (else conf/evaluation default)
    system_message: Optional[str] = None
    judge_method: Optional[str] = None    # override; usually None (canonical)


class AnalyzeTask(_TaskBase):
    """Post-processing mode: aggregate per-prompt verdicts across already-run
    `defense+evaluate` output dirs into derived metrics. Pure data processing —
    no target/judge calls, no cluster serving (the orchestrator classifies it
    NO_MODEL automatically).

    `source_dirs` are `defense+evaluate` output dirs (each holding results.json
    + raw_results.jsonl); glob patterns are expanded. Results are grouped by the
    results.json fields in `group_by` (default model × defense); within a group
    the per-`id` `asr` verdicts are OR-reduced across the distinct attack
    variants (the "best-of-all" / portfolio metric).
    """
    mode: Literal["analyze"]
    analysis: str = "portfolio_asr"
    source_dirs: list[str]
    # results.json field names to group within before OR-reducing over attacks.
    group_by: list[str] = Field(default_factory=lambda: ["target_model", "defense"])
    join_key: str = "id"
    # Restrict the best-of to these attack labels (None = all variants present).
    attack_subset: Optional[list[str]] = None
    # full_source_set: denom = union of ids in the group (excluded-everywhere
    # prompts count as not-broken — deployment-realistic). evaluated_union:
    # denom = ids evaluated (asr != None) by >= 1 attack.
    denominator_policy: Literal["full_source_set", "evaluated_union"] = "full_source_set"

    @model_validator(mode="after")
    def _validate(self):
        if not self.source_dirs:
            raise ValueError("source_dirs must contain at least one defense+evaluate dir or glob")
        return self


class RejudgeTask(_TaskBase):
    """Re-judge mode: re-score a stored `defense+evaluate` dir's saved responses
    with a different judge, WITHOUT re-querying the target model.

    Reads `responses_from`'s raw_results.jsonl (the target responses a prior
    defense+evaluate run saved), runs ONLY the judging step with `judge_model`
    (+ optional `judge_method` override; else the benchmark's canonical
    evaluator), and writes a NEW result dir under `outputs/<paper>/rejudge/`.
    Used to move the coverage grid from one judge to another (e.g.
    gpt-5-nano -> gpt-5-mini, or a WildGuard robustness pass) with zero target
    regeneration. Judges every within-budget, correctly-processed row.
    """
    mode: Literal["rejudge"]
    responses_from: str          # a stored defense+evaluate output dir
    judge_model: str             # judge LLM to apply (e.g. gpt-5-mini, wildguard)
    judge_method: Optional[str] = None  # evaluator override; else benchmark->evaluator
    base_dir: Optional[str] = None


class AdaptiveAttackTask(_TaskBase):
    """Unlike the hand-built adaptive attacks (channel splits, one-shot
    decode-evasion), this runs a query BUDGET per behavior: it queries the full
    `defense` pipeline, reads whether the response was a refusal/guard-block, and
    has an attacker LLM rewrite the request to steer around the observed failure,
    repeating up to `rounds` times. The FINAL kept response per behavior is judged
    by the project judge exactly like a defense+evaluate cell, so the reported ASR
    is comparable to every other number in the paper.

    Text-channel attack: candidates are plain text, so `is_multimodal` is False
    and the pipeline's recover step is a no-op — the attack stresses the
    decode+guard mechanism, which is the text-side defense. Reported as such.
    """
    mode: Literal["adaptive_attack"]
    source_file: str             # raw dataset JSONL of harmful behaviors
    target_model: str
    defense: str = "modality_complete"
    defense_config: dict[str, Any] = Field(default_factory=dict)
    attacker_model: str          # the LLM that rewrites candidates each round
    benchmark: Optional[str] = None
    prompt_range: Optional[list[int]] = None
    judge_model: Optional[str] = None
    judge_method: Optional[str] = None
    # adaptive-loop knobs; merged over conf/adaptive_attack/default.yaml
    attack_config: dict[str, Any] = Field(default_factory=dict)


# Discriminated union: Pydantic dispatches by `mode`.
TaskConfig = Annotated[
    Union[PromptTransformTask, DefenseEvaluateTask, AnalyzeTask, RejudgeTask,
          AdaptiveAttackTask],
    Discriminator("mode"),
]


class ExperimentPreset(BaseModel):
    """Top-level shape of conf/experiment/*.yaml."""
    num_main_job_threads: int = 5
    num_cluster_jobs: int = 8
    tasks: list[TaskConfig]


# ====================================================================
# LLM / Cluster configs — what conf/llm/*.yaml contains.
# Unchanged from the pre-refactor version; consumed by LLMServiceFactory.
# ====================================================================


class ModelConfig(BaseModel):
    """LLM request parameters."""
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    stop_sequences: list[str]
    seed: int
    n_completions: int
    stream: bool
    max_concurrency: int
    max_retries: int
    batch_poll_interval: int
    batch_timeout: int


class ClusterConfig(BaseModel):
    """Cluster/vLLM server parameters."""
    partition: str
    excluded_nodes: list[str]
    gpu_types_excluded: list[str]
    gpu_constraint: Optional[str] = None
    num_gpus: int
    num_instances: int
    cpus_per_task: int
    mem_gb: int
    time_limit: str
    port: int
    gpu_memory_utilization: float
    max_model_len: int
    dtype: Optional[str] = None
    chat_template: Optional[str] = None
    trust_remote_code: bool = False
    server_start_timeout: int
    endpoint_wait_timeout: int
    cluster_server_endpoint_timeout: int
    monitor_poll_interval: int
    health_check_timeout: int
    slurm_cmd_timeout: int
    conda_env: Optional[str] = None   # None when env_setup replaces conda (AICR / nurc_uv)
    cuda_module: str
    hf_home: str
    health_recheck_interval: int = 60
    gpu_request_style: str = "gres"
    # Shell lines that set up the Python env in the sbatch, REPLACING the default
    # `module load anaconda3 <cuda> ; source activate <conda_env>` pair. AICR/nurc_uv use a uv venv.
    # None -> the conda default.
    env_setup: Optional[list[str]] = None
    # NURC compute nodes: offline HF (historical); AICR sets false (compute-node internet).
    hf_offline: bool = True
    # Dispatch + QOS limits (read by multi_cluster.load_pool and the job-budget caps).
    sbatch: Optional[str] = None                      # orchestrator wrapper for this cluster
    budget: Optional[int] = None                      # max concurrent vLLM servers
    max_submit: int = 8                               # QOS MaxSubmitPU (running + pending)
    max_slurm_time_limit: Optional[str] = None        # gpu-partition QOS wall cap (clamps time_limit)


def _arch_row_for(model_str: str):
    """Resolve a conf/llm model string to a registry row for ARCHITECTURE
    checks (context ceiling, chat template). llm_utils v5's `from_string`
    refuses an id registered on several serving routes (today only
    meta-llama/Meta-Llama-3-8B-Instruct, local + cluster) — but route twins
    share the same weights, so architecture facts are identical across them;
    any matching row serves. Prefer the cluster row, since conf/llm/*.yaml
    configure cluster serving. Returns None when the string is not in the
    registry at all (fine-tuned checkpoints etc. — validators skip)."""
    from llm_utils.llm_model import LLMModel, Provider
    try:
        return LLMModel.from_string(model_str)
    except ValueError:
        matches = [m for m in LLMModel if m.model_id == model_str]
        if not matches:
            return None
        cluster = [m for m in matches if m.provider is Provider.SLURM_CLUSTER]
        return (cluster or matches)[0]


class LLMConfig(BaseModel):
    """Top-level LLM config — SCHEMA DOCUMENTATION ONLY, never instantiated.

    ⚠️ Nothing constructs this model. `Experiment._load_cluster_config` returns
    a plain merged dict and passes it straight to `start_server`, so a validator
    written here DOES NOT RUN. Two once lived here and had never fired (found
    2026-08-02):
      - the max_model_len architectural-ceiling check → moved to the live path
        as `Experiment._arch_ceiling_preflight`;
      - the chat-template file-existence check → deleted as redundant, since
        llm_utils v5.1.0 performs it upfront for every instance before the
        first sbatch (family cluster-job standard §5 T1).

    Keep this class as the readable shape of conf/llm/<model>.yaml. Put new
    cluster-config CHECKS in `Experiment`'s preflights, not here.
    """
    model: ModelConfig
    cluster: ClusterConfig


# ====================================================================
# Evaluation config — judge LLM settings shared by every evaluator.
# ====================================================================


class JudgeLLMConfig(BaseModel):
    """Judge LLM service config."""
    model: str = "gpt-5-nano"
    max_tokens: int = 16384
    temperature: float = 0.0
    # None = llm_utils auto-routing (batch iff worst-case cost >= $1); False =
    # always realtime. conf/evaluation/default.yaml pins this False — see the
    # comment there for the 2026-08-07 partial_judge incident it prevents.
    use_batch_api: Optional[bool] = None


class EvaluationConfig(BaseModel):
    """Top-level evaluation config."""
    judge_llm_config: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)
    judge_method: Optional[str] = None
