from typing import List

from .base_evaluator import BaseEvaluator
from .harmbench_evaluation.evaluator import HarmBenchEvaluator
from .jailbreakbench_evaluation.evaluator import JailbreakBenchEvaluator
from .jailbreakbench_refusal_evaluation.evaluator import JailbreakBenchRefusalEvaluator
from .orbench_evaluation.evaluator import ORBenchEvaluator
from .wildguard_evaluation.evaluator import WildGuardEvaluator


# Benchmark → canonical evaluator(s). For most benchmarks this is a single
# evaluator; `jailbreakbench` (harmful split) gets BOTH the harmful judge
# (→ ASR) and the refusal judge (→ refusal_rate) so a single evaluate task
# produces both metrics in one results.json. Other JBB-style data does not
# need both: jailbreakbench_benign only meaningfully supports refusal_rate
# (ASR on benign prompts is methodologically vacuous), and harmbench /
# orbench each have a single canonical judge.
#
# Why a benchmark-keyed entry point exists alongside the older method-keyed one:
# Dataset name is canonical — it lives in the `source_dir` path component and
# in `_infer_benchmark()`. Letting callers pass a free-form `judge_method`
# invites mismatches (e.g. running HarmBench's classifier on JBB prompts
# because a YAML wasn't updated). `create_from_benchmark` removes that footgun.


class EvaluatorFactory:
    """
    Factory for creating canonical benchmark evaluators.

    The judge model is CALLER-INJECTED: task.py::_resolve_evaluators passes
    `model=judge_model` (resolved from conf/evaluation/default.yaml, or the
    task's judge override) into every evaluator constructor. Evaluators do not
    hard-bind a judge — the explicit-model contract (see
    src/evaluation/constants.py) is what lets the rejudge mode re-score saved
    responses under a different judge without re-querying the target.

    Prefer `create_from_benchmark(benchmark)` — it returns the full canonical
    set of evaluators for that dataset (one or two), driven by `_infer_benchmark`
    on the source_dir path. `create(method=...)` is retained as an explicit
    single-evaluator override (legacy/ad-hoc use only).
    """

    @staticmethod
    def create_from_benchmark(benchmark: str, **kwargs) -> List[BaseEvaluator]:
        """Resolve a benchmark name to its canonical evaluator list.

        Returns a list because `jailbreakbench` runs two judges in one task
        (harmful → ASR, refusal → refusal_rate). Both verdicts land in the
        same raw_results.jsonl as parallel columns, and both metrics in the
        same results.json. Every other benchmark returns a single-element
        list.

        Args:
            benchmark: Dataset slug as returned by `_infer_benchmark` (e.g.
                'harmbench', 'jailbreakbench', 'jailbreakbench_benign',
                'orbench_harmful'). Case-insensitive; whitespace trimmed.
            **kwargs: Forwarded to each evaluator's constructor.

        Returns:
            List of BaseEvaluator subclass instances.

        Raises:
            ValueError: If `benchmark` is not in the supported set.
        """
        bench = benchmark.lower().strip()

        if bench == "harmbench":
            return [HarmBenchEvaluator(**kwargs)]

        if bench == "jailbreakbench":
            return [
                JailbreakBenchEvaluator(**kwargs),
                JailbreakBenchRefusalEvaluator(**kwargs),
            ]

        if bench == "jailbreakbench_benign":
            return [JailbreakBenchRefusalEvaluator(**kwargs)]

        if bench.startswith("orbench"):
            return [ORBenchEvaluator(**kwargs)]

        raise ValueError(
            f"No canonical evaluator for benchmark={benchmark!r}. "
            f"Supported: 'harmbench', 'jailbreakbench', "
            f"'jailbreakbench_benign', 'orbench_*'.")

    @staticmethod
    def create(method: str = "harmbench", **kwargs) -> BaseEvaluator:
        """Explicit method-based factory (emergency override / ad-hoc use).

        Prefer `create_from_benchmark` — it derives the evaluator from the
        canonical source_dir path and prevents dataset/judge mismatches.

        Args:
            method: Evaluation method ('harmbench', 'jailbreakbench', 'jbb_refusal', 'orbench').
            **kwargs: Forwarded to the evaluator's constructor (including the
                explicit `model=` judge — see the class docstring).

        Returns:
            An instance of a class inheriting from BaseEvaluator.

        Raises:
            ValueError: If the method is unknown.
        """
        method = method.lower().strip()

        if method == "harmbench":
            return HarmBenchEvaluator(**kwargs)
        if method in ("jailbreakbench", "jbb"):
            return JailbreakBenchEvaluator(**kwargs)
        if method in ("jbb_refusal", "refusal"):
            return JailbreakBenchRefusalEvaluator(**kwargs)
        if method == "orbench":
            return ORBenchEvaluator(**kwargs)
        if method == "wildguard":
            # Response-harm own-policy judge (rejudge panel/robustness view).
            # Not in create_from_benchmark: it is a judge_method override, never
            # a benchmark's canonical judge.
            return WildGuardEvaluator(**kwargs)

        raise ValueError(
            f"Unknown evaluation method: {method}. "
            "Supported: 'harmbench', 'jailbreakbench', 'jbb_refusal', 'orbench', 'wildguard'"
        )


# Evaluator classes that report refusal_rate (rather than attack_success_rate).
# Used by task.py to pick the right metric key after an evaluator is resolved,
# instead of string-matching on judge_method.
REFUSAL_RATE_EVALUATORS = (JailbreakBenchRefusalEvaluator, ORBenchEvaluator)

# Judge methods whose evaluator is a SELF-CONTAINED classifier: it scores on its
# own and never calls the judge LLM, so the method name IS the effective judge.
# Recording the config-default judge LLM for these (as task.py once did) is false
# provenance — e.g. claiming gpt-5-nano judged a run that wildguard actually scored.
SELF_CONTAINED_JUDGE_METHODS = frozenset({"wildguard"})


# Benchmark → list of judge_method strings (the inverse of how task.py looks
# up canonical evaluators). Each method name maps via the evaluator package's
# constants module to an LLMModel. Used by the orchestrator to pre-discover
# cluster-hosted judges *without* instantiating evaluators (instantiation
# would touch the LLMServiceFactory and require a server_manager).
_BENCHMARK_TO_JUDGE_METHODS = {
    "harmbench": ["harmbench"],
    "jailbreakbench": ["jailbreakbench", "refusal"],   # dual judges
    "jailbreakbench_benign": ["refusal"],
}


def judge_methods_for_benchmark(benchmark: str) -> list[str]:
    """List the canonical judge_method names a given benchmark uses.

    Returns names compatible with `_judge_model_for_method` in experiment.py
    (which resolves each method's judge LLM from
    conf/evaluation/default.yaml::judge_llm_config, or the task override —
    there is no per-evaluator JUDGE_MODEL constant).

    Raises:
        ValueError: If the benchmark is unknown.
    """
    bench = benchmark.lower().strip()
    if bench in _BENCHMARK_TO_JUDGE_METHODS:
        return list(_BENCHMARK_TO_JUDGE_METHODS[bench])
    if bench.startswith("orbench"):
        return ["orbench"]
    raise ValueError(
        f"No canonical judge mapping for benchmark={benchmark!r}.")
