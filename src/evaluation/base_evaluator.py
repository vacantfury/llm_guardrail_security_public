from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from llm_utils import BaseLLMService, LLMModel, LLMServiceFactory


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators.

    Concrete evaluators (HarmBench, JailbreakBench, ORBench, JBB-Refusal) take
    the judge model as an EXPLICIT constructor argument — there is no hard-bound
    per-evaluator judge constant. In production, task.py::_resolve_evaluators
    injects the project judge (conf/evaluation/default.yaml, currently
    gpt-5-mini) via EvaluatorFactory.create_from_benchmark(benchmark,
    model=judge_model, ...). This caller-injection is deliberate: it is what
    lets the rejudge mode swap judges over saved responses without re-querying
    the target. Judge choice is a methodology decision made at the config
    layer, not inside evaluator classes.
    """

    def __init__(self, model: Optional[LLMModel] = None, **kwargs):
        """
        Args:
            model: the judge LLM (caller-injected — see class docstring).
            **kwargs: forwarded to LLMServiceFactory.create (max_tokens,
                temperature, max_concurrency, etc.).
        """
        self.model = model
        self.kwargs = kwargs                 # legacy attribute, kept as-is
        self.service_kwargs = kwargs
        self._service: Optional[BaseLLMService] = None

    @property
    def service(self) -> BaseLLMService:
        """Lazy judge service. Temperature defaults to 0.0 (deterministic
        classification) unless the caller overrides it. Hoisted from the four
        concrete evaluators at the 2026-07-24 audit — behavior unchanged."""
        if self._service is None:
            kwargs = dict(self.service_kwargs)
            kwargs.setdefault("temperature", 0.0)
            self._service = LLMServiceFactory.create(self.model, **kwargs)
        return self._service

    @abstractmethod
    def evaluate(
        self,
        prompts: List[Dict[str, Any]],
        processed_prompts: List[str],
        responses: Dict[str, str],
        baseline_responses: Optional[Dict[str, str]] = None,
        verbose: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate the effectiveness of a jailbreak attack on pre-generated responses.

        Args:
            prompts: List of original prompt dictionaries (must contain 'prompt' key).
            processed_prompts: List of processed/jailbroken prompt strings.
            responses: Dict mapping prompt_id -> target model response text.
            baseline_responses: Optional dict mapping prompt_id -> baseline response text.
            verbose: Whether to log detailed progress.

        Returns:
            Tuple of:
                - detailed_df: DataFrame with per-prompt results
                - statistics: Dict with aggregate metrics
        """
        pass
