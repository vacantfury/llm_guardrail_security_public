"""Stochastic paraphrase encoder — the "which-paraphrase" variance knob (Paper D).

Stochasticity is the whole point: ``temperature`` MUST be > 0 (default 1.0) so the
N draws of one behavior differ. The chat services expose no per-call seed, so a
run is reproducible only up to sampling temperature — acceptable for a budget
attack whose power comes from diversity, not from a fixed draw sequence.

Refusal caveat (proposal §6): an aligned paraphraser may refuse to reword overtly
harmful text. For real runs set ``model`` to an OPEN, cluster-served model; the
prompt is framed as a neutral rewriting task to minimise refusals, and any
refusal / empty output falls back to the original text (a no-op draw).

Prompt template: conf/text_encoding/paraphrase.yaml.
"""
from typing import List, Optional

from llm_utils import BaseLLMService, LLMServiceFactory
from src.utils.logger import get_logger
from ..base_encoder import BaseEncoder
from ..prompt_loader import load_prompt_template

logger = get_logger(__name__)


class ParaphraseLLMEncoder(BaseEncoder):
    """Meaning-preserving stochastic paraphrase (temperature-driven diversity)."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        if model is None:
            raise ValueError(
                "model is required — set it in the preset or "
                "conf/text_encoding/default.yaml")
        super().__init__(model=model, **kwargs)

        config = load_prompt_template("paraphrase.yaml")
        self.paraphrase_prompt = config["paraphrase_prompt"].strip()

        self.service: BaseLLMService = LLMServiceFactory.create(model)
        # Stochasticity is load-bearing: at temperature 0 every draw of a
        # behavior is identical and the channel has zero variance.
        self.service.temperature = 1.0 if temperature is None else float(temperature)
        if max_tokens is not None:
            self.service.max_tokens = max_tokens

        logger.info(
            f"Initialized ParaphraseLLMEncoder (model={model}, "
            f"temperature={self.service.temperature})")

    def _batch_process_core(self, prompts: List[str], **kwargs) -> List[str]:
        """One batched LLM call: N (possibly identical) rows -> N paraphrases."""
        conversations = [
            (str(i), [(self.paraphrase_prompt.format(prompt=p), None)])
            for i, p in enumerate(prompts)
        ]
        results = self.service.batch_chat(conversations=conversations, is_test=False)
        by_id = {cid: (text or "").strip() for cid, text in results}
        # Refusal / empty -> fall back to the original request (a no-op draw).
        return [by_id.get(str(i)) or original for i, original in enumerate(prompts)]

    def process(self, prompt: str, **kwargs) -> str:
        """Single-prompt paraphrase (sequential fallback path)."""
        return self._batch_process_core([prompt])[0]
