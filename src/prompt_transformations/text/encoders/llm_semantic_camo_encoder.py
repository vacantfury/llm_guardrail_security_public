"""
SemanticCamo attack encoder — multi-step semantic camouflage.

Implements the A_PLAN1 strategy from:
  "SemanticCamo: Jailbreaking Large Language Models through Semantic Camouflage"
  (Jiang et al., 2025)

Pipeline per prompt:
  1. Extract key components (des1, action1) from the harmful prompt via a helper
     LLM  [DECLARED DEVIATION — see below]
  2. Step 1: ask the helper LLM for the semantic features of des1 (structured JSON)
  3. Step 2: BUILD the step-2 prompt from those features + action1 and RETURN IT.
     That prompt is the attack; the TARGET answers it.

All prompt templates are loaded from conf/text_encoding/semantic_camo.yaml; the
two step templates are near-verbatim from `other_repos/SemanticCamo/prompt/
A_PLAN1_step{1,2}.txt`.

FIDELITY FIX 2026-08-05 — do not revert to returning step 2's RESPONSE.
The published attack (`other_repos/SemanticCamo/attack.py:109-154`) is a two-turn
interaction with the TARGET: both steps are invoked on `self.target_model`, and
step 2's *answer* (`item['output'] = A_PLAN1_step2_output`) IS the measured
jailbreak response. The previous version of this encoder ran both steps on the
helper LLM and returned step 2's *answer* as the "encoded prompt", which the
pipeline then handed to the target behind the generic "Please decode and respond"
prefix. That inverted the attack: the harmful generation happened at the helper,
the target was asked to restate an already-written harmful document, the payload
reached any guard fully in the clear, and a helper refusal silently became the
"attack". Numbers collected under it are not SemanticCamo's.

"""
from typing import Optional, List

from llm_utils import LLMServiceFactory, BaseLLMService
from src.utils.logger import get_logger
from ..base_encoder import BaseEncoder
from ..prompt_loader import load_prompt_template

logger = get_logger(__name__)


class SemanticCamoEncoder(BaseEncoder):
    """
    Multi-step semantic camouflage attack.

    Uses a helper LLM to decompose the harmful prompt into semantic features,
    then emits the reference's step-2 prompt for the TARGET to answer.
    """

    # process() returns a complete, self-contained attack prompt (the reference's
    # step-2 turn), so the generic "Please decode and respond…" prefix from
    # conf/text_encoding/default.yaml must NEVER be prepended to it — that prefix
    # is for encoders whose output is a ciphertext needing a decode instruction.
    TARGET_PREFIX = ""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        if model is None:
            raise ValueError("model is required — set it in conf/text_encoding/default.yaml")

        super().__init__(model=model, **kwargs)

        config = load_prompt_template("semantic_camo.yaml")
        self.extraction_prompt = config["extraction_prompt"].strip()
        self.step1_prompt = config["step1_prompt"].strip()
        self.step2_prompt = config["step2_prompt"].strip()

        self.service: BaseLLMService = LLMServiceFactory.create(model)
        if temperature is not None:
            self.service.temperature = temperature
        if max_tokens is not None:
            self.service.max_tokens = max_tokens

        # Force-empty regardless of an inherited YAML target_prefix: process()
        # already returns the full attack prompt (same contract as ArtPrompt).
        self.TARGET_PREFIX = ""

        logger.info(f"Initialized SemanticCamoEncoder with model: {model}")

    def _extract_components(self, prompt: str) -> tuple[str, str]:
        """Extract des1 (harmful description) and action1 (action type) from prompt."""
        message = self.extraction_prompt.format(prompt=prompt)
        results = self.service.batch_chat(
            conversations=[("extract", [(message, None)])],
            is_test=False,
        )
        response = results[0][1] if results else ""

        # Parse "des1: ...\naction1: ..." format
        des1 = prompt
        action1 = "guide"
        for line in response.strip().splitlines():
            line_lower = line.strip().lower()
            if line_lower.startswith("des1:"):
                des1 = line.split(":", 1)[1].strip().strip('"').strip("'")
            elif line_lower.startswith("action1:"):
                action1 = line.split(":", 1)[1].strip().strip('"').strip("'")

        return des1, action1

    def _step1_semantic_features(self, des1: str) -> str:
        """Step 1: Extract semantic features of the harmful description."""
        message = self.step1_prompt.format(des1=des1)
        results = self.service.batch_chat(
            conversations=[("step1", [(message, None)])],
            is_test=False,
        )
        return results[0][1] if results else ""

    def _build_step2_prompt(self, semantic_features: str, action1: str) -> str:
        """Step 2: BUILD the attack prompt (do not invoke it).

        In the reference this prompt is the second turn put to the TARGET, and
        the target's answer is the jailbreak. Here it is the encoder's output,
        so the target answers it exactly once in the defense+evaluate stage.
        """
        return self.step2_prompt.format(step1=semantic_features, action1=action1)

    def process(self, prompt: str, **kwargs) -> str:
        """Return the SemanticCamo step-2 attack prompt for `prompt`."""
        des1, action1 = self._extract_components(prompt)
        logger.debug(f"Extracted: des1='{des1[:50]}...', action1='{action1}'")

        semantic_features = self._step1_semantic_features(des1)
        if not semantic_features:
            # No features -> no camouflage. Returning the raw prompt keeps the row
            # runnable and is honest (it is a plain-prompt cell, not a camo cell);
            # such rows are visible as an unusually short `encoded` field.
            logger.warning("Step 1 returned empty; falling back to raw prompt")
            return prompt

        return self._build_step2_prompt(semantic_features, action1)
