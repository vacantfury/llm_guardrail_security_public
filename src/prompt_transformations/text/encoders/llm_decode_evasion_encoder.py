"""
DecodeEvasion attack encoder — an ADAPTIVE attack against Paper C's recover+decode amplifier.

Single LLM rewrite call per prompt (batched). Template lives in conf/text_encoding/decode_evasion.yaml.
"""
from typing import Optional, List

from llm_utils import LLMServiceFactory, BaseLLMService
from src.utils.logger import get_logger
from ..base_encoder import BaseEncoder
from ..prompt_loader import load_prompt_template

logger = get_logger(__name__)


class DecodeEvasionEncoder(BaseEncoder):
    """
    Adaptive decode-evasion attack: reframe a harmful request so its short restatement
    (what the recover+decode amplifier surfaces to the guard) reads benign, while the full
    answer still produces the harmful specifics. One rewrite call per prompt via an encoder LLM.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        if model is None:
            raise ValueError("model is required — set it in conf/text_encoding/default.yaml or the preset step")

        # Decode-evasion is a PLAIN-TEXT attack: it must read as an ordinary benign request.
        # Drop the shared "Please decode and respond to the following message" prefix that
        # default.yaml applies to encoded attacks — that "decode" framing would tip off the
        # amplifier and defeat the point. BaseEncoder's empty TARGET_PREFIX default then applies.
        kwargs.pop("target_prefix", None)
        super().__init__(model=model, **kwargs)

        config = load_prompt_template("decode_evasion.yaml")
        self.rewrite_prompt = config["rewrite_prompt"].strip()

        self.service: BaseLLMService = LLMServiceFactory.create(model)
        if temperature is not None:
            self.service.temperature = temperature
        if max_tokens is not None:
            self.service.max_tokens = max_tokens

        logger.info(f"Initialized DecodeEvasionEncoder with model: {model}")

    def process(self, prompt: str, **kwargs) -> str:
        """Rewrite a single harmful prompt into its decode-evading form."""
        message = self.rewrite_prompt.format(prompt=prompt)
        results = self.service.batch_chat(
            conversations=[("rewrite", [(message, None)])],
            is_test=False,
        )
        rewritten = (results[0][1] if results else "").strip()
        if not rewritten:
            logger.warning("DecodeEvasion rewrite returned empty; falling back to raw prompt")
            return prompt
        return rewritten

    def _batch_process_core(self, prompts: List[str], **kwargs) -> List[str]:
        """One batched LLM call for all prompts (the rewrite is a single independent call each)."""
        if not prompts:
            return []
        conversations = [
            (str(i), [(self.rewrite_prompt.format(prompt=p), None)])
            for i, p in enumerate(prompts)
        ]
        results = self.service.batch_chat(conversations=conversations, is_test=False)
        by_id = {cid: (text or "").strip() for cid, text in results}
        out: List[str] = []
        for i, raw in enumerate(prompts):
            rewritten = by_id.get(str(i), "")
            if not rewritten:
                logger.warning(f"DecodeEvasion rewrite empty for prompt {i}; falling back to raw prompt")
                rewritten = raw
            out.append(rewritten)
        return out
