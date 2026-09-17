"""
NoDefense — pure-attack baseline.

Just queries the target model with the prompt verbatim. Use this as the
`defense` value in a defense+evaluate task to measure ASR without any defense
applied — the headline baseline for the destroyer matrix.

The upstream image transformation has already set up Prompt.encoded with the
right text-channel content (either the original encoded attack with keep_text=True,
or a stock "check the image" instruction with keep_text=False), so NoDefense
just forwards (text, image) straight to the model.
"""
from pathlib import Path
from typing import Optional

from src.experiment.schemas import Prompt
from llm_utils.base_llm_service import BaseLLMService
from src.utils.logger import get_logger
from .base import Defense, build_conversation_message
from .defender_factory import register_defense

logger = get_logger(__name__)


@register_defense
class NoDefense(Defense):
    """Pass-through: call the target model with no defense wrapping."""

    type_name = "no_defense"

    def query(
        self,
        prompts: list[Prompt],
        target_service: BaseLLMService,
        is_multimodal: bool,
        source_dir: Optional[Path] = None,
        system_message: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        conversations: list[tuple[str, list]] = []
        for p in prompts:
            messages = build_conversation_message(
                p, is_multimodal=is_multimodal, source_dir=source_dir,
            )
            conversations.append((p.id, messages))

        logger.info(
            f"NoDefense: forwarding {len(conversations)} prompts to target "
            f"(is_multimodal={is_multimodal})")
        return target_service.batch_chat(
            conversations=conversations,
            system_message=system_message,
            is_test=True,
        )
