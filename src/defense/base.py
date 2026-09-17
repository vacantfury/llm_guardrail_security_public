"""
Base interface for defenses.

A Defense OWNS the target-model interaction. It is free to:
  - wrap the input with extra instructions (SAGE)
  - call the model multiple times and aggregate (SemanticSmooth)
  - call the model, inspect the response, conditionally re-call (ECSO)
  - just forward to the model unchanged (NoDefense)

The stage-2 pipeline (`defense+evaluate`) constructs the Defense, hands it the
target LLMService and the upstream prompt_transform step's output, and lets
the Defense produce the final responses. The caller then runs the canonical
evaluator(s) on those responses.

Multimodal handling: the upstream image transformation has ALREADY set up
both channels of the Prompt:
  - Prompt.encoded     : final text-channel content (either the original
                          encoded attack OR a stock "check the image"
                          instruction, decided by the image transformation's
                          keep_text param)
  - Prompt.image_encoded : relative path to the rendered image

Defenses therefore don't need a separate image_instruction param — the upstream
chain has resolved that question already.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.experiment.schemas import Prompt
from llm_utils.base_llm_service import BaseLLMService


class Defense(ABC):
    """Abstract base for all defenses (including NoDefense)."""

    # Canonical factory key.
    type_name: str = ""

    def __init__(self, **kwargs):
        # Stored verbatim for results.json provenance.
        self._config: dict = dict(kwargs)

    @abstractmethod
    def query(
        self,
        prompts: list[Prompt],
        target_service: BaseLLMService,
        is_multimodal: bool,
        source_dir: Optional[Path] = None,
        system_message: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        """Run defense + target-model interaction for every prompt.

        Args:
            prompts: input Prompts (from a prompt_transform step subfolder).
            target_service: target LLM service to query.
            is_multimodal: True if the upstream step produced image output
                (Prompt.image_encoded is populated). The Defense will resolve
                the image against source_dir and pass it to the model in the
                multimodal-message tuple.
            source_dir: filesystem dir to resolve relative image paths. Required
                when is_multimodal=True.
            system_message: optional system prompt forwarded to the target service.

        Returns:
            One (prompt_id, response_text) per input prompt, same order.
        """
        raise NotImplementedError

    def get_config(self) -> dict:
        """Resolved params for results.json."""
        return dict(self._config)

    def get_usage(self) -> Optional[dict]:
        """Additional LLM usage incurred BY the defense (e.g., a separate
        perturbation or self-check model). Distinct from the target model's
        own usage, which the caller logs separately from
        target_service.get_usage()."""
        return None


# --------------- shared conversation-construction helper ---------------

def build_conversation_message(
    prompt: Prompt,
    is_multimodal: bool,
    source_dir: Optional[Path],
) -> list[tuple[str, object]]:
    """Build a (single-turn) [(text, image_or_None)] message for one Prompt.

    is_multimodal=False → [(prompt.encoded, None)]
    is_multimodal=True  → [(prompt.encoded, PIL.Image | list[PIL.Image])]
        opens every path in prompt.image_encoded relative to source_dir. The
        paginating renderer emits multiple images per prompt; a single page is
        passed as one PIL.Image, multiple pages as a list (the LLM service
        layer accepts either).
    """
    from PIL import Image

    if not is_multimodal:
        return [(prompt.encoded, None)]

    if not prompt.image_encoded:
        raise ValueError(
            f"Prompt {prompt.id} marked is_multimodal=True but has no "
            f"image_encoded path. Did the upstream image transformation run?")
    if source_dir is None:
        raise ValueError(
            "is_multimodal=True requires source_dir to resolve image paths.")

    # image_encoded is normally a list; tolerate a bare string defensively.
    rel_paths = prompt.image_encoded
    if isinstance(rel_paths, str):
        rel_paths = [rel_paths]

    images = []
    for rel in rel_paths:
        image_path = source_dir / rel
        if not image_path.exists():
            raise FileNotFoundError(
                f"Prompt {prompt.id}: image not found at {image_path}")
        images.append(Image.open(image_path))

    image_side = images[0] if len(images) == 1 else images
    return [(prompt.encoded, image_side)]
