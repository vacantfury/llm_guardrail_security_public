"""
Adapter base for legacy BaseEncoder subclasses.

The text encoders that pre-date the destroyer refactor (set_theory, formal_logic,
classical_language, semantic_camo, baseline, etc.) inherit from BaseEncoder and
expose a `batch_process(list[str]) -> list[str]` interface. This module wraps
any such legacy encoder as a PromptTransformation so the new pipeline can run
them via the standard `apply(prompts, step_dir)` interface.
"""
from pathlib import Path
from typing import ClassVar, Optional, Type

from src.experiment.schemas import Prompt
from src.prompt_transformations.base import PromptTransformation, Modality
from .base_encoder import BaseEncoder


# Internal kwargs that the chain executor injects (or that PromptTransformation
# already consumes) and must NOT be forwarded to legacy encoder constructors —
# those constructors will reject unknown args.
_CHAIN_META_KWARGS = frozenset({
    "type_name", "input_modality", "output_modality",
})


class TextEncoderTransformation(PromptTransformation):
    """Adapter wrapping a legacy BaseEncoder subclass.

    Subclasses set `type_name` and `encoder_class`. The chain executor sees the
    same interface as every other PromptTransformation: `apply(prompts, step_dir)`
    returns prompts with `encoded` rewritten by `legacy.batch_process`.
    """

    input_modality = Modality.TEXT
    output_modality = Modality.TEXT

    encoder_class: ClassVar[Optional[Type[BaseEncoder]]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.encoder_class is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set `encoder_class`")
        encoder_kwargs = {
            k: v for k, v in kwargs.items() if k not in _CHAIN_META_KWARGS
        }
        self._encoder: BaseEncoder = self.encoder_class(**encoder_kwargs)

    def apply(self, prompts: list[Prompt], step_dir: Path) -> list[Prompt]:
        inputs = [p.encoded for p in prompts]
        outputs = self._encoder.batch_process(inputs)
        return [
            p.model_copy(update={"encoded": out, "encoding": self.type_name})
            for p, out in zip(prompts, outputs)
        ]

    def get_usage(self) -> Optional[dict]:
        if hasattr(self._encoder, "get_usage"):
            return self._encoder.get_usage()
        return None
