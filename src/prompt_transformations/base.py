"""
Base interface for prompt transformations.

A PromptTransformation takes a list of Prompt records and returns a list of
transformed Prompt records of equal length, preserving id alignment. It may
also write per-step artifacts (e.g., images) to a provided step_dir.

Modality declaration:
  - input_modality:  what kind of Prompt content this expects to consume
  - output_modality: what kind of Prompt content this produces
The chain executor uses these to derive cumulative is_multimodal state and
to validate chain ordering at config-load time.
"""
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

from src.experiment.schemas import Prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Modality(str, Enum):
    """Channel-shape of Prompt content at a given pipeline point."""
    TEXT = "text"             # only text fields populated
    MULTIMODAL = "multimodal"  # at least one image_* field populated


class PromptTransformation(ABC):
    """Abstract base for every step in a prompt_transform chain.

    Subclasses set:
      - type_name        : canonical factory key (e.g. "set_theory", "ir_plain")
      - input_modality   : Modality.TEXT or Modality.MULTIMODAL
      - output_modality  : Modality.TEXT or Modality.MULTIMODAL

    and implement apply().
    """

    type_name: str = ""
    input_modality: Modality = Modality.TEXT
    output_modality: Modality = Modality.TEXT

    def __init__(self, **kwargs):
        # Stored verbatim for results.json provenance — every kwarg the
        # factory passed in (after YAML merge) ends up here.
        self._config: dict = dict(kwargs)

    @abstractmethod
    def apply(self, prompts: list[Prompt], step_dir: Path) -> list[Prompt]:
        """Transform a list of Prompt records.

        Args:
            prompts: input Prompts (already sliced to prompt_range).
            step_dir: directory dedicated to this step. The transformation
                may write artifacts here (e.g., images/). Created by the
                chain executor before this is called.

        Returns:
            New list of Prompt records, same length, same id order.
        """
        raise NotImplementedError

    def get_config(self) -> dict:
        """Resolved params for this transformation, stored in results.json.

        Default: returns the kwargs the factory built this with. Subclasses
        may override to add derived/normalized fields.
        """
        return dict(self._config)

    def get_usage(self) -> Optional[dict]:
        """LLM usage incurred by THIS transformation (e.g., encoder LLM calls).

        Returns None for rule-based transformations. LLM-based encoders
        return a dict like {"algorithm": {...}, "total": {...}}.
        """
        return None

    def step_metrics(self) -> dict:
        """Per-step metrics merged into results_history[step].metrics.

        Default: empty. Subclasses can override to add things like
        image_count, character_count_delta, etc.
        """
        return {}
