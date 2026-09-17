"""
Baseline processor - passthrough that returns input unchanged.

Used to measure target model behavior without any encoding/transformation.
Supports is_repeating for "baseline + repeat" defense experiments.
"""
from ..base_encoder import BaseEncoder


class BaselineEncoder(BaseEncoder):
    """
    No-op processor that passes prompts through unchanged.
    
    This replaces the old 'baseline' mode in task.py, allowing baseline
    to participate in the normal processor pipeline (including repeat).
    """

    def process(self, prompt: str, **kwargs) -> str:
        """Return the prompt unchanged."""
        return prompt
