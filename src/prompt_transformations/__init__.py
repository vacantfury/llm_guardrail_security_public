"""
Prompt transformations — the stage 1 pipeline layer.

A PromptTransformation is any operation applied to a list of Prompt records:
  - Text transformations (set_theory, formal_logic, deep_inception, ...)
  - Image transformations (ir_plain, ir_fc_typo, ir_figstep, ...)

All transformations share one interface (apply(prompts, step_dir) -> prompts)
and one factory (create_transformation). A chain runs them one-by-one,
saving each step's output in its own subfolder under the task's output dir.
"""
from .base import PromptTransformation, Modality
from .transformation_factory import (
    create_transformation,
    list_transformations,
    resolve_transformation_name,
)

__all__ = [
    "PromptTransformation",
    "Modality",
    "create_transformation",
    "list_transformations",
    "resolve_transformation_name",
]
