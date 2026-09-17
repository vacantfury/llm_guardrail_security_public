"""Defenses for the defense+evaluate stage.

A Defense owns the target-model interaction loop. Provided implementations:
  - no_defense        : pure baseline, pass-through to the target model
  - sage              : prepends a self-discrimination safety instruction
  - semantic_smooth   : N paraphrase summaries + majority vote
  - ecso              : 3-call response-coupled defense (TELL → CAP → safe-gen)
  - llm_self_defense  : output-side self-examination gate (Phute et al. 2024)
  - selfdefend        : input-side gate via a SEPARATE shadow model (Wang 2025)

The last two straddle the boundary the borrowed-strength account predicts:
llm_self_defense screens with the TARGET (borrows its disposition, so coverage
should vary by target), selfdefend screens with a fixed separate shadow model
(borrows nothing, so coverage should not). See their conf/defense/*.yaml for
the pre-registered predictions.
"""
from .base import Defense, build_conversation_message
from .defender_factory import (
    DEFENSES,
    create_defense,
    create_defender,
    list_defenses,
    register_defense,
)

__all__ = [
    "Defense",
    "build_conversation_message",
    "DEFENSES",
    "create_defense",
    "create_defender",
    "list_defenses",
    "register_defense",
]
