"""
Fixed algorithmic constants for text encoding strategies.

Tunable parameters (model, num_parts, num_symbols, target_prefix, etc.)
live in conf/text_encoding/*.yaml configs. This file contains only
non-tunable implementation constants.
"""
from typing import Final, List

# =============================================================================
# Split-Reassemble Algorithm
# =============================================================================

# Boundary detection scoring (algorithm internals, not tunable)
BOUNDARY_WINDOW_PERCENT: Final[float] = 0.1  # Look within ±10% of target size
SENTENCE_BOUNDARY_SCORE: Final[int] = 3
PHRASE_BOUNDARY_SCORE: Final[int] = 2
WORD_BOUNDARY_SCORE: Final[int] = 1

# Content analysis guards
MIN_CONTENT_WORD_LENGTH: Final[int] = 4
MIN_WORDS_PER_PART: Final[int] = 2

# Sentence and phrase boundary indicators
SENTENCE_ENDERS: Final[List[str]] = [".", "!", "?", ";", ":"]
PHRASE_ENDERS: Final[List[str]] = [",", "-", ")", "]", "}", '"']

# =============================================================================
# Symbol Injection
# =============================================================================

# The fixed pool of symbols available for injection.
DEFAULT_INJECTION_SYMBOLS: Final[List[str]] = [
    # Mathematical operators
    "∀", "∃", "∈", "∉", "⊂", "⊆", "∪", "∩", "∅", "×", "÷", "±", "∞", "≈", "≠", "≤", "≥",
    "∫", "∑", "∏", "∂", "∇", "√", "∛", "∜", "∝", "∧", "∨", "¬", "⇒", "⇔", "→", "←", "↔",
    # Greek letters
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "λ", "μ", "π", "ρ", "σ", "τ", "φ", "χ", "ψ", "ω",
    "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Λ", "Μ", "Π", "Ρ", "Σ", "Τ", "Φ", "Χ", "Ψ", "Ω",
    # Other special symbols
    "†", "‡", "§", "¶", "•", "◦", "○", "●", "★", "☆", "♠", "♣", "♥", "♦", "◊", "■", "□", "▪",
    "▫", "▲", "△", "▼", "▽", "◀", "▶", "⊕", "⊗", "⊙", "⋅", "∘", "·", "…", "⋯", "⋮", "⋰", "⋱"
]
