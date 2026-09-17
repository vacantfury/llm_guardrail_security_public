"""
Unified factory for prompt transformations (attacks + image renderers).

Name resolution:
  1. Direct registry hit (e.g. "set_theory", "ir_plain", "deep_inception")
  2. Alias map (e.g. "plain" → "non_llm_baseline")
  3. Classical-language auto-resolve (e.g. "latin_literary" → llm_classical_language
     with language=latin, strategy=literary)

Subpackages:
  text/   — pure text-to-text transformations (encoders + new attacks)
  image/  — text-to-multimodal transformations (image renderers)

Each subpackage is imported at module load; their constructors register into
the TRANSFORMATIONS dict via the @register_transformation decorator.
"""
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from .base import PromptTransformation

logger = get_logger(__name__)


# Registry populated by imports below. type_name → class.
TRANSFORMATIONS: dict[str, type[PromptTransformation]] = {}


# User-facing shorthand → canonical type_name.
# Only when the user-friendly name differs from the registered type_name.
TRANSFORMATION_ALIASES: dict[str, str] = {
    "plain": "non_llm_baseline",
    "set_theory": "llm_set_theory",
    "formal_logic": "llm_formal_logic",
    "quantum": "llm_quantum_mechanics",
    "quantum_mechanics": "llm_quantum_mechanics",
    "addition_equation": "non_llm_addition_equation_split_reassemble",
    "conditional_probability": "non_llm_conditional_probability",
    "symbol_injection": "non_llm_symbol_injection",
    "semantic_camo": "llm_semantic_camo",
    "decode_evasion": "llm_decode_evasion",
    "cipher": "non_llm_cipher",
    # Variance-Channel Best-of-N (Paper D)
    "paraphrase": "llm_paraphrase",
    "variance_channel": "variance_channel_bon",
    "bon_wrapper": "variance_channel_bon",
    # Established multimodal attacks (Paper C ensemble, added 2026-07-16)
    "low_contrast": "ir_low_contrast",
    "occluded": "ir_occluded",
    "mm_typo": "ir_mm_typo",
    "mm_safetybench": "ir_mm_typo",
    "semantic_split": "ir_semantic_split",
    "distraction": "ir_distraction_grid",
    "distraction_grid": "ir_distraction_grid",
    "camo": "ir_camo",
    "cross_modal_obfuscation": "ir_camo",
}


# Classical-language autoresolve directory (YAML configs).
_CLASSICAL_LANG_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "conf" / "text_encoding" / "classical_language"
)
_LITERARY_SUFFIX = "_literary"


def register_transformation(cls: type[PromptTransformation]) -> type[PromptTransformation]:
    """Class decorator: register the transformation under its type_name.

    Use as:
        @register_transformation
        class SetTheoryEncoder(PromptTransformation):
            type_name = "llm_set_theory"
            ...
    """
    name = cls.type_name
    if not name:
        raise ValueError(
            f"{cls.__name__} must declare a non-empty `type_name` class attr "
            f"to be registered.")
    if name in TRANSFORMATIONS and TRANSFORMATIONS[name] is not cls:
        raise ValueError(
            f"Transformation name {name!r} already registered to "
            f"{TRANSFORMATIONS[name].__name__}; cannot re-register {cls.__name__}.")
    TRANSFORMATIONS[name] = cls
    return cls


def _resolve_classical_language(name: str) -> Optional[tuple[str, dict]]:
    """Match e.g. 'latin' or 'latin_literary' to a classical_language YAML config."""
    strategy = "direct"
    lang = name
    if name.endswith(_LITERARY_SUFFIX):
        strategy = "literary"
        lang = name[: -len(_LITERARY_SUFFIX)]
    if (_CLASSICAL_LANG_DIR / f"{lang}.yaml").exists():
        return "llm_classical_language", {"language": lang, "strategy": strategy}
    return None


def resolve_transformation_name(name: str) -> tuple[str, dict]:
    """Resolve a user-facing transformation name to (canonical_type_name, extra_kwargs).

    Order:
      1. Alias map (set_theory → llm_set_theory)
      2. Direct registry hit
      3. Classical-language auto-resolve
    """
    if name in TRANSFORMATION_ALIASES:
        return TRANSFORMATION_ALIASES[name], {}
    if name in TRANSFORMATIONS:
        return name, {}
    classical = _resolve_classical_language(name)
    if classical is not None:
        return classical
    available = sorted(set(TRANSFORMATIONS.keys()) | set(TRANSFORMATION_ALIASES.keys()))
    raise ValueError(
        f"Unknown transformation {name!r}. Available: {available}")


def create_transformation(name: str, **kwargs) -> PromptTransformation:
    """Create a transformation instance by name.

    User kwargs override any auto-resolved kwargs (e.g. from classical_language
    autoresolve).
    """
    canonical, auto_kwargs = resolve_transformation_name(name)
    if canonical not in TRANSFORMATIONS:
        raise ValueError(
            f"Resolved transformation {name!r} → {canonical!r} but no class "
            f"is registered. Available: {sorted(TRANSFORMATIONS.keys())}.")
    merged = {**auto_kwargs, **kwargs}
    cls = TRANSFORMATIONS[canonical]
    try:
        return cls(**merged)
    except Exception:
        logger.exception(f"Error constructing transformation {canonical!r}")
        raise


def list_transformations() -> list[str]:
    """All registered transformation type_names (canonical, not aliases)."""
    return sorted(TRANSFORMATIONS.keys())


# Import subpackages so their @register_transformation calls fire.
# (deferred to bottom to avoid circular imports during decorator execution)
from . import text as _text  # noqa: E402,F401
from . import image as _image  # noqa: E402,F401
