"""
Unified factory for defenses (NoDefense + SAGE + SemanticSmooth + ECSO + ...).

Use as:
    from src.defense import create_defense
    defense = create_defense("no_defense")          # pure attack baseline
    defense = create_defense("sage")
    defense = create_defense("ecso", judge_model="gpt-5-nano")
"""
from src.utils.logger import get_logger
from .base import Defense

logger = get_logger(__name__)


# Registry populated by submodule imports + @register_defense decorator.
DEFENSES: dict[str, type[Defense]] = {}


def register_defense(cls: type[Defense]) -> type[Defense]:
    """Class decorator: register Defense subclass under its type_name."""
    name = cls.type_name
    if not name:
        raise ValueError(
            f"{cls.__name__} must declare a non-empty type_name to register.")
    if name in DEFENSES and DEFENSES[name] is not cls:
        raise ValueError(
            f"Defense {name!r} already registered to {DEFENSES[name].__name__}; "
            f"cannot re-register {cls.__name__}.")
    DEFENSES[name] = cls
    return cls


def create_defense(name: str, **kwargs) -> Defense:
    """Create a Defense instance by canonical type_name."""
    if name not in DEFENSES:
        raise ValueError(
            f"Unknown defense {name!r}. Available: {sorted(DEFENSES)}")
    try:
        return DEFENSES[name](**kwargs)
    except Exception:
        logger.exception(f"Error constructing defense {name!r}")
        raise


def list_defenses() -> list[str]:
    return sorted(DEFENSES.keys())


# Backward-compat alias for older imports.
create_defender = create_defense


# Import submodules so their @register_defense decorators fire.
# Order doesn't matter; each class registers under its own type_name.
from . import no_defense  # noqa: E402,F401
from . import sage  # noqa: E402,F401
from . import semantic_smooth  # noqa: E402,F401
from . import ecso  # noqa: E402,F401
# Paper C defenses (import after sage/ecso — they reuse those primitives).
from . import modality_complete  # noqa: E402,F401
from . import joint_verify  # noqa: E402,F401
# Paper C multi-surface baseline: AMIA intention-analysis component (no masking).
from . import amia_ia  # noqa: E402,F401
# Round-3: guard-as-deployed baseline + amplifier-generalized modality_complete
# both live behind guard_utils.py's shared parsers/formatters.
from . import guard_baseline  # noqa: E402,F401
# Paper D: canonicalization defense (effective-N reduction vs Best-of-N).
from . import canonicalize  # noqa: E402,F401
# Paper D / R4: canonicalize -> guard-screen (the guarded architecture the
# guardless `canonicalize` above lacked; see proposal.md §9).
from . import canonicalize_guard  # noqa: E402,F401
# Paper D / P6: two MORE published self-check architectures, added so the
# borrowed-strength account is tested beyond SAGE alone. They straddle the
# boundary the account predicts — llm_self_defense screens with the TARGET
# (borrows, should vary by target), selfdefend screens with a fixed SEPARATE
# shadow model (borrows nothing, should not vary).
from . import llm_self_defense  # noqa: E402,F401
from . import selfdefend  # noqa: E402,F401
from . import cider  # noqa: E402,F401
