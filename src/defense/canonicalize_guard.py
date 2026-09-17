"""CanonicalizeGuard — canonicalize the input, THEN a guard screens it (Paper D / R4).

The effective-N defense against Best-of-N, done as the paper's thesis intends
(the earlier `canonicalize` defense fed the canonical form straight to the TARGET,
with no guard — which amplified rather than defended; see proposal.md §8-9).

Mechanism, per prompt:
  1. Canonicalize the input (NFKC + strip-control + case-fold + whitespace),
     collapsing the *canonicalizable* Best-of-N axes (random capitalization,
     whitespace, Unicode homoglyphs) to ONE canonical form.
  2. A guard screens that canonical form; UNSAFE -> block, SAFE -> forward the
     canonical form to the target.

On the canonicalizable axes all N Best-of-N variants map to the SAME canonical
string, so the guard renders ONE decision for the whole sampling budget ->
**N-independent** protection (a slope/exponent kill, not the per-variant
constant-multiplier a repeated-judge gate like DATDP gives). The irreducible
tail (word scrambling, ASCII noising) survives canonicalization and is what an
adaptive attacker relocates onto; measuring that surviving tail's cost is the R4
gate.

Implementation = GuardBaseline (guard-screens-input -> block-or-forward) applied
to CANONICALIZED prompts, so both the guard verdict and the forwarded target
query see the canonical form. The guard model + canonicalization toggles come
from `defense_config`; the guard is a SEPARATE served model (e.g. wildguard).
"""
from pathlib import Path
from typing import Optional

from src.experiment.schemas import Prompt
from llm_utils.base_llm_service import BaseLLMService
from src.utils.logger import get_logger
from .canonicalize import canonicalize_text
from .defender_factory import register_defense
from .guard_baseline import GuardBaseline

logger = get_logger(__name__)


@register_defense
class CanonicalizeGuardDefense(GuardBaseline):
    """Canonicalize the input, then screen the canonical form with a guard."""

    type_name = "canonicalize_guard"

    def __init__(self, guard_model: str, nfkc: bool = True,
                 strip_control: bool = True, case_fold: bool = True,
                 normalize_whitespace: bool = True, unscramble: bool = False,
                 **kwargs):
        super().__init__(guard_model=guard_model, nfkc=nfkc,
                         strip_control=strip_control, case_fold=case_fold,
                         normalize_whitespace=normalize_whitespace,
                         unscramble=unscramble, **kwargs)
        self._canon_opts = dict(nfkc=nfkc, strip_control=strip_control,
                                case_fold=case_fold,
                                normalize_whitespace=normalize_whitespace,
                                unscramble=unscramble)

    def query(
        self,
        prompts: list[Prompt],
        target_service: BaseLLMService,
        is_multimodal: bool,
        source_dir: Optional[Path] = None,
        system_message: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        canon_prompts = [
            p.model_copy(update={"encoded": canonicalize_text(
                p.encoded or "", **self._canon_opts)})
            for p in prompts
        ]
        logger.info(
            f"CanonicalizeGuard: canonicalized {len(canon_prompts)} prompts; "
            f"screening the canonical form with {self._guard_model_name}")
        return super().query(
            canon_prompts, target_service, is_multimodal, source_dir,
            system_message)
