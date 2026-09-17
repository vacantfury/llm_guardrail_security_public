"""
Unicode homoglyph encoder (non-LLM, rule-based).

Replaces ASCII letters with visually-identical Unicode confusables (homoglyphs),
**reusing the `homoglyphs` library** for the confusable map. The text looks
identical to a human and reads the same to the target model, but differs
byte-wise from what a safety classifier matches on — the "classifier fooled,
model understands" decode/normalize gap.

Citation: the homoglyph attack (Boucher et al., 2022, "Bad Characters",
`9833641`). The substitution map comes from the `homoglyphs` library (Unicode
confusables), NOT a hand-rolled table."""
import string
from typing import Optional, Dict

from llm_utils import LLMModel
from src.utils.logger import get_logger
from ..base_encoder import BaseEncoder

logger = get_logger(__name__)

# Only ASCII letters are substituted (digits/punctuation/space left alone), so
# the obfuscation lands on words while keeping structure readable.
_ELIGIBLE = set(string.ascii_letters)


class HomoglyphEncoder(BaseEncoder):
    """Substitute ASCII letters with confusable Unicode homoglyphs.

    Parameters:
        ratio: fraction (0-1) of eligible characters to substitute. 1.0
            (default) = every eligible char (max classifier evasion); lower
            leaves some ASCII for partial obfuscation.
        deterministic: reproducible left-to-right substitution (default True).
    """

    TARGET_PREFIX = ""  # homoglyph'd text is read directly — no decode prefix.

    def __init__(
        self,
        model: Optional[LLMModel] = None,
        ratio: float = 1.0,
        deterministic: bool = True,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.ratio = max(0.0, min(1.0, float(ratio)))
        self.deterministic = deterministic
        self.TARGET_PREFIX = ""          # force-empty regardless of inherited YAML.
        self._hg = None                  # lazy homoglyphs.Homoglyphs() handle.
        self._char_map: Dict[str, Optional[str]] = {}  # ascii char -> confusable.
        logger.info(
            f"Initialized HomoglyphEncoder (ratio={self.ratio}, "
            f"deterministic={self.deterministic})")

    def _get_homoglyph(self, ch: str) -> Optional[str]:
        """Deterministic single-char non-ASCII confusable for `ch`, or None."""
        if ch in self._char_map:
            return self._char_map[ch]
        if self._hg is None:
            try:
                import homoglyphs as hg
            except ImportError as e:  # pragma: no cover - env without the dep
                raise ImportError(
                    "The homoglyph encoder requires the 'homoglyphs' package "
                    "(pip install homoglyphs)."
                ) from e
            # Load cross-script alphabets so Latin letters map to Cyrillic/Greek
            # look-alikes (not just Latin variants).
            self._hg = hg.Homoglyphs(
                categories=("LATIN", "COMMON", "CYRILLIC", "GREEK"))
        chosen: Optional[str] = None
        for cand in self._hg.get_combinations(ch):
            if len(cand) == 1 and ord(cand) > 0x7F:  # single-char, non-ASCII
                chosen = cand
                break
        self._char_map[ch] = chosen
        return chosen

    def process(self, prompt: str, **kwargs) -> str:
        if not prompt:
            return prompt
        eligible_idx = [i for i, ch in enumerate(prompt) if ch in _ELIGIBLE]
        if not eligible_idx:
            return prompt
        n_sub = int(round(len(eligible_idx) * self.ratio))
        if n_sub <= 0:
            return prompt
        if n_sub >= len(eligible_idx):
            chosen_idx = set(eligible_idx)
        else:  # even-spaced positions for a deterministic partial substitution
            step = len(eligible_idx) / n_sub
            chosen_idx = {eligible_idx[int(k * step)] for k in range(n_sub)}

        out = []
        n_done = 0
        for i, ch in enumerate(prompt):
            if i in chosen_idx:
                hgly = self._get_homoglyph(ch)
                if hgly:
                    out.append(hgly)
                    n_done += 1
                    continue
            out.append(ch)
        if n_done == 0:
            logger.warning(
                "HomoglyphEncoder substituted 0 chars — check the 'homoglyphs' "
                "install / category config; returning near-original text.")
        return "".join(out)

    def __repr__(self) -> str:
        return f"HomoglyphEncoder(ratio={self.ratio!r})"
