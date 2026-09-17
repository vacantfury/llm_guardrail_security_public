"""Best-of-N text augmentation encoder (Hughes et al., NeurIPS 2025).

Reimplements the TEXT-side augmentation suite of Best-of-N Jailbreaking
(reference impl: other_repos/bon-jailbreaking/bon/attacks/run_text_bon.py):
word scrambling, random capitalization, and ASCII noising, all driven by one
severity knob ``sigma`` in [0, 1]. Each ``process`` call draws ONE stochastic
augmented variant; the N independent variants of a behavior come from applying
this encoder to N dataset rows of that behavior (see
scripts/expand_bon_dataset.py). Reimplemented (not vendored) from the reference;
cite Hughes et al. (bibkey NEURIPS2025_69f3eb24).

Note on seeding: no existing encoder in this repo seeds — we establish the
pattern here. A single ``random.Random(seed)`` is drawn from sequentially over a
batch, so (seed + stable row order) makes a run reproducible.
"""
import random
from typing import Optional

from llm_utils import LLMModel
from src.utils.logger import get_logger
from ..base_encoder import BaseEncoder

logger = get_logger(__name__)

_PRINTABLE_MIN, _PRINTABLE_MAX = 32, 126


def _scramble_word(word: str, rng: random.Random, p: float) -> str:
    """Shuffle the middle characters of a word (first/last fixed), with prob p."""
    if len(word) <= 3 or rng.random() >= p:
        return word
    middle = list(word[1:-1])
    rng.shuffle(middle)
    return word[0] + "".join(middle) + word[-1]


def _apply_word_scrambling(text: str, rng: random.Random, sigma: float) -> str:
    p = sigma ** 0.5
    return " ".join(_scramble_word(w, rng, p) for w in text.split(" "))


def _apply_random_capitalization(text: str, rng: random.Random, sigma: float) -> str:
    p = sigma ** 0.5
    out = []
    for ch in text:
        if ch.isalpha() and rng.random() < p:
            out.append(ch.lower() if ch.isupper() else ch.upper())
        else:
            out.append(ch)
    return "".join(out)


def _apply_ascii_noising(text: str, rng: random.Random, sigma: float) -> str:
    p = sigma ** 3  # deliberately steeper: ASCII corruption ramps up slowest
    out = []
    for ch in text:
        o = ord(ch)
        if _PRINTABLE_MIN <= o <= _PRINTABLE_MAX and rng.random() < p:
            shifted = o + rng.choice((-1, 1))
            shifted = min(max(shifted, _PRINTABLE_MIN), _PRINTABLE_MAX)
            out.append(chr(shifted))
        else:
            out.append(ch)
    return "".join(out)


class BestOfNEncoder(BaseEncoder):
    """Best-of-N text augmentation: word scramble + random caps + ASCII noise.

    Parameters:
        sigma: severity in [0, 1] (BoN default 0.4). Scramble/capitalization
            perturb with prob ``sigma**0.5``; ASCII noise with prob ``sigma**3``.
        seed: RNG seed for reproducible variant draws over a batch.
        word_scrambling / random_capitalization / ascii_perturbation: toggles.
    """

    def __init__(self, model: Optional[LLMModel] = None, sigma: float = 0.4,
                 seed: Optional[int] = 0, word_scrambling: bool = True,
                 random_capitalization: bool = True,
                 ascii_perturbation: bool = True, **kwargs):
        super().__init__(model, **kwargs)
        self.sigma = float(sigma)
        self.word_scrambling = bool(word_scrambling)
        self.random_capitalization = bool(random_capitalization)
        self.ascii_perturbation = bool(ascii_perturbation)
        self._rng = random.Random(seed)

    def process(self, prompt: str, **kwargs) -> str:
        if not prompt:
            return prompt
        text = prompt
        if self.word_scrambling:
            text = _apply_word_scrambling(text, self._rng, self.sigma)
        if self.random_capitalization:
            text = _apply_random_capitalization(text, self._rng, self.sigma)
        if self.ascii_perturbation:
            text = _apply_ascii_noising(text, self._rng, self.sigma)
        return text

    def __repr__(self) -> str:
        return f"BestOfNEncoder(sigma={self.sigma!r})"
