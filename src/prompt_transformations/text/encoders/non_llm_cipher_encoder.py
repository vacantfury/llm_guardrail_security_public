"""
Cipher encoder (non-LLM, rule-based).

Deterministic classical/naive cipher family, selected via a `cipher` param.
Two ciphers are implemented:

  - `base64` (default) — standard library Base64. NOT a CipherChat cipher
    (CipherChat has no base64 variant); this is the well-known encoding
    baseline requested alongside Caesar.
  - `caesar` — mod-26 substitution shift, default shift 3. Ported to match
    CipherChat's `encode_experts.py::CaesarExpert.encode` exactly: only
    `[a-zA-Z]` are shifted (case-preserving), everything else passes
    through unchanged. The naive/textbook Caesar cipher has no meaningful
    difference from CipherChat's here, so no divergence to note.

Citation: CipherChat — "GPT-4 Is Too Smart To Be Safe: Stealthy Chat with
LLMs via Cipher" (Yuan et al., ICLR 2024; code: github.com/RobustNLP/
CipherChat, vendored at other_repos/CipherChat). CipherChat's premise:
safety alignment is trained almost entirely on natural-language text, so a
deterministic, reversible re-encoding of the harmful request evades the
surface-level safety filter while a sufficiently capable model can still
decode-and-answer — the same "classifier fooled, model understands" decode
gap that non_llm_homoglyph tests via Unicode normalization instead.

CipherChat ciphers NOT ported here (scope control; `_CIPHERS` is a plain
dispatch dict, so adding one later is a small diff, not a restructure):
  - `morse`, `ascii` — straightforward faithful ports, omitted only to keep
    this first cut to the two ciphers asked for.
  - `atbash` — CipherChat's own `AtbashExpert.encode` has a latent bug: its
    `N - ord(s)` reflection constant (`ord('z') + ord('a')` = 219) only
    reflects correctly for LOWERCASE letters; uppercase input maps to
    non-ASCII control-range chars. A faithful port would corrupt every
    capitalized prompt (English sentences start capitalized), so it's
    excluded rather than silently shipping CipherChat's bug.
  - `unicode` / `utf` / `gbk` — near-duplicate unicode/byte-escape variants;
    `gbk` is CJK-encoding specific.
  - `selfdefine` — Chinese-character substitution table; a different
    research axis (overlaps llm_classical_language's chinese-script angle).
"""
import base64 as _base64
from typing import Optional

from llm_utils import LLMModel
from src.utils.logger import get_logger
from ..base_encoder import BaseEncoder

logger = get_logger(__name__)

_DEFAULT_CAESAR_SHIFT = 3


def _encode_base64(text: str, shift: int) -> str:
    """Standard Base64 (stdlib `base64`). `shift` is unused; kept for a
    uniform dispatch signature across ciphers."""
    return _base64.b64encode(text.encode("utf-8")).decode("ascii")


def _encode_caesar(text: str, shift: int) -> str:
    """Caesar substitution cipher — mod-26 shift, case-preserving,
    non-letters unchanged. Verbatim logic match to CipherChat's
    `encode_experts.py::CaesarExpert.encode`."""
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr(ord("a") + (ord(ch) - ord("a") + shift) % 26))
        elif "A" <= ch <= "Z":
            out.append(chr(ord("A") + (ord(ch) - ord("A") + shift) % 26))
        else:
            out.append(ch)
    return "".join(out)


# type_name(str) -> encode_fn(text, shift) -> ciphertext. Plain dispatch dict
# so a new cipher is a new function + one entry here (see module docstring).
_CIPHERS = {
    "base64": _encode_base64,
    "caesar": _encode_caesar,
}


class CipherEncoder(BaseEncoder):
    """Encode a prompt with a deterministic classical/naive cipher.

    Parameters:
        cipher: which cipher to apply — 'base64' (default) or 'caesar'.
        shift: Caesar shift amount (default 3, matches CipherChat's
            `CaesarExpert`). Ignored by ciphers other than 'caesar'.
    """

    def __init__(
        self,
        model: Optional[LLMModel] = None,
        cipher: str = "base64",
        shift: int = _DEFAULT_CAESAR_SHIFT,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        cipher = (cipher or "base64").lower()
        if cipher not in _CIPHERS:
            raise ValueError(
                f"Unknown cipher {cipher!r}. Available: {sorted(_CIPHERS)}")
        self.cipher = cipher
        self.shift = int(shift)
        logger.info(
            f"Initialized CipherEncoder (cipher={self.cipher}, shift={self.shift})")

    def process(self, prompt: str, **kwargs) -> str:
        if not prompt:
            return prompt
        return _CIPHERS[self.cipher](prompt, self.shift)

    def __repr__(self) -> str:
        return f"CipherEncoder(cipher={self.cipher!r}, shift={self.shift!r})"
