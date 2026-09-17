"""Variance-Channel Best-of-N wrapper — the general two-knob attack (Paper D).

Realizes the paper's core apparatus: a Best-of-N attack parameterized by its
VARIANCE CHANNEL — *where* the N draws of a behavior differ. One transform, two
uniform knobs, a selectable channel:

  - surface    : delegate to the vanilla Best-of-N encoder (char/caps/ASCII
                 noise). Correlated draws; an input-normalization defense
                 collapses them (effective-N -> ~1).
  - paraphrase : the which-paraphrase knob — a stochastic meaning-preserving
                 reword per row (llm_paraphrase). Survives normalization.
  - strategy   : the which-attack knob — sample an attack FAMILY per row from a
                 configured bank (optionally after a paraphrase pre-step).
                 Anti-correlated draws; strongest defense-survival.

Generality is structural: the bank holds any registered PromptTransformation
type_name, resolved through the factory — no per-attack code. The N draws come
from N dataset rows (scripts/expand_bon_dataset.py fans a behavior into N rows
BEFORE this step); this transform decides, per row, WHAT varies.

Vary DURING, not AFTER (proposal §3): the variance is the per-row choice of
transform/paraphrase, never surface noise bolted onto a structured attack's
output. Sub-transform LLM usage is aggregated so the work-factor accounting
(total attacker compute, not just N) stays honest."""
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.experiment.schemas import Prompt
from src.prompt_transformations.base import Modality, PromptTransformation
from src.prompt_transformations.transformation_factory import (
    create_transformation,
    register_transformation,
)
from src.utils.logger import get_logger
from .prompt_loader import load_prompt_template

logger = get_logger(__name__)

_CHANNELS = ("surface", "paraphrase", "strategy")
_CONFIG_FILE = "variance_channel.yaml"


@register_transformation
class VarianceChannelTransformation(PromptTransformation):
    """Best-of-N parameterized by variance channel (surface / paraphrase / strategy)."""

    type_name = "variance_channel_bon"
    input_modality = Modality.TEXT
    # Text-only bank for now; supporting image renderers in the bank would flip
    # this to MULTIMODAL (a future extension noted in the proposal).
    output_modality = Modality.TEXT

    def __init__(
        self,
        channel: Optional[str] = None,
        seed: Optional[int] = None,
        model: Optional[str] = None,
        paraphrase: Optional[dict] = None,
        attack_bank: Optional[list] = None,
        surface: Optional[dict] = None,
        single_attack: Optional[dict] = None,
        **kwargs,
    ):
        # Structural defaults live in the YAML (no magic numbers in code);
        # explicit kwargs (from a preset) take precedence.
        cfg = load_prompt_template(_CONFIG_FILE)
        channel = channel or cfg.get("channel", "strategy")
        seed = cfg.get("seed", 0) if seed is None else seed
        model = model or cfg.get("model")
        paraphrase = paraphrase if paraphrase is not None else (cfg.get("paraphrase") or {})
        attack_bank = attack_bank if attack_bank is not None else (cfg.get("attack_bank") or [])
        surface = surface if surface is not None else (cfg.get("surface") or {})
        single_attack = single_attack if single_attack is not None else (cfg.get("single_attack") or {})

        super().__init__(
            channel=channel, seed=seed, model=model, paraphrase=paraphrase,
            attack_bank=attack_bank, surface=surface,
            single_attack=single_attack, **kwargs,
        )

        if channel not in _CHANNELS:
            raise ValueError(f"channel must be one of {_CHANNELS}, got {channel!r}")
        if channel == "strategy" and not attack_bank:
            raise ValueError("channel='strategy' requires a non-empty attack_bank")

        self._channel = channel
        self._seed = int(seed)
        self._model = model
        self._paraphrase_cfg = dict(paraphrase)
        self._attack_bank = list(attack_bank)
        self._surface_cfg = dict(surface)
        self._single_attack_cfg = dict(single_attack)
        self._usage_totals: dict = {}

        logger.info(
            f"VarianceChannel: channel={channel}, seed={self._seed}, "
            f"bank_size={len(self._attack_bank)}, "
            f"paraphrase={'on' if self._paraphrase_cfg.get('enabled') else 'off'}")

    # -- sub-transform helpers ----------------------------------------------

    def _make(self, type_name: str, params: Optional[dict]) -> PromptTransformation:
        merged = dict(params or {})
        if self._model is not None:
            merged.setdefault("model", self._model)
        return create_transformation(type_name, **merged)

    def _run(self, sub: PromptTransformation, prompts, step_dir):
        out = sub.apply(prompts, step_dir)
        self._accumulate_usage(sub.get_usage())
        return out

    def _accumulate_usage(self, usage: Optional[dict]) -> None:
        if not isinstance(usage, dict):
            return
        total = usage["total"] if isinstance(usage.get("total"), dict) else usage
        for k, v in total.items():
            if isinstance(v, (int, float)):
                self._usage_totals[k] = self._usage_totals.get(k, 0) + v

    def _paraphrase_step(self, prompts, step_dir):
        sub = self._make(
            self._paraphrase_cfg.get("type", "llm_paraphrase"),
            self._paraphrase_cfg.get("params"))
        return self._run(sub, prompts, step_dir)

    def _retag(self, prompts, tag: Optional[str] = None):
        return [p.model_copy(update={"encoding": tag or self.type_name})
                for p in prompts]

    # -- channels -----------------------------------------------------------

    def apply(self, prompts: list[Prompt], step_dir: Path) -> list[Prompt]:
        if not prompts:
            return prompts
        return {
            "surface": self._apply_surface,
            "paraphrase": self._apply_paraphrase,
            "strategy": self._apply_strategy,
        }[self._channel](prompts, step_dir)

    def _apply_surface(self, prompts, step_dir):
        sub = self._make("non_llm_best_of_n", {"seed": self._seed, **self._surface_cfg})
        return self._retag(self._run(sub, prompts, step_dir))

    def _apply_paraphrase(self, prompts, step_dir):
        para = self._paraphrase_step(prompts, step_dir)
        atk = self._single_attack_cfg.get("type")
        if not atk:
            return self._retag(para)
        sub = self._make(atk, self._single_attack_cfg.get("params"))
        return self._retag(self._run(sub, para, step_dir), tag=f"{self.type_name}:{atk}")

    def _apply_strategy(self, prompts, step_dir):
        base = (self._paraphrase_step(prompts, step_dir)
                if self._paraphrase_cfg.get("enabled") else prompts)
        rng = random.Random(self._seed)
        weights = [float(e.get("weight", 1.0)) for e in self._attack_bank]
        population = list(range(len(self._attack_bank)))
        # Per-row attack choice, drawn in stable row order (BoN-encoder pattern).
        chosen = [rng.choices(population, weights=weights)[0] for _ in base]

        groups: dict[int, list[int]] = defaultdict(list)
        for row_idx, bank_idx in enumerate(chosen):
            groups[bank_idx].append(row_idx)

        out: list[Optional[Prompt]] = [None] * len(base)
        for bank_idx, row_idxs in groups.items():
            entry = self._attack_bank[bank_idx]
            sub = self._make(entry["type"], entry.get("params"))
            res = self._run(sub, [base[i] for i in row_idxs], step_dir)
            tag = f"{self.type_name}:{entry['type']}"
            for i, r in zip(row_idxs, res):
                out[i] = r.model_copy(update={"encoding": tag})
        return out  # every index assigned above

    # -- provenance ---------------------------------------------------------

    def get_usage(self) -> Optional[dict]:
        if not self._usage_totals:
            return None
        return {"algorithm": self.type_name, "total": dict(self._usage_totals)}

    def step_metrics(self) -> dict:
        m = {"variance_channel": self._channel}
        if self._channel == "strategy":
            m["bank_size"] = len(self._attack_bank)
        return m
