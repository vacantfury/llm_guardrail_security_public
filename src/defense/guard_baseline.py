"""
GuardBaseline — "guard as actually deployed" honest baseline (Round-3).

No recover, no decode: the guard sees the RAW input exactly as it would in a
real deployment — the still-possibly-encoded text channel (+ image channel,
for a multimodal guard). This is deliberately the WEAKEST guard condition in
the Round-3 ablation; it exists to measure what a guard catches ALONE, as the
reference point the recover+decode amplifier (`ModalityComplete`'s
`guard_model=` mode, see modality_complete.py) is measured against.

Mechanism, per prompt:
  1. Guard verdict on the raw input:
       - multimodal guard  -> image + text via build_conversation_message
       - text-only guard   -> text channel ONLY (p.encoded); the image is
         NEVER forwarded, even if the upstream prompt is multimodal (see
         guard_utils.TEXT_ONLY_GUARDS).
  2. UNSAFE -> return guard_utils.GUARD_REFUSAL_TEXT, no target call.
     SAFE   -> query the TARGET with the ORIGINAL input (build_conversation_
       message), return its response.

Same lazy-second-model pattern as SemanticSmooth (see semantic_smooth.py).
"""
from pathlib import Path
from typing import Optional

from src.experiment.schemas import Prompt
from llm_utils.base_llm_service import BaseLLMService
from llm_utils.llm_model import LLMModel
from llm_utils.llm_service_factory import LLMServiceFactory
from src.utils.logger import get_logger
from .base import Defense, build_conversation_message
from .defender_factory import register_defense
from .guard_utils import GUARD_REFUSAL_TEXT, TEXT_ONLY_GUARDS, query_guard

logger = get_logger(__name__)


@register_defense
class GuardBaseline(Defense):
    """Guard model as deployed: raw-input verdict, no recover/decode."""

    type_name = "guard_baseline"

    def __init__(self, guard_model: str, **kwargs):
        """
        Args:
            guard_model: required model name (string) for the guard —
                resolved via LLMModel.from_string at construction time
                (fails loud on a typo, before any API/cluster call).
        """
        super().__init__(guard_model=guard_model, **kwargs)
        self._guard_model_name = guard_model
        self._guard_model = LLMModel.from_string(guard_model)
        self._guard_service: Optional[BaseLLMService] = None

    def _get_guard_service(self) -> BaseLLMService:
        if self._guard_service is None:
            self._guard_service = LLMServiceFactory.create(self._guard_model_name)
        return self._guard_service

    def query(
        self,
        prompts: list[Prompt],
        target_service: BaseLLMService,
        is_multimodal: bool,
        source_dir: Optional[Path] = None,
        system_message: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        guard_service = self._get_guard_service()
        guard_is_text_only = self._guard_model in TEXT_ONLY_GUARDS

        # Oracle vs deployable arm — see module docstring. Default "encoded"
        # preserves every recorded campaign's behavior.
        query_source = self._config.get("query_source", "encoded")
        if query_source not in ("original", "encoded"):
            raise ValueError(
                f"unknown guard_baseline query_source {query_source!r} "
                "(expected 'original' or 'encoded')")

        # ---------- Step 1: guard verdict on the RAW input ----------
        guard_items: list[tuple[str, str, object]] = []
        for p in prompts:
            if query_source == "original":
                # Oracle: the guard reads the unencoded request (pure text,
                # no image — the unencoded request has no image channel).
                guard_items.append((p.id, p.original or p.encoded or "", None))
            elif guard_is_text_only:
                guard_items.append((p.id, p.encoded or "", None))
            else:
                messages = build_conversation_message(p, is_multimodal, source_dir)
                text_side, image_side = messages[0]
                guard_items.append((p.id, text_side or "", image_side))

        logger.info(
            f"GuardBaseline: {len(guard_items)} raw-input verdicts via "
            f"{self._guard_model_name} (text_only={guard_is_text_only}, "
            f"is_multimodal={is_multimodal}, query_source={query_source})")
        verdicts = query_guard(
            guard_service, self._guard_model, guard_items, is_test=True)

        # ---------- Step 2: SAFE -> target with ORIGINAL input ----------
        target_convs: list[tuple[str, list]] = []
        for p in prompts:
            # Fail-closed default: an id missing a verdict (shouldn't
            # happen — batch_chat returns one row per input) is treated as
            # unsafe, never silently forwarded.
            if not verdicts.get(p.id, True):
                messages = build_conversation_message(p, is_multimodal, source_dir)
                target_convs.append((p.id, messages))

        logger.info(
            f"GuardBaseline: {len(target_convs)}/{len(prompts)} passed guard "
            f"-> querying target")
        target_results: dict[str, str] = {}
        if target_convs:
            target_results = dict(target_service.batch_chat(
                conversations=target_convs,
                system_message=system_message,
                is_test=True,
            ))

        return [
            (p.id, target_results.get(p.id, GUARD_REFUSAL_TEXT))
            for p in prompts
        ]

    def get_usage(self) -> Optional[dict]:
        """LLM usage from the guard model (a SEPARATE service from the
        target). The target's own usage is tracked by target_service and
        surfaced via the task's `target_usage` field, not re-counted here."""
        if self._guard_service is not None:
            return {"guard": self._guard_service.get_usage()}
        return None
