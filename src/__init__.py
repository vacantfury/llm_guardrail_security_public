"""Host package init — wires the llm_utils base package to this repo's config.

llm_utils (v2.2.0+) has no host-config knowledge; the host registers a loader
here so ``LLMServiceFactory.create()`` keeps merging ``conf/llm/*.yaml``
per-model defaults exactly as the vendored copy did.
"""
from llm_utils import LLMServiceFactory


def _llm_model_defaults(model) -> dict:
    from src.experiment.config import load_conf  # lazy: avoids import cycles
    return load_conf(
        "llm", section="model",
        match_field="model.model", match_value=model.model_id)


LLMServiceFactory.set_config_loader(_llm_model_defaults)
