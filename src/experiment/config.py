"""
Configuration loading.

Pydantic models for typed configs live in `src/experiment/schemas.py`
(LLMConfig, ModelConfig, ClusterConfig, TaskConfig union, ExperimentPreset).
This module is just the YAML loading machinery — no schemas defined here.

`load_conf` returns a merged plain dict from the 3-layer YAML composition
(default → override → task_overrides). Call sites that want typed access do
`ModelConfig.model_validate(load_conf("llm", section="model", ...))`.
"""
from pathlib import Path
from typing import Any, Optional

import yaml


CONF_DIR = Path(__file__).resolve().parent.parent.parent / "conf"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge: override wins, nested dicts merge per-key.

    Used as the OmegaConf.merge replacement. List/scalar values are
    replaced wholesale (matches OmegaConf default semantics).
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    """Load a YAML file → dict. Returns {} for empty files."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping at top level of {path}, got "
            f"{type(data).__name__}.")
    # Strip dead Hydra directives; they have no effect here.
    data.pop("defaults", None)
    return data


def load_conf(subdir: str, override_name: Optional[str] = None, *,
              section: Optional[str] = None, match_field: Optional[str] = None,
              match_value: Optional[str] = None,
              task_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """3-layer YAML merge: default.yaml → override → task_overrides.

    Args:
        subdir: Config subdirectory under conf/ (e.g. "llm", "imaging",
                "text_encoding", "defense").
        override_name: File stem to overlay (e.g. "plain", "gpt_4o"). If
                       None, falls back to match_field/match_value search.
        section: If set, returns only that top-level key from the merged dict.
        match_field: Dotted-path field to match when searching by content
                     (e.g. "model.model" → check candidate.model.model).
        match_value: Value the matched field must equal.
        task_overrides: Highest-priority overrides from the task YAML's
                        nested sub-config (e.g. an `encoder:` block).

    Returns:
        Merged plain dict. Validate via Pydantic at the call site.

    Examples:
        load_conf("llm", section="model", match_field="model.model", match_value="gpt-4o")
        load_conf("imaging", override_name="plain")
        load_conf("text_encoding", override_name="formal_logic", task_overrides={...})
    """
    conf_path = CONF_DIR / subdir

    config: dict[str, Any] = {}

    # Layer 1: default.yaml
    default_path = conf_path / "default.yaml"
    if default_path.exists():
        config = _deep_merge(config, _load_yaml(default_path))

    # Layer 2: named override OR search by match_field
    if override_name:
        override_path = conf_path / f"{override_name}.yaml"
        if override_path.exists():
            config = _deep_merge(config, _load_yaml(override_path))
    elif match_field and match_value is not None:
        for yaml_file in sorted(conf_path.glob("*.yaml")):
            # Skip macOS AppleDouble sidecars (._<name>.yaml): pathlib's glob
            # matches these dotfiles, and they are binary resource-fork junk that
            # yaml.safe_load chokes on ("UnicodeDecodeError: 0xa3"). They appear
            # whenever a config file is carried from a Mac to a cluster. (2026-07-22)
            if yaml_file.name.startswith("._") or yaml_file.stem == "default":
                continue
            candidate = _load_yaml(yaml_file)
            if _select(candidate, match_field) == match_value:
                config = _deep_merge(config, candidate)
                break

    # Layer 3: task-level overrides (highest priority)
    if task_overrides:
        config = _deep_merge(config, task_overrides)

    if section:
        return config.get(section, {})
    return config


def _select(d: dict, dotted_path: str) -> Any:
    """Walk a dotted path through nested dicts; None if any segment missing."""
    cur: Any = d
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_cluster_profile(profile: str, conf_dir: Path = CONF_DIR) -> dict[str, Any]:
    """Merge a cluster profile's config from conf/clusters/.

    Layers (base-most first): conf/clusters/_defaults.yaml `cluster` block →
    the profile's `extends` chain → the profile's own `cluster` block. Returns
    the merged `cluster` dict (server params + limits + dispatch keys like
    sbatch/budget/max_submit). Per-MODEL overrides are applied by the caller
    on top of this (they win).

    A cluster file may set top-level `extends: <parent-profile>` to inherit that
    parent's `cluster` block (e.g. nurc_uv extends nurc). Raises FileNotFoundError
    for an unknown profile and ValueError on a circular extends chain.
    """
    clusters_dir = conf_dir / "clusters"

    merged: dict[str, Any] = {}
    defaults_path = clusters_dir / "_defaults.yaml"
    if defaults_path.exists():
        merged = dict(_load_yaml(defaults_path).get("cluster", {}))

    # Walk the extends chain, collecting each file's `cluster` block.
    chain: list[dict] = []
    seen: set[str] = set()
    name: Optional[str] = profile
    while name:
        if name in seen:
            raise ValueError(
                f"circular 'extends' in cluster profile chain at '{name}'")
        seen.add(name)
        path = clusters_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"cluster profile '{name}' not found at {path}")
        data = _load_yaml(path)
        chain.append(data.get("cluster", {}))
        name = data.get("extends")

    for block in reversed(chain):   # parent → child, child wins
        merged = _deep_merge(merged, block)
    return merged
