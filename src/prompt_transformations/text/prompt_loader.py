"""
Prompt template loader for text encoders.

Loads prompt templates from YAML files in conf/text_encoding/.
"""
import yaml
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Path to encoder config/template directory.
# This file lives at src/prompt_transformations/text/prompt_loader.py, so
# `.parents[3]` is the project root (src/prompt_transformations/text → text →
# prompt_transformations → src → root).
CONF_DIR = Path(__file__).resolve().parents[3] / "conf" / "text_encoding"


def load_prompt_template(template_path: str) -> dict[str, Any]:
    """
    Load a prompt template from a YAML file.
    
    Args:
        template_path: Relative path within conf/text_encoding/
                       e.g., "classical_language/classical_chinese.yaml"
    
    Returns:
        Parsed YAML content as dict.
    
    Raises:
        FileNotFoundError: If template file doesn't exist.
    """
    full_path = CONF_DIR / template_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {full_path}\n"
            f"Available templates: {list_prompt_templates()}")
    
    with open(full_path) as f:
        data = yaml.safe_load(f)
    
    logger.info(f"Loaded prompt template: {template_path}")
    return data


def list_prompt_templates() -> list[str]:
    """List all available prompt template files."""
    if not CONF_DIR.exists():
        return []
    return [
        str(p.relative_to(CONF_DIR))
        for p in CONF_DIR.rglob("*.yaml")
    ]
