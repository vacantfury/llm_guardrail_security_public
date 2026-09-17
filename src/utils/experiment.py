"""
Experiment directory utilities.
"""
import os
import random
import time
from typing import Optional

RANDOM_NUMBER_LIMIT = 100000000

# Active paper namespace for output dirs (set once per experiment run). When set,
# outputs/<mode>/... becomes outputs/<paper>/<mode>/... so multiple paper lines
# (autoattack_defense, bestofn_defense, ...) don't collide in outputs/.
_CURRENT_PAPER: Optional[str] = None


def set_current_paper(paper: Optional[str]) -> None:
    """Set (or clear with None) the paper namespace prepended to output dirs."""
    global _CURRENT_PAPER
    _CURRENT_PAPER = paper


def get_new_experiment_data_dir(experiment_dir: str, dataset: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Create a structured experiment output directory.
    
    Layout: {experiment_dir}/{dataset}/{model_shortname}_{timestamp}_{rand}/
    
    A random suffix prevents collisions when concurrent tasks start in the
    same second.
    
    Args:
        experiment_dir: Base directory (e.g., outputs/baseline_experiment_data)
        dataset: Dataset name (e.g., "BBEH"). Creates a subdirectory.
        model: Model identifier (e.g., "mistralai/Pixtral-12B-2409").
               Uses the short name after the last '/'.
    
    Returns:
        Path to the new experiment directory (already created).
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    rand_suffix = random.randint(0, RANDOM_NUMBER_LIMIT)
    
    path = experiment_dir
    if _CURRENT_PAPER and path.startswith("outputs/"):
        path = f"outputs/{_CURRENT_PAPER}/" + path[len("outputs/"):]
    if dataset:
        path = os.path.join(path, dataset)
    
    model_short = model.split("/")[-1] if model else None
    folder_name = f"{model_short}_{timestamp}_{rand_suffix}" if model_short else f"{timestamp}_{rand_suffix}"
    
    path = os.path.join(path, folder_name)
    os.makedirs(path, exist_ok=True)
    return path