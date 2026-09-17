"""
Experiment package: task execution, SLURM orchestration, preset loading.

Lazy module-level imports for `run_task` / `Experiment` / etc. — they pull in
`src.prompt_transformations` which imports `src.experiment.schemas` for type
hints. Eager imports here would cycle through that chain. The lazy
`__getattr__` keeps `from src.experiment import run_task` working without
forcing import order.
"""
from .constants import MAX_SUBMIT_JOBS_PER_USER

__all__ = [
    "run_task",
    "Experiment",
    "load_preset",
    "run_experiment_from_preset",
    "MAX_SUBMIT_JOBS_PER_USER",
]


def __getattr__(name):
    if name == "run_task":
        from .task import run_task
        return run_task
    if name == "Experiment":
        from .experiment import Experiment
        return Experiment
    if name == "load_preset":
        from .experiment import load_preset
        return load_preset
    if name == "run_experiment_from_preset":
        from .experiment import run_experiment_from_preset
        return run_experiment_from_preset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
