"""Post-processing analyses over already-run pipeline outputs.

Pure data processing — no model or judge calls. The `analyze` task mode
(src/experiment/task.py) dispatches here.
"""
from src.analysis.portfolio import run_analysis

__all__ = ["run_analysis"]
