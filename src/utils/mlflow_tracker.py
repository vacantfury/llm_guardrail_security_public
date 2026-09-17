"""
MLflow experiment tracking for Encoding × Modality jailbreaking.

Thin wrapper around MLflow that handles run lifecycle, param/metric logging,
and artifact storage. All data is stored locally in mlruns/ (no server needed).

MLflow is an INDEX over runs, not a second copy of the data: results.json on
disk is the system of record. We log tags + scalar metrics + a pointer to the
output dir — we do NOT mirror results/prompts/raw_results into mlruns/.

Usage (called by task.py — not directly):
    tracker = MLflowTracker()
    run_id = tracker.start_run(mode="defense+evaluate", encoding="set_theory",
                               model="Qwen2.5-VL-7B-Instruct", modality="image")
    tracker.log_params(task_config)
    tracker.log_metrics({"attack_success_rate": 45.2})
    tracker.log_output_dir("outputs/defense+evaluate/harmbench/...")  # pointer, not a copy
    tracker.end_run()
"""
import os
from typing import Any, Optional

# Newer MLflow puts the local file-store backend in "maintenance mode" and refuses
# writes unless this opt-out is set; keep local runs (e.g. gpt-5-mini rejudge passes)
# working without migrating to a DB backend. Set before importing mlflow.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow

from src.paths import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

# MLflow experiment name
EXPERIMENT_NAME = "encoding_modality_jailbreak"

# Local tracking directory (relative to project root)
TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"


class MLflowTracker:
    """Thin wrapper for MLflow experiment tracking.
    
    Manages run lifecycle and provides simple methods for logging
    params, metrics, and artifacts. All data stored locally in mlruns/.
    """
    
    def __init__(self):
        self._run_id: Optional[str] = None
        self._active = False
        
        # Point MLflow to local storage
        mlflow.set_tracking_uri(TRACKING_URI)
    
    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID (None if no active run)."""
        return self._run_id
    
    def start_run(self, mode: str, encoding: str = "",
                  model: str = "", modality: str = "") -> str:
        """Start an MLflow run.
        
        Args:
            mode: Task mode ("prompt_transform", "defense+evaluate")
            encoding: Encoding / chain label ("set_theory", "formal_logic", ...)
            model: Target model ("Qwen2.5-VL-7B-Instruct", etc.)
            modality: Input modality ("text", "image")
        
        Returns:
            The MLflow run ID (UUID string)
        """
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        parts = [mode]
        if encoding:
            parts.append(encoding)
        if model:
            parts.append(model)
        if modality:
            parts.append(modality)
        run_name = "_".join(parts)
        
        run = mlflow.start_run(run_name=run_name)
        self._run_id = run.info.run_id
        self._active = True
        
        # Set tags for easy filtering in MLflow UI
        tags = {"mode": mode}
        if encoding:
            tags["encoding"] = encoding
        if model:
            tags["model"] = model
        if modality:
            tags["modality"] = modality
        mlflow.set_tags(tags)
        
        logger.info(f"MLflow run started: {run_name} (ID: {self._run_id})")
        return self._run_id
    
    def log_params(self, task_config: dict) -> None:
        """Log task config as MLflow parameters.
        
        Args:
            task_config: The task configuration dict from experiment YAML
        """
        if not self._active:
            return
        
        try:
            def _cap(v):  # MLflow rejects >250-char param values — cap, don't fail
                s = str(v)
                return s if len(s) <= 250 else s[:247] + "..."
            params = {}
            for key, value in task_config.items():
                if isinstance(value, dict):
                    # Flatten one level: renderer.font_size, etc.
                    for sub_key, sub_val in value.items():
                        params[f"{key}.{sub_key}"] = _cap(sub_val)
                else:
                    params[key] = _cap(value)

            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} params to MLflow")
        except Exception as e:
            # Don't let MLflow errors break the experiment
            logger.warning(f"MLflow param logging failed (non-fatal): {e}")
    
    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log evaluation metrics.
        
        Only logs numeric values (int, float). Skips non-numeric fields.
        
        Args:
            metrics: Dict with evaluation metrics (e.g., attack_success_rate)
        """
        if not self._active:
            return
        
        try:
            numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            if numeric:
                mlflow.log_metrics(numeric)
                logger.debug(f"Logged {len(numeric)} metrics to MLflow: {list(numeric.keys())}")
        except Exception as e:
            logger.warning(f"MLflow metric logging failed (non-fatal): {e}")
    
    def log_artifact(self, file_path: str) -> None:
        """Log a file as an MLflow artifact.
        
        Args:
            file_path: Absolute path to the file to log
        """
        if not self._active:
            return
        
        try:
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)
                logger.debug(f"Logged artifact to MLflow: {os.path.basename(file_path)}")
        except Exception as e:
            logger.warning(f"MLflow artifact logging failed (non-fatal): {e}")

    def log_output_dir(self, output_dir: str) -> None:
        """Record the canonical output dir as a pointer (param + tag) instead of
        copying results/prompts/raw_results into mlruns/. results.json on disk is
        the system of record; MLflow is just a searchable index over runs.
        """
        if not self._active:
            return
        try:
            mlflow.log_param("output_dir", str(output_dir))
            mlflow.set_tags({"output_dir": str(output_dir)})
        except Exception as e:
            logger.warning(f"MLflow output_dir logging failed (non-fatal): {e}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self._active:
            return
        
        try:
            mlflow.end_run()
            logger.info(f"MLflow run ended: {self._run_id}")
        except Exception as e:
            logger.warning(f"MLflow end_run failed (non-fatal): {e}")
        finally:
            self._active = False
