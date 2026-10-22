"""Central configuration for the RAG evaluation framework.

Single source of truth for:
- pass/fail thresholds per metric (used for CI gating in later phases)
- judge model + retrieval settings
- filesystem paths used across the eval pipeline

All values are overridable via environment variables / .env,
following the same pattern as app/core/config.py.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# advanced-rag/
BASE_DIR = Path(__file__).resolve().parents[2]


class EvalThresholds(BaseSettings):
    """Minimum acceptable score (0.0-1.0) for each metric.

    A run is marked FAIL if any metric's average score drops
    below its threshold. Used by the CI regression gate (Phase 6).
    """

    faithfulness_min: float = 0.70
    answer_relevancy_min: float = 0.70
    contextual_relevancy_min: float = 0.60
    contextual_recall_min: float = 0.60
    contextual_precision_min: float = 0.60
    citation_correctness_min: float = 0.70
    citation_completeness_min: float = 0.80

    model_config = SettingsConfigDict(
        env_prefix="EVAL_THRESHOLD_",
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

class EvalSettings(BaseSettings):
    """Runtime configuration for evaluation runs."""

    # Judge LLM (via GroqJudge, see app/evaluation/models/groq.py)
    judge_temperature: float = 0.0

    # Retrieval
    retrieval_k: int = 5

    # Execution
    max_concurrency: int = 1
    request_timeout_seconds: int = 300

    # Paths (all relative to the project root)
    golden_dataset_dir: Path = BASE_DIR / "eval" / "datasets"
    legacy_golden_dataset_path: Path = BASE_DIR / "eval" / "golden_dataset.json"
    results_dir: Path = BASE_DIR / "eval" / "results"
    documents_dir: Path = BASE_DIR / "storage" / "documents"

    # Experiment tracking
    mlflow_experiment_name: str = "rag-evaluation"

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_eval_settings() -> EvalSettings:
    return EvalSettings()


@lru_cache
def get_eval_thresholds() -> EvalThresholds:
    return EvalThresholds()
