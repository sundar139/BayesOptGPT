from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DashboardConfig:
    """Runtime configuration for the Streamlit dashboard."""

    title: str
    subtitle: str
    evaluation_dir: Path
    bundle_dir: Path
    api_base_url: str | None

    @classmethod
    def from_env(cls) -> DashboardConfig:
        title = os.getenv("DASHBOARD_TITLE", "LLM Calibration & Evaluation Dashboard")
        subtitle = os.getenv(
            "DASHBOARD_SUBTITLE",
            "AG News | Bayesian Hyperparameter Optimization | Uncertainty Quantification",
        )
        evaluation_dir = Path(
            os.getenv("DASHBOARD_ARTIFACT_DIR", "artifacts/evaluation_full_run")
        )
        bundle_dir = Path(os.getenv("DASHBOARD_BUNDLE_DIR", "artifacts/model/bundle"))
        api_base_url = _normalize_optional_url(os.getenv("API_BASE_URL"))
        return cls(
            title=title,
            subtitle=subtitle,
            evaluation_dir=evaluation_dir,
            bundle_dir=bundle_dir,
            api_base_url=api_base_url,
        )


def _normalize_optional_url(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped.rstrip("/")
