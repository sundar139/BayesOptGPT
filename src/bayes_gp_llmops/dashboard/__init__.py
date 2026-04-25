"""Dashboard utilities for Streamlit visualization and optional live inference."""

from .config import DashboardConfig
from .data import (
    CLASS_LABELS,
    DashboardData,
    calibration_comparison_rows,
    load_dashboard_data,
    metric_number,
    per_class_f1_rows,
    uncertainty_summary,
)
from .inference import (
    PredictionResult,
    fetch_serving_metadata,
    normalize_api_base_url,
    run_batch_prediction,
    run_single_prediction,
)

__all__ = [
    "CLASS_LABELS",
    "DashboardConfig",
    "DashboardData",
    "PredictionResult",
    "calibration_comparison_rows",
    "fetch_serving_metadata",
    "load_dashboard_data",
    "metric_number",
    "normalize_api_base_url",
    "per_class_f1_rows",
    "run_batch_prediction",
    "run_single_prediction",
    "uncertainty_summary",
]
