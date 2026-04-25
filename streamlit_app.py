from __future__ import annotations

from html import escape
from importlib import import_module
from pathlib import Path

import pandas as pd
import streamlit as st

from bayes_gp_llmops.dashboard import (
    DashboardConfig,
    PredictionResult,
    calibration_comparison_rows,
    fetch_serving_metadata,
    load_dashboard_data,
    metric_number,
    normalize_api_base_url,
    per_class_f1_rows,
    run_batch_prediction,
    run_single_prediction,
    uncertainty_summary,
)
from bayes_gp_llmops.serving.metadata_safety import (
    sanitize_metadata_mapping,
    sanitize_path_value,
)

KPI_SPECS: tuple[tuple[str, str], ...] = (
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("nll", "NLL"),
    ("brier_score", "Brier Score"),
    ("ece", "ECE"),
)
PERCENT_METRIC_KEYS = {"accuracy", "macro_f1"}
INVERSE_DELTA_KEYS = {"nll", "brier_score", "ece"}


def main() -> None:
    config = DashboardConfig.from_env()
    st.set_page_config(
        page_title=config.title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_theme_css()
    _render_header_banner(config)

    evaluation_dir, bundle_dir, api_base_url = _resolve_runtime_inputs(config)
    dashboard_data = load_dashboard_data(
        evaluation_dir=evaluation_dir,
        bundle_dir=bundle_dir,
    )

    if dashboard_data.warnings:
        for item in dashboard_data.warnings:
            st.warning(item)

    tabs = st.tabs(
        [
            "Overview",
            "Results",
            "Visualizations",
            "Calibration & Uncertainty",
            "Live Inference",
            "Model Metadata",
        ]
    )

    with tabs[0]:
        _render_overview(dashboard_data.metrics_validation, dashboard_data.metrics_test)
    with tabs[1]:
        _render_results(
            validation_metrics=dashboard_data.metrics_validation,
            test_metrics=dashboard_data.metrics_test,
            validation_calibrated=dashboard_data.metrics_validation_calibrated,
            test_calibrated=dashboard_data.metrics_test_calibrated,
        )
    with tabs[2]:
        _render_visualizations(dashboard_data.image_paths, dashboard_data.metrics_test)
    with tabs[3]:
        _render_calibration_and_uncertainty(
            dashboard_data.metrics_validation,
            dashboard_data.metrics_test,
        )
    with tabs[4]:
        _render_live_inference(api_base_url)
    with tabs[5]:
        _render_metadata(
            evaluation_dir=evaluation_dir,
            bundle_dir=bundle_dir,
            api_base_url=api_base_url,
            bundle_metadata=dashboard_data.bundle_metadata,
            champion_manifest=dashboard_data.champion_manifest,
        )


def _resolve_runtime_inputs(config: DashboardConfig) -> tuple[Path, Path, str | None]:
    return (
        config.evaluation_dir,
        config.bundle_dir,
        normalize_api_base_url(config.api_base_url),
    )


def _render_overview(
    validation_metrics: dict[str, object] | None,
    test_metrics: dict[str, object] | None,
) -> None:
    st.subheader("Full-Split KPI Overview")
    validation_column, test_column = st.columns(2, gap="large")
    with validation_column:
        _render_split_heading(label="Validation Split", accent_color="#60a5fa")
        _render_kpi_cards(metrics=validation_metrics)

    with test_column:
        _render_split_heading(label="Test Split", accent_color="#34d399")
        _render_kpi_cards(metrics=test_metrics, baseline_metrics=validation_metrics)

    _try_style_metric_cards()

    st.markdown(
        """
<div style="background: linear-gradient(135deg, #0d2137, #1a3a5c);
            border-left: 4px solid #60a5fa; border-radius: 10px;
            padding: 18px 24px; margin-top: 16px;">
  <p style="color: #93c5fd; font-size: 0.95rem; margin: 0; line-height: 1.6;">
    Full-split performance is strong. Calibration remains stable across validation and test.
    Business/Sci-Tech is still the most confusable category pair.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_cards(
    *,
    metrics: dict[str, object] | None,
    baseline_metrics: dict[str, object] | None = None,
) -> None:
    row_slices = (KPI_SPECS[:3], KPI_SPECS[3:])
    for row_slice in row_slices:
        columns = st.columns(len(row_slice))
        for index, (key, label) in enumerate(row_slice):
            value = metric_number(metrics, key)
            baseline_value = metric_number(baseline_metrics, key)
            with columns[index]:
                if value is None:
                    st.metric(label, "N/A")
                    continue

                formatted_value = _format_kpi_value(key, value)
                delta = _format_delta_badge(key, value, baseline_value)
                if delta is None:
                    st.metric(label, formatted_value)
                    continue

                delta_color = "inverse" if key in INVERSE_DELTA_KEYS else "normal"
                st.metric(
                    label,
                    formatted_value,
                    delta=delta,
                    delta_color=delta_color,
                )


def _render_results(
    *,
    validation_metrics: dict[str, object] | None,
    test_metrics: dict[str, object] | None,
    validation_calibrated: dict[str, object] | None,
    test_calibrated: dict[str, object] | None,
) -> None:
    st.subheader("Results")
    sample_columns = st.columns(2)
    with sample_columns[0]:
        val_samples = metric_number(validation_metrics, "num_samples")
        st.metric("Validation samples", _format_int_metric(val_samples))
    with sample_columns[1]:
        test_samples = metric_number(test_metrics, "num_samples")
        st.metric("Test samples", _format_int_metric(test_samples))

    validation_rows = per_class_f1_rows(validation_metrics)
    test_rows = per_class_f1_rows(test_metrics)
    if validation_rows or test_rows:
        merged = _merge_class_f1(validation_rows, test_rows)
        st.markdown("**Per-class F1**")
        st.dataframe(pd.DataFrame(merged), use_container_width=True, hide_index=True)
    else:
        st.warning("Per-class F1 values are not available.")

    val_uncertainty = uncertainty_summary(validation_metrics)
    test_uncertainty = uncertainty_summary(test_metrics)
    uncertainty_df = pd.DataFrame(
        [
            {
                "split": "validation",
                "mean_confidence": val_uncertainty["mean_confidence"],
                "mean_entropy": val_uncertainty["mean_entropy"],
            },
            {
                "split": "test",
                "mean_confidence": test_uncertainty["mean_confidence"],
                "mean_entropy": test_uncertainty["mean_entropy"],
            },
        ]
    )
    st.markdown("**Confidence and entropy summary**")
    st.dataframe(uncertainty_df, use_container_width=True, hide_index=True)

    st.markdown("**Calibration comparison (optional)**")
    val_comparison = calibration_comparison_rows(
        raw_metrics=validation_metrics,
        calibrated_metrics=validation_calibrated,
    )
    test_comparison = calibration_comparison_rows(
        raw_metrics=test_metrics,
        calibrated_metrics=test_calibrated,
    )
    if val_comparison:
        st.markdown("Validation raw vs calibrated")
        st.dataframe(pd.DataFrame(val_comparison), use_container_width=True, hide_index=True)
    else:
        st.caption("Validation calibrated metrics were not found.")
    if test_comparison:
        st.markdown("Test raw vs calibrated")
        st.dataframe(pd.DataFrame(test_comparison), use_container_width=True, hide_index=True)
    else:
        st.caption("Test calibrated metrics were not found.")


def _render_visualizations(
    image_paths: dict[str, Path | None],
    test_metrics: dict[str, object] | None,
) -> None:
    st.subheader("Visualizations")
    figures = (
        ("Confusion Matrix", "confusion_matrix"),
        ("Reliability Diagram", "reliability_diagram"),
        ("Confidence Histogram", "confidence_histogram"),
        ("Entropy Histogram", "entropy_histogram"),
    )
    first_row = st.columns(2)
    second_row = st.columns(2)
    all_columns = [first_row[0], first_row[1], second_row[0], second_row[1]]
    for index, (title, key) in enumerate(figures):
        with all_columns[index]:
            st.markdown(f"**{title}**")
            path = image_paths.get(key)
            if path is None:
                st.warning(f"{title} image is not available.")
            else:
                st.image(str(path), use_container_width=True)

    st.markdown("---")
    _render_kpi_gauges(test_metrics)


def _render_calibration_and_uncertainty(
    validation_metrics: dict[str, object] | None,
    test_metrics: dict[str, object] | None,
) -> None:
    st.subheader("Calibration & Uncertainty")
    metrics = (
        ("ece", "ECE"),
        ("brier_score", "Brier Score"),
        ("mean_confidence", "Mean Confidence"),
        ("mean_entropy", "Mean Entropy"),
    )

    val_uncertainty = uncertainty_summary(validation_metrics)
    test_uncertainty = uncertainty_summary(test_metrics)
    rows: list[dict[str, float | str | None]] = []
    for key, label in metrics:
        if key in ("mean_confidence", "mean_entropy"):
            val_value = val_uncertainty[key]
            test_value = test_uncertainty[key]
        else:
            val_value = metric_number(validation_metrics, key)
            test_value = metric_number(test_metrics, key)
        rows.append(
            {
                "metric": label,
                "validation": val_value,
                "test": test_value,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(
        "- **ECE** quantifies confidence calibration error (lower is better).\n"
        "- **Brier Score** measures probabilistic forecast quality (lower is better).\n"
        "- **Mean Confidence** summarizes the model's predicted certainty.\n"
        "- **Mean Entropy** summarizes predictive uncertainty distribution."
    )


def _render_live_inference(api_base_url: str | None) -> None:
    st.subheader("Live Inference")
    if api_base_url is None:
        st.info(
            "Live inference is disabled. Set API_BASE_URL to enable requests against serving."
        )
        return

    st.caption(f"Connected endpoint: {api_base_url}")
    single_text = st.text_area(
        "Single input text",
        value="Global markets reacted positively to the latest economic report.",
        height=120,
    )
    if st.button("Run single prediction", use_container_width=False):
        try:
            prediction = run_single_prediction(
                api_base_url=api_base_url,
                text=single_text,
            )
            _render_prediction(prediction)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))

    st.markdown("---")
    batch_text = st.text_area(
        "Batch input (one record per line)",
        value=(
            "World leaders met to discuss global climate targets.\n"
            "The team secured a last-minute victory in the championship."
        ),
        height=120,
    )
    if st.button("Run batch prediction", use_container_width=False):
        lines = [line for line in batch_text.splitlines() if line.strip()]
        try:
            predictions = run_batch_prediction(
                api_base_url=api_base_url,
                texts=lines,
            )
            _render_batch_predictions(predictions)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))


def _render_prediction(prediction: PredictionResult) -> None:
    top_columns = st.columns(4)
    with top_columns[0]:
        st.metric("Predicted label", prediction.label)
    with top_columns[1]:
        st.metric("Confidence", f"{prediction.confidence:.6f}")
    with top_columns[2]:
        st.metric("Entropy", f"{prediction.entropy:.6f}")
    with top_columns[3]:
        st.metric("Margin", f"{prediction.margin:.6f}")
    st.caption(f"Calibrated: {prediction.calibrated}")

    probability_rows = [
        {"label": label, "probability": probability}
        for label, probability in prediction.probabilities.items()
    ]
    probability_df = pd.DataFrame(probability_rows).sort_values(
        by="probability",
        ascending=False,
    )
    st.dataframe(probability_df, use_container_width=True, hide_index=True)


def _render_batch_predictions(predictions: list[PredictionResult]) -> None:
    table_rows = [
        {
            "label": item.label,
            "confidence": item.confidence,
            "entropy": item.entropy,
            "margin": item.margin,
            "calibrated": item.calibrated,
        }
        for item in predictions
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


def _render_metadata(
    *,
    evaluation_dir: Path,
    bundle_dir: Path,
    api_base_url: str | None,
    bundle_metadata: dict[str, object] | None,
    champion_manifest: dict[str, object] | None,
) -> None:
    st.subheader("Model Metadata")
    workspace_root = Path(__file__).resolve().parent
    display_evaluation_dir = sanitize_path_value(evaluation_dir, root=workspace_root)
    display_bundle_dir = sanitize_path_value(bundle_dir, root=workspace_root)

    st.markdown("**Artifact paths**")
    st.code(
        "\n".join(
            [
                f"evaluation_dir={display_evaluation_dir}",
                f"bundle_dir={display_bundle_dir}",
            ]
        )
    )

    st.markdown("**Bundle metadata**")
    if bundle_metadata is None:
        st.info("bundle_metadata.json is not available.")
    else:
        st.json(_sanitize_metadata_payload(bundle_metadata, root=workspace_root))

    st.markdown("**Champion manifest**")
    if champion_manifest is None:
        st.info("champion_manifest.json is not available.")
    else:
        st.json(_sanitize_metadata_payload(champion_manifest, root=workspace_root))

    st.markdown("**Serving metadata (optional)**")
    if api_base_url is None:
        st.caption("Configure API base URL to read /metadata.")
    else:
        if st.button("Fetch /metadata", use_container_width=False):
            try:
                metadata = fetch_serving_metadata(api_base_url=api_base_url)
                st.json(_sanitize_metadata_payload(metadata, root=workspace_root))
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

    st.markdown(
        "Deployment summary: this dashboard consumes immutable evaluation artifacts and can "
        "optionally call the live serving API for runtime predictions."
    )


def _merge_class_f1(
    validation_rows: list[dict[str, float | str]],
    test_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str | None]]:
    val_map = {str(item["class"]): item["f1"] for item in validation_rows}
    test_map = {str(item["class"]): item["f1"] for item in test_rows}
    labels = sorted(set(val_map) | set(test_map))
    return [
        {
            "class": label,
            "validation_f1": _as_optional_float(val_map.get(label)),
            "test_f1": _as_optional_float(test_map.get(label)),
        }
        for label in labels
    ]


def _as_optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_int_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{int(value)}"


def _inject_theme_css() -> None:
    st.markdown(
        """
<style>
.stApp {
    background: radial-gradient(ellipse at top left, #0d1b2a 0%, #0a0f1e 60%, #06080f 100%);
}
[data-testid="stSidebar"] {
    display: none;
}
[data-testid="collapsedControl"] {
    display: none;
}
[data-testid="stSidebarNav"] {
    display: none;
}
[data-testid="stTabs"] button {
    color: #64748b !important;
    font-weight: 500;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #60a5fa !important;
}
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1f35 0%, #0d1828 100%);
    border: 1px solid #1e3d6e;
    border-top: 2px solid #60a5fa;
    border-radius: 14px;
    padding: 20px 16px;
    box-shadow: 0 0 20px rgba(96, 165, 250, 0.08), 0 4px 24px rgba(0, 0, 0, 0.5);
}
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.4rem !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
}
h1, h2, h3 {
    color: #cbd5e1 !important;
}
[data-testid="stInfo"] {
    background: linear-gradient(135deg, #0d1f35, #0f2540) !important;
    border: 1px solid #1e4a7f !important;
    border-left: 4px solid #60a5fa !important;
    border-radius: 12px !important;
    color: #93c5fd !important;
}
[data-testid="stSuccess"] {
    background: linear-gradient(135deg, #052e16, #064e3b) !important;
    border-left: 4px solid #34d399 !important;
    border-radius: 10px !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_header_banner(config: DashboardConfig) -> None:
    st.markdown(
        f"""
<div style="background: linear-gradient(135deg, #0d1b2a 0%, #130a2e 50%, #0d1b2a 100%);
            border: 1px solid #1e3a5f; border-left: 4px solid #60a5fa;
            border-radius: 16px; padding: 28px 36px; margin-bottom: 24px;">
    <p style="color: #38bdf8; font-size: 0.75rem; letter-spacing: 0.15em;
            text-transform: uppercase; margin: 0 0 8px 0; font-weight: 600;">
    BAYESIAN | GAUSSIAN PROCESS | LLMOPS
  </p>
  <h1 style="background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             font-size: 2.2rem; font-weight: 800; margin: 0 0 8px 0; line-height: 1.2;">
    {escape(config.title)}
  </h1>
  <p style="color: #94a3b8; margin: 0;">{escape(config.subtitle)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_split_heading(*, label: str, accent_color: str) -> None:
    st.markdown(
        f"""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
  <div style="width: 10px; height: 10px; background: {accent_color}; border-radius: 50%;"></div>
  <span style="color: {accent_color}; font-size: 0.8rem; font-weight: 700;
               text-transform: uppercase; letter-spacing: 0.1em;">{label}</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def _format_kpi_value(key: str, value: float) -> str:
    if key in PERCENT_METRIC_KEYS:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def _format_delta_badge(
    key: str,
    value: float | None,
    baseline_value: float | None,
) -> str | None:
    if value is None or baseline_value is None:
        return None
    if key in PERCENT_METRIC_KEYS:
        return f"{(value - baseline_value) * 100:+.2f}% vs Val"
    return f"{value - baseline_value:+.4f} vs Val"


def _sanitize_metadata_payload(
    payload: dict[str, object],
    *,
    root: Path,
) -> dict[str, object]:
    return sanitize_metadata_mapping(payload, root=root)


def _render_kpi_gauges(test_metrics: dict[str, object] | None) -> None:
    st.markdown("**KPI Gauges (Test split)**")
    if test_metrics is None:
        st.caption("KPI gauges are unavailable because test metrics were not found.")
        return

    try:
        plotly_go = import_module("plotly.graph_objects")
    except Exception:
        st.caption("Install plotly to render KPI gauge charts.")
        return

    upper_row = st.columns(3)
    lower_row = st.columns(2)
    all_columns = [upper_row[0], upper_row[1], upper_row[2], lower_row[0], lower_row[1]]

    for index, (key, label) in enumerate(KPI_SPECS):
        value = metric_number(test_metrics, key)
        with all_columns[index]:
            if value is None:
                st.metric(label, "N/A")
                continue

            display_value, axis_range, steps = _gauge_config(key, value)
            figure = plotly_go.Figure(
                plotly_go.Indicator(
                    mode="gauge+number",
                    value=display_value,
                    title={"text": f"Test {label}"},
                    gauge={
                        "axis": {"range": axis_range},
                        "bar": {"color": "#60a5fa"},
                        "steps": steps,
                        "threshold": {
                            "line": {"color": "#a78bfa", "width": 4},
                            "value": display_value,
                        },
                    },
                )
            )
            figure.update_layout(height=260, margin={"l": 12, "r": 12, "t": 48, "b": 12})
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _gauge_config(
    key: str,
    value: float,
) -> tuple[float, list[float], list[dict[str, object]]]:
    if key in PERCENT_METRIC_KEYS:
        display = value * 100
        return (
            display,
            [0.0, 100.0],
            [
                {"range": [0.0, 70.0], "color": "#10253d"},
                {"range": [70.0, 85.0], "color": "#163b60"},
                {"range": [85.0, 95.0], "color": "#1f4f80"},
                {"range": [95.0, 100.0], "color": "#2b6cb0"},
            ],
        )

    upper = max(0.1, value * 2.0)
    return (
        value,
        [0.0, upper],
        [
            {"range": [0.0, upper * 0.33], "color": "#163b60"},
            {"range": [upper * 0.33, upper * 0.66], "color": "#1f4f80"},
            {"range": [upper * 0.66, upper], "color": "#2b6cb0"},
        ],
    )


def _try_style_metric_cards() -> None:
    try:
        metric_cards_module = import_module("streamlit_extras.metric_cards")
        style_metric_cards = getattr(metric_cards_module, "style_metric_cards", None)
        if callable(style_metric_cards):
            style_metric_cards(
                background_color="#0f1f35",
                border_left_color="#60a5fa",
            )
    except Exception:
        return


if __name__ == "__main__":
    main()
