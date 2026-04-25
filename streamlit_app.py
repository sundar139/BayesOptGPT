from __future__ import annotations

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


def main() -> None:
    config = DashboardConfig.from_env()
    st.set_page_config(
        page_title=config.title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(config.title)
    st.caption(config.subtitle)

    evaluation_dir, bundle_dir, api_base_url = _render_sidebar(config)
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
        _render_visualizations(dashboard_data.image_paths)
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


def _render_sidebar(config: DashboardConfig) -> tuple[Path, Path, str | None]:
    with st.sidebar:
        st.subheader("Dashboard Configuration")
        evaluation_dir = Path(
            st.text_input(
                "Evaluation artifact directory",
                value=str(config.evaluation_dir),
                help="Directory containing metrics JSON and visualization PNG outputs.",
            )
        )
        bundle_dir = Path(
            st.text_input(
                "Promoted bundle directory",
                value=str(config.bundle_dir),
                help="Directory containing bundle_metadata.json and champion_manifest.json.",
            )
        )
        api_base_input = st.text_input(
            "API base URL",
            value=config.api_base_url or "",
            help="Optional FastAPI base URL, for example http://localhost:7860.",
        )
        api_base_url = normalize_api_base_url(api_base_input)
        if api_base_url is None:
            st.info("Live inference is disabled until API base URL is configured.")
        else:
            st.success(f"Live inference enabled via {api_base_url}")
    return evaluation_dir, bundle_dir, api_base_url


def _render_overview(
    validation_metrics: dict[str, object] | None,
    test_metrics: dict[str, object] | None,
) -> None:
    st.subheader("Full-Split KPI Overview")
    _render_kpi_cards("Validation", validation_metrics)
    _render_kpi_cards("Test", test_metrics)
    st.info(
        "Full-split performance is strong, calibration is stable across validation and test, "
        "and Business/Sci-Tech remains the most confusable category pair."
    )


def _render_kpi_cards(
    split_name: str,
    metrics: dict[str, object] | None,
) -> None:
    st.markdown(f"**{split_name}**")
    keys = (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("nll", "NLL"),
        ("brier_score", "Brier Score"),
        ("ece", "ECE"),
    )
    columns = st.columns(len(keys))
    for index, (key, label) in enumerate(keys):
        value = metric_number(metrics, key)
        with columns[index]:
            if value is None:
                st.metric(label, "N/A")
            else:
                st.metric(label, f"{value:.6f}")


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


def _render_visualizations(image_paths: dict[str, Path | None]) -> None:
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
        st.info("Live inference is disabled. Set API base URL in the sidebar to enable requests.")
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
    st.markdown("**Artifact paths**")
    st.code(
        "\n".join(
            [
                f"evaluation_dir={evaluation_dir}",
                f"bundle_dir={bundle_dir}",
            ]
        )
    )

    st.markdown("**Bundle metadata**")
    if bundle_metadata is None:
        st.info("bundle_metadata.json is not available.")
    else:
        st.json(bundle_metadata)

    st.markdown("**Champion manifest**")
    if champion_manifest is None:
        st.info("champion_manifest.json is not available.")
    else:
        st.json(champion_manifest)

    st.markdown("**Serving metadata (optional)**")
    if api_base_url is None:
        st.caption("Configure API base URL to read /metadata.")
    else:
        if st.button("Fetch /metadata", use_container_width=False):
            try:
                metadata = fetch_serving_metadata(api_base_url=api_base_url)
                st.json(metadata)
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


if __name__ == "__main__":
    main()
