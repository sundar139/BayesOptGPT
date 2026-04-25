from __future__ import annotations

import argparse
from pathlib import Path

from bayes_gp_llmops.evaluation.pipeline import run_evaluation_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AG News classifier checkpoint."
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--disable-temperature-scaling", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_evaluation_pipeline(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        checkpoint_path=args.checkpoint,
        device_override=args.device,
        output_dir=args.output_dir,
        enable_temperature_scaling=not args.disable_temperature_scaling,
        debug_mode=args.debug,
    )
    print(f"evaluation_output_dir={artifacts.output_dir}")
    print(f"metrics_validation={artifacts.metrics_validation_path}")
    print(f"metrics_test={artifacts.metrics_test_path}")
    if artifacts.metrics_validation_calibrated_path is not None:
        print(f"metrics_validation_calibrated={artifacts.metrics_validation_calibrated_path}")
    if artifacts.metrics_test_calibrated_path is not None:
        print(f"metrics_test_calibrated={artifacts.metrics_test_calibrated_path}")
    print(f"predictions_validation={artifacts.predictions_validation_path}")
    print(f"predictions_test={artifacts.predictions_test_path}")
    print(f"temperature_scaling={artifacts.temperature_scaling_path}")
    print(f"confusion_matrix_plot={artifacts.confusion_matrix_plot_path}")
    print(f"reliability_diagram_plot={artifacts.reliability_diagram_plot_path}")
    print(f"confidence_histogram_plot={artifacts.confidence_histogram_plot_path}")
    print(f"entropy_histogram_plot={artifacts.entropy_histogram_plot_path}")


if __name__ == "__main__":
    main()
