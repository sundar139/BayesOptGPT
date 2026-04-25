from __future__ import annotations

import argparse
from pathlib import Path

from bayes_gp_llmops.training.pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny LLaMA-style AG News classifier.")
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data pipeline configuration.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model.yaml"),
        help="Path to model configuration.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to training configuration.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device preference (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run with reduced dataset and epoch limits for a quick debug run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = run_training_pipeline(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        device_override=args.device,
        debug_mode=args.debug,
    )
    print(f"best_checkpoint={artifacts.best_checkpoint_path}")
    print(f"latest_checkpoint={artifacts.latest_checkpoint_path}")
    print(f"history_path={artifacts.history_path}")
    print(f"resolved_config_path={artifacts.resolved_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
