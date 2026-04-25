from __future__ import annotations

import argparse
from pathlib import Path

from bayes_gp_llmops.tuning.optuna_runner import run_optuna_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter tuning for the AG News classifier."
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--tune-config", type=Path, default=Path("configs/tune.yaml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = run_optuna_study(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        tune_config_path=args.tune_config,
        device_override=args.device,
        n_trials_override=args.n_trials,
        timeout_override=args.timeout,
        debug_override=True if args.debug else None,
    )
    print(f"study_storage={artifacts.storage_path}")
    print(f"study_output_dir={artifacts.output_dir}")
    print(f"best_trial_number={artifacts.best_trial_number}")
    print(f"best_value={artifacts.best_value:.6f}")
    print(f"best_params={artifacts.best_params}")
    print(f"best_params_path={artifacts.best_params_path}")
    print(f"trial_results_path={artifacts.trial_results_path}")
    print(f"study_summary_path={artifacts.study_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
