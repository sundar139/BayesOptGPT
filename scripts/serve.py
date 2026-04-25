from __future__ import annotations

import argparse
import os
from pathlib import Path

from bayes_gp_llmops.cli import run_serve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the bundle-driven FastAPI inference service."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to serving configuration YAML.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Override bundle directory path.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override listening host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override listening port.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device preference (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Override maximum batch size for /predict/batch.",
    )
    parser.add_argument(
        "--max-input-length-chars",
        type=int,
        default=None,
        help="Override maximum input text length in characters.",
    )
    parser.add_argument(
        "--enable-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable temperature scaling at inference time.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override service log level.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config is not None:
        os.environ["SERVING_CONFIG_PATH"] = str(args.config)
    if args.bundle_dir is not None:
        os.environ["SERVING_BUNDLE_DIR"] = str(args.bundle_dir)
    if args.host is not None:
        os.environ["SERVING_HOST"] = args.host
    if args.port is not None:
        os.environ["SERVING_PORT"] = str(args.port)
    if args.device is not None:
        os.environ["SERVING_DEVICE_PREFERENCE"] = args.device
    if args.max_batch_size is not None:
        os.environ["SERVING_MAX_BATCH_SIZE"] = str(args.max_batch_size)
    if args.max_input_length_chars is not None:
        os.environ["SERVING_MAX_INPUT_LENGTH_CHARS"] = str(args.max_input_length_chars)
    if args.enable_calibration is not None:
        os.environ["SERVING_ENABLE_CALIBRATION"] = (
            "true" if args.enable_calibration else "false"
        )
    if args.log_level is not None:
        os.environ["SERVING_LOG_LEVEL"] = args.log_level

    run_serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
