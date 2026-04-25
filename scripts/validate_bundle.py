"""Validate the integrity of a packaged inference bundle.

Checks that all required files are present and that their SHA-256
checksums match the values recorded in checksums.json.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bayes_gp_llmops.serving.bundle import validate_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a packaged inference bundle (required files + checksums)."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/model/bundle"),
        help="Path to the bundle directory (default: artifacts/model/bundle).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    bundle_dir: Path = args.bundle_dir
    try:
        metadata = validate_bundle(bundle_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"validation=FAILED\nerror={exc}", file=sys.stderr)
        return 1

    print("validation=passed")
    print(f"bundle_dir={bundle_dir}")
    print(f"champion_trial_number={metadata.champion_trial_number}")
    print(f"champion_study_name={metadata.champion_study_name}")
    print(f"created_at={metadata.created_at}")
    print(f"has_calibration={metadata.has_calibration}")
    print(f"included_files={len(metadata.included_files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
