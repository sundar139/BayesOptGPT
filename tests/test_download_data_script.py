from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_download_data_script_dry_run(tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: ag_news",
                "  config: null",
                "  text_field: text",
                "  label_field: label",
                "  validation_split_ratio: 0.1",
                "  random_seed: 42",
                "  debug_subset_size: null",
                "tokenizer:",
                "  vocab_size: 128",
                "  min_frequency: 1",
                "  max_sequence_length: 32",
                "dataloader:",
                "  batch_size: 2",
                "  num_workers: 0",
                "  pin_memory: false",
                "paths:",
                f"  dataset_cache_dir: {tmp_path / 'hf_cache'}",
                f"  processed_cache_dir: {tmp_path / 'processed'}",
                f"  tokenizer_dir: {tmp_path / 'tokenizer'}",
            ]
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "scripts/download_data.py",
        "--config",
        str(config_path),
        "--dry-run",
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "dry_run=true" in result.stdout
    assert "train_tokenizer=True" in result.stdout
