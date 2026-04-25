from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from bayes_gp_llmops.data import datasets as datasets_module
from bayes_gp_llmops.data.config import DataPipelineConfig


def _fake_ag_news_dataset() -> DatasetDict:
    train_text = [f"train sample {index}" for index in range(40)]
    train_labels = [index % 4 for index in range(40)]
    test_text = [f"test sample {index}" for index in range(12)]
    test_labels = [index % 4 for index in range(12)]
    return DatasetDict(
        {
            "train": Dataset.from_dict({"text": train_text, "label": train_labels}),
            "test": Dataset.from_dict({"text": test_text, "label": test_labels}),
        }
    )


def _build_config(
    tmp_path: Path, *, seed: int, debug_subset_size: int | None = None
) -> DataPipelineConfig:
    return DataPipelineConfig.model_validate(
        {
            "dataset": {
                "name": "ag_news",
                "config": None,
                "text_field": "text",
                "label_field": "label",
                "validation_split_ratio": 0.2,
                "random_seed": seed,
                "debug_subset_size": debug_subset_size,
            },
            "tokenizer": {
                "vocab_size": 128,
                "min_frequency": 1,
                "max_sequence_length": 32,
            },
            "dataloader": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
            },
            "paths": {
                "dataset_cache_dir": str(tmp_path / "hf_cache"),
                "processed_cache_dir": str(tmp_path / "processed"),
                "tokenizer_dir": str(tmp_path / "tokenizer"),
            },
        }
    )


def test_load_ag_news_splits_returns_expected_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        datasets_module,
        "load_dataset",
        lambda **_: _fake_ag_news_dataset(),
    )
    config = _build_config(tmp_path, seed=7)
    splits = datasets_module.load_ag_news_splits(config)
    assert set(splits.as_dict()) == {"train", "validation", "test"}


def test_validation_split_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        datasets_module,
        "load_dataset",
        lambda **_: _fake_ag_news_dataset(),
    )
    config = _build_config(tmp_path, seed=123)
    splits_a = datasets_module.load_ag_news_splits(config)
    splits_b = datasets_module.load_ag_news_splits(config)
    assert list(splits_a.validation["text"]) == list(splits_b.validation["text"])
    assert list(splits_a.train["text"]) == list(splits_b.train["text"])


def test_labels_remain_in_expected_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        datasets_module,
        "load_dataset",
        lambda **_: _fake_ag_news_dataset(),
    )
    config = _build_config(tmp_path, seed=19, debug_subset_size=5)
    splits = datasets_module.load_ag_news_splits(config)
    for split in splits.as_dict().values():
        labels = [int(value) for value in split["label"]]
        assert all(0 <= label <= 3 for label in labels)
