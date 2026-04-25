from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from .config import DataPipelineConfig


@dataclass(frozen=True)
class DatasetSplits:
    """Normalized train/validation/test split container."""

    train: Dataset
    validation: Dataset
    test: Dataset

    def as_dict(self) -> dict[str, Dataset]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def load_ag_news_splits(config: DataPipelineConfig) -> DatasetSplits:
    """Load AG News with deterministic splitting and optional debug subsetting."""

    cache_dir = _ensure_directory(config.paths.dataset_cache_dir)
    raw = load_dataset(
        path=config.dataset.name,
        name=config.dataset.config,
        cache_dir=str(cache_dir),
    )
    if not isinstance(raw, DatasetDict):
        raise TypeError("Expected a DatasetDict from Hugging Face load_dataset.")
    if "train" not in raw or "test" not in raw:
        raise KeyError("Expected both 'train' and 'test' splits in the dataset.")

    train_split = raw["train"]
    test_split = raw["test"]
    _validate_fields(train_split, config.dataset.text_field, config.dataset.label_field, "train")
    _validate_fields(test_split, config.dataset.text_field, config.dataset.label_field, "test")

    split_result = train_split.train_test_split(
        test_size=config.dataset.validation_split_ratio,
        seed=config.dataset.random_seed,
        shuffle=True,
    )
    if "train" not in split_result or "test" not in split_result:
        raise KeyError("Deterministic validation split creation failed.")

    normalized = DatasetSplits(
        train=_apply_debug_subset(split_result["train"], config.dataset.debug_subset_size),
        validation=_apply_debug_subset(split_result["test"], config.dataset.debug_subset_size),
        test=_apply_debug_subset(test_split, config.dataset.debug_subset_size),
    )
    return normalized


def summarize_split_sizes(splits: DatasetSplits) -> dict[str, int]:
    """Return row counts for each dataset split."""

    return {
        "train": len(splits.train),
        "validation": len(splits.validation),
        "test": len(splits.test),
    }


def iter_split_records(split: Dataset) -> Iterable[Mapping[str, object]]:
    """Yield records from a dataset split for downstream processing."""

    for row in split:
        if not isinstance(row, Mapping):
            raise TypeError("Each dataset row must be a mapping.")
        yield row


def _apply_debug_subset(split: Dataset, debug_subset_size: int | None) -> Dataset:
    if debug_subset_size is None:
        return split
    subset_size = min(debug_subset_size, len(split))
    return split.select(range(subset_size))


def _validate_fields(split: Dataset, text_field: str, label_field: str, split_name: str) -> None:
    available_columns = set(split.column_names)
    missing = [name for name in (text_field, label_field) if name not in available_columns]
    if missing:
        message = f"Missing expected columns {missing} in split '{split_name}'."
        raise KeyError(message)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
