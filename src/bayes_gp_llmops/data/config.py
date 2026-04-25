from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Dataset loading and split controls."""

    name: str = Field(default="ag_news")
    config: str | None = Field(default=None)
    text_field: str = Field(default="text")
    label_field: str = Field(default="label")
    validation_split_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
    random_seed: int = Field(default=42)
    debug_subset_size: int | None = Field(default=None, ge=1)


class TokenizerConfig(BaseModel):
    """Tokenizer training parameters."""

    vocab_size: int = Field(default=16000, ge=128)
    min_frequency: int = Field(default=2, ge=1)
    max_sequence_length: int = Field(default=256, ge=8)


class DataLoaderConfig(BaseModel):
    """Data loader parameters for training pipelines."""

    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = Field(default=False)


class PathConfig(BaseModel):
    """Filesystem locations for cached and generated artifacts."""

    dataset_cache_dir: Path = Field(default=Path("data/hf_cache"))
    processed_cache_dir: Path = Field(default=Path("data/processed"))
    tokenizer_dir: Path = Field(default=Path("artifacts/tokenizer"))


class DataPipelineConfig(BaseModel):
    """Complete configuration for the data and tokenizer stack."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    paths: PathConfig = Field(default_factory=PathConfig)


def load_data_config(path: Path) -> DataPipelineConfig:
    """Load a data pipeline configuration file from YAML."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at the top level.")
    return DataPipelineConfig.model_validate(payload)


def to_serializable_config(config: DataPipelineConfig) -> dict[str, Any]:
    """Convert a validated data configuration into a JSON-serializable mapping."""

    return config.model_dump(mode="json")
