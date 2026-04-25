from __future__ import annotations

from pathlib import Path

import pytest
import torch
from datasets import Dataset

from bayes_gp_llmops.data.config import DataPipelineConfig
from bayes_gp_llmops.data.datamodule import build_split_dataloaders, collate_tokenized_batch
from bayes_gp_llmops.data.datasets import DatasetSplits
from bayes_gp_llmops.data.preprocessing import validate_label
from bayes_gp_llmops.data.tokenizer import (
    load_tokenizer,
    resolve_tokenizer_artifacts,
    train_and_save_tokenizer,
)


def _sample_corpus() -> list[str]:
    return [
        "World markets rise on policy announcement",
        "Technology firms release quarterly earnings",
        "Sports team secures championship title",
        "Health officials update public guidance",
    ]


def _build_config(tmp_path: Path) -> DataPipelineConfig:
    return DataPipelineConfig.model_validate(
        {
            "dataset": {
                "name": "ag_news",
                "config": None,
                "text_field": "text",
                "label_field": "label",
                "validation_split_ratio": 0.1,
                "random_seed": 42,
                "debug_subset_size": None,
            },
            "tokenizer": {
                "vocab_size": 128,
                "min_frequency": 1,
                "max_sequence_length": 12,
            },
            "dataloader": {
                "batch_size": 2,
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


def test_tokenizer_save_and_load_round_trip(tmp_path: Path) -> None:
    corpus = _sample_corpus()
    output_dir = tmp_path / "tokenizer"
    artifacts = train_and_save_tokenizer(
        corpus,
        output_dir=output_dir,
        vocab_size=128,
        min_frequency=1,
        max_sequence_length=8,
        corpus_source="unit-test",
    )
    resolved = resolve_tokenizer_artifacts(output_dir)
    assert artifacts == resolved
    assert resolved.tokenizer_json.exists()
    assert resolved.tokenizer_config_json.exists()
    assert resolved.special_tokens_map_json.exists()
    assert resolved.metadata_json.exists()

    tokenizer = load_tokenizer(output_dir)
    encoded = tokenizer.encode("World markets rise after earnings report in technology sector")
    assert len(encoded.ids) <= 8


def test_validate_label_rejects_out_of_range_values() -> None:
    assert validate_label(2, num_classes=4) == 2
    with pytest.raises(ValueError):
        validate_label(4, num_classes=4)


def test_collate_function_pads_batches() -> None:
    batch = collate_tokenized_batch(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": 0},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": 1},
        ],
        pad_token_id=0,
    )
    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert batch["input_ids"][1, 2].item() == 0
    assert batch["attention_mask"][1, 2].item() == 0
    assert batch["labels"].dtype == torch.long


def test_dataloader_batch_contains_required_tensors(tmp_path: Path) -> None:
    split = Dataset.from_dict(
        {
            "text": [
                "Business outlook improves after earnings growth",
                "Sports coverage highlights the final game",
                "Global policy update affects market expectations",
            ],
            "label": [2, 1, 0],
        }
    )
    splits = DatasetSplits(
        train=split,
        validation=split.select([0, 1]),
        test=split.select([1, 2]),
    )
    config = _build_config(tmp_path)
    train_and_save_tokenizer(
        _sample_corpus(),
        output_dir=config.paths.tokenizer_dir,
        vocab_size=config.tokenizer.vocab_size,
        min_frequency=config.tokenizer.min_frequency,
        max_sequence_length=config.tokenizer.max_sequence_length,
        corpus_source="unit-test",
    )
    tokenizer = load_tokenizer(config.paths.tokenizer_dir)
    loaders = build_split_dataloaders(splits, tokenizer=tokenizer, config=config)
    batch = next(iter(loaders.train))

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["input_ids"].shape == batch["attention_mask"].shape
