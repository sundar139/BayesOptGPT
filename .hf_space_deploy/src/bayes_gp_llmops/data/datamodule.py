from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TypedDict, cast

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from .config import DataPipelineConfig
from .datasets import DatasetSplits
from .preprocessing import PreprocessingOptions, prepare_sample
from .tokenizer import PAD_TOKEN


class TokenizedSample(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: int


@dataclass(frozen=True)
class DataLoaders:
    """Grouped train/validation/test data loader objects."""

    train: DataLoader[dict[str, torch.Tensor]]
    validation: DataLoader[dict[str, torch.Tensor]]
    test: DataLoader[dict[str, torch.Tensor]]


class AGNewsTokenizedDataset(TorchDataset[TokenizedSample]):
    """PyTorch dataset wrapper for on-the-fly tokenization."""

    def __init__(
        self,
        split: Dataset,
        *,
        tokenizer: Tokenizer,
        text_field: str,
        label_field: str,
        max_sequence_length: int,
        num_classes: int = 4,
        preprocessing_options: PreprocessingOptions | None = None,
    ) -> None:
        self._split = split
        self._tokenizer = tokenizer
        self._text_field = text_field
        self._label_field = label_field
        self._max_sequence_length = max_sequence_length
        self._num_classes = num_classes
        self._preprocessing_options = preprocessing_options or PreprocessingOptions()

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, index: int) -> TokenizedSample:
        record = cast(Mapping[str, object], self._split[index])
        sample = prepare_sample(
            record,
            text_field=self._text_field,
            label_field=self._label_field,
            num_classes=self._num_classes,
            options=self._preprocessing_options,
        )
        encoding = self._tokenizer.encode(sample["text"])
        input_ids = encoding.ids[: self._max_sequence_length]
        attention_mask = encoding.attention_mask[: self._max_sequence_length]
        if not attention_mask:
            attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": sample["label"],
        }


def collate_tokenized_batch(
    batch: Sequence[TokenizedSample],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pad variable-length tokenized samples into a tensor batch."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    max_length = max(len(sample["input_ids"]) for sample in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.empty(batch_size, dtype=torch.long)

    for row_index, sample in enumerate(batch):
        sequence_length = len(sample["input_ids"])
        input_ids[row_index, :sequence_length] = torch.tensor(
            sample["input_ids"],
            dtype=torch.long,
        )
        attention_mask[row_index, :sequence_length] = torch.tensor(
            sample["attention_mask"],
            dtype=torch.long,
        )
        labels[row_index] = sample["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_split_dataloaders(
    splits: DatasetSplits,
    *,
    tokenizer: Tokenizer,
    config: DataPipelineConfig,
    num_classes: int = 4,
) -> DataLoaders:
    """Build train/validation/test data loaders from configured components."""

    train_dataset = AGNewsTokenizedDataset(
        splits.train,
        tokenizer=tokenizer,
        text_field=config.dataset.text_field,
        label_field=config.dataset.label_field,
        max_sequence_length=config.tokenizer.max_sequence_length,
        num_classes=num_classes,
    )
    validation_dataset = AGNewsTokenizedDataset(
        splits.validation,
        tokenizer=tokenizer,
        text_field=config.dataset.text_field,
        label_field=config.dataset.label_field,
        max_sequence_length=config.tokenizer.max_sequence_length,
        num_classes=num_classes,
    )
    test_dataset = AGNewsTokenizedDataset(
        splits.test,
        tokenizer=tokenizer,
        text_field=config.dataset.text_field,
        label_field=config.dataset.label_field,
        max_sequence_length=config.tokenizer.max_sequence_length,
        num_classes=num_classes,
    )
    pad_token_id = _resolve_pad_token_id(tokenizer)
    collate_fn = partial(collate_tokenized_batch, pad_token_id=pad_token_id)

    train_loader = cast(
        DataLoader[dict[str, torch.Tensor]],
        DataLoader(
            train_dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            pin_memory=config.dataloader.pin_memory,
            collate_fn=collate_fn,
        ),
    )
    validation_loader = cast(
        DataLoader[dict[str, torch.Tensor]],
        DataLoader(
            validation_dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=False,
            num_workers=config.dataloader.num_workers,
            pin_memory=config.dataloader.pin_memory,
            collate_fn=collate_fn,
        ),
    )
    test_loader = cast(
        DataLoader[dict[str, torch.Tensor]],
        DataLoader(
            test_dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=False,
            num_workers=config.dataloader.num_workers,
            pin_memory=config.dataloader.pin_memory,
            collate_fn=collate_fn,
        ),
    )
    return DataLoaders(train=train_loader, validation=validation_loader, test=test_loader)


def _resolve_pad_token_id(tokenizer: Tokenizer) -> int:
    pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
    if pad_token_id is None:
        raise ValueError(f"Tokenizer must include '{PAD_TOKEN}' token.")
    return int(pad_token_id)
