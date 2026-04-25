from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict

_WHITESPACE_PATTERN = re.compile(r"\s+")

TruncationMode = Literal["none", "right"]


class PreparedSample(TypedDict):
    text: str
    label: int


@dataclass(frozen=True)
class PreprocessingOptions:
    """Configurable text preprocessing behavior."""

    normalize: bool = True
    max_characters: int | None = None
    truncation_mode: TruncationMode = "none"


def normalize_text(text: str) -> str:
    """Normalize text while preserving semantic content."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def truncate_text(
    text: str,
    *,
    max_characters: int | None,
    truncation_mode: TruncationMode,
) -> str:
    """Apply configurable truncation to text."""

    if max_characters is None:
        return text
    if max_characters < 1:
        raise ValueError("max_characters must be at least 1 when provided.")
    if truncation_mode == "none":
        return text
    if truncation_mode == "right":
        return text[:max_characters]
    raise ValueError(f"Unsupported truncation mode: {truncation_mode}")


def validate_label(raw_label: object, *, num_classes: int) -> int:
    """Validate and normalize labels to integer class indices."""

    if isinstance(raw_label, bool) or not isinstance(raw_label, int):
        raise TypeError(f"Label must be an integer class index. Received: {raw_label!r}")
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")
    if raw_label < 0 or raw_label >= num_classes:
        raise ValueError(f"Label {raw_label} is outside expected range [0, {num_classes - 1}].")
    return raw_label


def prepare_sample(
    record: Mapping[str, object],
    *,
    text_field: str,
    label_field: str,
    num_classes: int,
    options: PreprocessingOptions | None = None,
) -> PreparedSample:
    """Convert a raw record into a validated sample for tokenization."""

    resolved_options = options or PreprocessingOptions()
    if text_field not in record:
        raise KeyError(f"Record is missing required text field '{text_field}'.")
    if label_field not in record:
        raise KeyError(f"Record is missing required label field '{label_field}'.")

    raw_text = record[text_field]
    if not isinstance(raw_text, str):
        raise TypeError(f"Text field '{text_field}' must contain a string.")
    text_value = normalize_text(raw_text) if resolved_options.normalize else raw_text
    text_value = truncate_text(
        text_value,
        max_characters=resolved_options.max_characters,
        truncation_mode=resolved_options.truncation_mode,
    )
    label_value = validate_label(record[label_field], num_classes=num_classes)
    return {"text": text_value, "label": label_value}


def iter_text_corpus(
    records: Iterable[Mapping[str, object]],
    *,
    text_field: str,
    options: PreprocessingOptions | None = None,
) -> Iterable[str]:
    """Yield normalized text records suitable for tokenizer training."""

    resolved_options = options or PreprocessingOptions()
    for record in records:
        if text_field not in record:
            raise KeyError(f"Record is missing required text field '{text_field}'.")
        raw_text = record[text_field]
        if not isinstance(raw_text, str):
            raise TypeError(f"Text field '{text_field}' must contain a string.")
        text_value = normalize_text(raw_text) if resolved_options.normalize else raw_text
        yield truncate_text(
            text_value,
            max_characters=resolved_options.max_characters,
            truncation_mode=resolved_options.truncation_mode,
        )
