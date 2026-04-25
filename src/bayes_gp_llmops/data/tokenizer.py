from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


@dataclass(frozen=True)
class TokenizerArtifacts:
    """Tokenizer artifact file locations."""

    tokenizer_json: Path
    tokenizer_config_json: Path
    special_tokens_map_json: Path
    metadata_json: Path


def train_bpe_tokenizer(
    corpus: Iterable[str],
    *,
    vocab_size: int,
    min_frequency: int,
    max_sequence_length: int,
) -> Tokenizer:
    """Train a BPE tokenizer with a deterministic special-token scheme."""

    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive.")

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN],
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    bos_id = _required_token_id(tokenizer, BOS_TOKEN)
    eos_id = _required_token_id(tokenizer, EOS_TOKEN)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",
        special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
    )
    tokenizer.enable_truncation(max_length=max_sequence_length)
    return tokenizer


def train_and_save_tokenizer(
    corpus: Iterable[str],
    *,
    output_dir: Path,
    vocab_size: int,
    min_frequency: int,
    max_sequence_length: int,
    corpus_source: str,
) -> TokenizerArtifacts:
    """Train a tokenizer from corpus and persist all artifacts."""

    tokenizer = train_bpe_tokenizer(
        corpus,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_sequence_length=max_sequence_length,
    )
    return save_tokenizer_artifacts(
        tokenizer,
        output_dir=output_dir,
        corpus_source=corpus_source,
        requested_vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_sequence_length=max_sequence_length,
    )


def save_tokenizer_artifacts(
    tokenizer: Tokenizer,
    *,
    output_dir: Path,
    corpus_source: str,
    requested_vocab_size: int,
    min_frequency: int,
    max_sequence_length: int,
) -> TokenizerArtifacts:
    """Persist tokenizer artifacts and metadata for reproducible reuse."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = resolve_tokenizer_artifacts(output_dir)
    tokenizer.save(str(artifacts.tokenizer_json))

    tokenizer_config: dict[str, Any] = {
        "model_type": "bpe",
        "min_frequency": min_frequency,
        "requested_vocab_size": requested_vocab_size,
        "max_sequence_length": max_sequence_length,
        "special_tokens": {
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "bos_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
        },
    }
    _write_json(artifacts.tokenizer_config_json, tokenizer_config)

    special_tokens_map: dict[str, str] = {
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
    }
    _write_json(artifacts.special_tokens_map_json, special_tokens_map)

    metadata: dict[str, Any] = {
        "corpus_source": corpus_source,
        "requested_vocab_size": requested_vocab_size,
        "trained_vocab_size": tokenizer.get_vocab_size(),
        "min_token_frequency": min_frequency,
        "max_sequence_length": max_sequence_length,
    }
    _write_json(artifacts.metadata_json, metadata)
    return artifacts


def load_tokenizer(output_dir: Path, *, max_sequence_length: int | None = None) -> Tokenizer:
    """Load a tokenizer and restore truncation settings."""

    artifacts = resolve_tokenizer_artifacts(output_dir)
    if not artifacts.tokenizer_json.exists():
        raise FileNotFoundError(f"Tokenizer artifact not found: {artifacts.tokenizer_json}")

    tokenizer = Tokenizer.from_file(str(artifacts.tokenizer_json))
    configured_max_length = max_sequence_length
    if configured_max_length is None and artifacts.tokenizer_config_json.exists():
        config_payload = _read_json(artifacts.tokenizer_config_json)
        value = config_payload.get("max_sequence_length")
        if isinstance(value, int):
            configured_max_length = value
    if configured_max_length is not None:
        tokenizer.enable_truncation(max_length=configured_max_length)
    return tokenizer


def tokenizer_artifacts_exist(output_dir: Path) -> bool:
    """Return whether all expected tokenizer artifacts exist."""

    artifacts = resolve_tokenizer_artifacts(output_dir)
    return (
        artifacts.tokenizer_json.exists()
        and artifacts.tokenizer_config_json.exists()
        and artifacts.special_tokens_map_json.exists()
        and artifacts.metadata_json.exists()
    )


def resolve_tokenizer_artifacts(output_dir: Path) -> TokenizerArtifacts:
    """Resolve artifact paths for a tokenizer directory."""

    return TokenizerArtifacts(
        tokenizer_json=output_dir / "tokenizer.json",
        tokenizer_config_json=output_dir / "tokenizer_config.json",
        special_tokens_map_json=output_dir / "special_tokens_map.json",
        metadata_json=output_dir / "tokenizer_metadata.json",
    )


def _required_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Required token '{token}' was not added to the vocabulary.")
    return int(token_id)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded
