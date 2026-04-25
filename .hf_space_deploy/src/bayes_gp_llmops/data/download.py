from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from datasets import Dataset

from bayes_gp_llmops.config import get_settings

from .config import DataPipelineConfig, load_data_config, to_serializable_config
from .datasets import iter_split_records, load_ag_news_splits, summarize_split_sizes
from .preprocessing import PreprocessingOptions, iter_text_corpus
from .tokenizer import (
    load_tokenizer,
    resolve_tokenizer_artifacts,
    tokenizer_artifacts_exist,
    train_and_save_tokenizer,
)

LOGGER = logging.getLogger("bayes_gp_llmops.data.download")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_data_config(args.config)
    _configure_logging()
    _ensure_output_directories(config)

    if args.dry_run:
        _print_dry_run_summary(config, args.train_tokenizer)
        return 0

    splits = load_ag_news_splits(config)
    split_sizes = summarize_split_sizes(splits)
    LOGGER.info("Dataset loaded with split sizes: %s", split_sizes)

    if args.train_tokenizer:
        _ensure_tokenizer(config, splits.train, retrain=args.retrain_tokenizer)

    print(f"dataset_cache_dir={config.paths.dataset_cache_dir}")
    print(f"processed_cache_dir={config.paths.processed_cache_dir}")
    print(f"tokenizer_dir={config.paths.tokenizer_dir}")
    print(f"split_sizes={split_sizes}")
    return 0


def _ensure_tokenizer(config: DataPipelineConfig, train_split: Dataset, *, retrain: bool) -> None:
    tokenizer_dir = config.paths.tokenizer_dir
    artifacts = resolve_tokenizer_artifacts(tokenizer_dir)
    if tokenizer_artifacts_exist(tokenizer_dir) and not retrain:
        load_tokenizer(
            tokenizer_dir,
            max_sequence_length=config.tokenizer.max_sequence_length,
        )
        LOGGER.info("Tokenizer artifacts already exist at %s; reuse enabled.", tokenizer_dir)
        LOGGER.info("Tokenizer file: %s", artifacts.tokenizer_json)
        return

    records = iter_split_records(train_split)
    corpus = iter_text_corpus(
        records,
        text_field=config.dataset.text_field,
        options=PreprocessingOptions(),
    )
    written = train_and_save_tokenizer(
        corpus,
        output_dir=tokenizer_dir,
        vocab_size=config.tokenizer.vocab_size,
        min_frequency=config.tokenizer.min_frequency,
        max_sequence_length=config.tokenizer.max_sequence_length,
        corpus_source=f"{config.dataset.name}:train",
    )
    LOGGER.info("Tokenizer artifacts written to %s", tokenizer_dir)
    LOGGER.info("tokenizer_json=%s", written.tokenizer_json)
    LOGGER.info("tokenizer_config_json=%s", written.tokenizer_config_json)
    LOGGER.info("special_tokens_map_json=%s", written.special_tokens_map_json)
    LOGGER.info("metadata_json=%s", written.metadata_json)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch AG News data and prepare tokenizer artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data configuration YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print configuration without downloading or training tokenizer.",
    )
    parser.add_argument(
        "--train-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train tokenizer artifacts if they do not already exist.",
    )
    parser.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="Force retraining and overwrite tokenizer artifacts.",
    )
    return parser.parse_args(argv)


def _configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _ensure_output_directories(config: DataPipelineConfig) -> None:
    config.paths.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    config.paths.processed_cache_dir.mkdir(parents=True, exist_ok=True)
    config.paths.tokenizer_dir.mkdir(parents=True, exist_ok=True)


def _print_dry_run_summary(config: DataPipelineConfig, train_tokenizer: bool) -> None:
    payload = to_serializable_config(config)
    print("dry_run=true")
    print(f"config={payload}")
    print(f"train_tokenizer={train_tokenizer}")
