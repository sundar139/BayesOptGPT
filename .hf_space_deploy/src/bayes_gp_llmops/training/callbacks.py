from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class EarlyStopping:
    """Patience-based early stopping for maximizing a monitored metric."""

    patience: int
    min_delta: float = 0.0
    best_value: float | None = None
    counter: int = 0

    def step(self, value: float) -> bool:
        if self.best_value is None or value > (self.best_value + self.min_delta):
            self.best_value = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


@dataclass(frozen=True)
class CheckpointPaths:
    """Checkpoint and metadata output paths."""

    best: Path
    latest: Path
    history: Path
    resolved_config: Path


class CheckpointManager:
    """Save and load checkpoint artifacts for model training."""

    def __init__(self, checkpoint_dir: Path) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.paths = CheckpointPaths(
            best=checkpoint_dir / "best.ckpt",
            latest=checkpoint_dir / "latest.ckpt",
            history=checkpoint_dir / "training_history.json",
            resolved_config=checkpoint_dir / "resolved_config.json",
        )
        self.best_metric: float | None = None

    def save_latest(self, state: dict[str, Any]) -> None:
        torch.save(state, self.paths.latest)

    def save_best(self, state: dict[str, Any], metric_value: float) -> bool:
        if self.best_metric is None or metric_value > self.best_metric:
            self.best_metric = metric_value
            torch.save(state, self.paths.best)
            return True
        return False

    def write_history(self, history: list[dict[str, Any]]) -> None:
        with self.paths.history.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def write_resolved_config(self, resolved_config: dict[str, Any]) -> None:
        with self.paths.resolved_config.open("w", encoding="utf-8") as handle:
            json.dump(resolved_config, handle, indent=2, sort_keys=True)
            handle.write("\n")
