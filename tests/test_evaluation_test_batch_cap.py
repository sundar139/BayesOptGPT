"""Tests for the max_test_batches_per_epoch cap logic in evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bayes_gp_llmops.training.config import TrainConfig


def _make_train_config(**kwargs: object) -> TrainConfig:
    base: dict[str, object] = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 1,
        "gradient_clip_norm": 1.0,
        "early_stopping_patience": 3,
        "checkpoint_dir": Path("artifacts/checkpoints"),
    }
    base.update(kwargs)
    return TrainConfig(**base)  # type: ignore[arg-type]


class TestTestBatchCapSelection:
    """The cap selection logic chooses test cap over validation cap when set."""

    @pytest.mark.parametrize(
        "max_test, max_val, expected",
        [
            (10, 5, 10),         # explicit test cap overrides val cap
            (None, 5, 5),        # test cap unset → fall back to val cap
            (None, None, None),  # both unset → no cap
        ],
    )
    def test_cap_selection_logic(
        self, max_test: int | None, max_val: int | None, expected: int | None
    ) -> None:
        """Replicate the cap selection logic from evaluation/pipeline.py."""
        max_test_batches_per_epoch = max_test
        max_validation_batches_per_epoch = max_val

        test_batch_cap = (
            max_test_batches_per_epoch
            if max_test_batches_per_epoch is not None
            else max_validation_batches_per_epoch
        )
        assert test_batch_cap == expected

    def test_train_config_has_max_test_batches_per_epoch_field(self) -> None:
        config = _make_train_config()
        assert hasattr(config, "max_test_batches_per_epoch")

    def test_train_config_test_field_defaults_to_none(self) -> None:
        config = _make_train_config()
        assert config.max_test_batches_per_epoch is None

    def test_train_config_test_field_accepts_int(self) -> None:
        config = _make_train_config(max_test_batches_per_epoch=50)
        assert config.max_test_batches_per_epoch == 50

    def test_pipeline_module_uses_max_test_batches(self) -> None:
        """The evaluation pipeline source must use max_test_batches_per_epoch."""
        import inspect

        import bayes_gp_llmops.evaluation.pipeline as pipeline_mod

        src = inspect.getsource(pipeline_mod)
        assert "max_test_batches_per_epoch" in src, (
            "evaluation/pipeline.py does not reference max_test_batches_per_epoch"
        )

    def test_pipeline_module_has_run_inference(self) -> None:
        import bayes_gp_llmops.evaluation.pipeline as pipeline_mod

        assert hasattr(pipeline_mod, "_run_inference"), (
            "evaluation/pipeline.py must expose _run_inference"
        )

    def test_pipeline_test_cap_separated_from_val_cap(self) -> None:
        """The pipeline source must pass test_batch_cap (not max_validation_batches_per_epoch)
        to the test split inference call."""
        import inspect

        import bayes_gp_llmops.evaluation.pipeline as pipeline_mod

        src = inspect.getsource(pipeline_mod)
        # Both cap variables must appear (not just one)
        assert "test_batch_cap" in src
        assert "max_validation_batches_per_epoch" in src
        # The test call must use the computed test_batch_cap
        assert "max_batches=test_batch_cap" in src
