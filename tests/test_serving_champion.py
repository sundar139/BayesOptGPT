"""Tests for champion selection policy, manifest I/O, and candidate loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayes_gp_llmops.serving.champion import (
    CandidateMetrics,
    build_champion_manifest,
    load_candidates_from_tuning_dir,
    load_champion_manifest,
    select_champion,
    write_champion_manifest,
)


def _make_candidate(
    trial_number: int,
    macro_f1: float,
    checkpoint_name: str = "best.ckpt",
) -> CandidateMetrics:
    return CandidateMetrics(
        study_name="test-study",
        trial_number=trial_number,
        checkpoint_path=Path(f"artifacts/trial_{trial_number:04d}/{checkpoint_name}"),
        validation_macro_f1=macro_f1,
        validation_nll=0.5,
        validation_brier=0.1,
        validation_ece=0.05,
    )


class TestSelectChampion:
    def test_single_candidate_returned(self) -> None:
        candidate = _make_candidate(0, 0.80)
        assert select_champion([candidate]) is candidate

    def test_maximizes_macro_f1(self) -> None:
        candidates = [
            _make_candidate(0, 0.70),
            _make_candidate(1, 0.90),
            _make_candidate(2, 0.85),
        ]
        champion = select_champion(candidates)
        assert champion.trial_number == 1
        assert champion.validation_macro_f1 == pytest.approx(0.90)

    def test_tie_breaking_by_trial_number(self) -> None:
        candidates = [
            _make_candidate(5, 0.90),
            _make_candidate(2, 0.90),
            _make_candidate(8, 0.90),
        ]
        champion = select_champion(candidates)
        assert champion.trial_number == 2

    def test_tie_breaking_by_checkpoint_path(self) -> None:
        # Same trial_number is unrealistic but the policy must still be deterministic.
        c1 = CandidateMetrics(
            study_name="s",
            trial_number=3,
            checkpoint_path=Path("z/best.ckpt"),
            validation_macro_f1=0.88,
        )
        c2 = CandidateMetrics(
            study_name="s",
            trial_number=3,
            checkpoint_path=Path("a/best.ckpt"),
            validation_macro_f1=0.88,
        )
        champion = select_champion([c1, c2])
        # Lexicographic: "a/best.ckpt" < "z/best.ckpt"
        assert str(champion.checkpoint_path) == str(c2.checkpoint_path)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            select_champion([])

    def test_selection_is_deterministic(self) -> None:
        import random

        candidates = [_make_candidate(i, round(random.uniform(0.5, 0.95), 4)) for i in range(10)]
        shuffled = candidates[:]
        random.shuffle(shuffled)
        assert select_champion(candidates).trial_number == select_champion(shuffled).trial_number


class TestChampionManifest:
    def test_build_manifest_fields(self) -> None:
        candidate = _make_candidate(3, 0.91)
        manifest = build_champion_manifest(candidate)
        assert manifest.trial_number == 3
        assert manifest.study_name == "test-study"
        assert manifest.selected_metrics["validation_macro_f1"] == pytest.approx(0.91)
        assert manifest.selected_metrics["validation_nll"] == pytest.approx(0.5)
        assert "timestamp_utc" in manifest.model_dump()
        assert manifest.schema_version == "1.0"
        assert "maximize" in manifest.selection_policy.lower()

    def test_write_and_load_round_trip(self, tmp_path: Path) -> None:
        candidate = _make_candidate(7, 0.85)
        manifest = build_champion_manifest(candidate)
        path = write_champion_manifest(manifest, tmp_path)

        assert path.exists()
        assert path.name == "champion_manifest.json"

        loaded = load_champion_manifest(path)
        assert loaded.trial_number == 7
        assert loaded.study_name == "test-study"
        assert loaded.selected_metrics["validation_macro_f1"] == pytest.approx(0.85)
        assert loaded.selection_policy == manifest.selection_policy

    def test_manifest_json_is_valid(self, tmp_path: Path) -> None:
        candidate = _make_candidate(1, 0.77)
        manifest = build_champion_manifest(candidate)
        write_champion_manifest(manifest, tmp_path)

        raw = (tmp_path / "champion_manifest.json").read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed["trial_number"] == 1
        assert "timestamp_utc" in parsed

    def test_load_manifest_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_champion_manifest(tmp_path / "nonexistent.json")

    def test_write_manifest_sanitizes_absolute_checkpoint_path(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "checkpoints" / "best.ckpt"
        candidate = CandidateMetrics(
            study_name="test-study",
            trial_number=10,
            checkpoint_path=checkpoint_path,
            validation_macro_f1=0.90,
        )

        manifest = build_champion_manifest(candidate)
        manifest_path = write_champion_manifest(manifest, tmp_path)
        written = load_champion_manifest(manifest_path)
        assert written.checkpoint_path == "best.ckpt"


class TestLoadCandidatesFromTuningDir:
    def _build_trial_dir(
        self,
        trials_dir: Path,
        trial_number: int,
        macro_f1: float,
        with_calibrated: bool = False,
        with_resolved_config: bool = True,
    ) -> Path:
        trial_dir = trials_dir / f"trial_{trial_number:04d}"
        ckpt_dir = trial_dir / "checkpoints"
        eval_dir = trial_dir / "evaluation"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        (ckpt_dir / "best.ckpt").write_bytes(b"fake-checkpoint")

        metrics = {"macro_f1": macro_f1, "nll": 0.4, "brier_score": 0.08, "ece": 0.03}
        (eval_dir / "metrics_validation.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )

        if with_calibrated:
            cal_metrics = {"macro_f1": macro_f1 + 0.01, "nll": 0.38}
            (eval_dir / "metrics_validation_calibrated.json").write_text(
                json.dumps(cal_metrics), encoding="utf-8"
            )

        if with_resolved_config:
            config = {"model": {"vocab_size": 512}, "data": {}, "training": {}}
            (ckpt_dir / "resolved_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
        return trial_dir

    def test_loads_valid_trials(self, tmp_path: Path) -> None:
        trials_dir = tmp_path / "trials"
        self._build_trial_dir(trials_dir, 0, 0.80)
        self._build_trial_dir(trials_dir, 1, 0.88)

        candidates = load_candidates_from_tuning_dir(tmp_path, study_name="unit-study")
        assert len(candidates) == 2
        trial_numbers = {c.trial_number for c in candidates}
        assert trial_numbers == {0, 1}

    def test_selects_best_candidate(self, tmp_path: Path) -> None:
        trials_dir = tmp_path / "trials"
        self._build_trial_dir(trials_dir, 0, 0.70)
        self._build_trial_dir(trials_dir, 1, 0.92)

        candidates = load_candidates_from_tuning_dir(tmp_path, study_name="unit-study")
        champion = select_champion(candidates)
        assert champion.trial_number == 1

    def test_skips_trial_without_checkpoint(self, tmp_path: Path) -> None:
        trials_dir = tmp_path / "trials"
        self._build_trial_dir(trials_dir, 0, 0.80)

        # Trial 1 has no checkpoint
        bad_dir = trials_dir / "trial_0001"
        eval_dir = bad_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "metrics_validation.json").write_text(
            json.dumps({"macro_f1": 0.95}), encoding="utf-8"
        )

        candidates = load_candidates_from_tuning_dir(tmp_path, study_name="unit-study")
        assert len(candidates) == 1
        assert candidates[0].trial_number == 0

    def test_skips_trial_without_metrics(self, tmp_path: Path) -> None:
        trials_dir = tmp_path / "trials"
        self._build_trial_dir(trials_dir, 0, 0.80)

        bad_dir = trials_dir / "trial_0001" / "checkpoints"
        bad_dir.mkdir(parents=True, exist_ok=True)
        (bad_dir / "best.ckpt").write_bytes(b"ckpt")

        candidates = load_candidates_from_tuning_dir(tmp_path, study_name="unit-study")
        assert len(candidates) == 1

    def test_reads_calibrated_metrics_when_present(self, tmp_path: Path) -> None:
        trials_dir = tmp_path / "trials"
        self._build_trial_dir(trials_dir, 0, 0.80, with_calibrated=True)

        candidates = load_candidates_from_tuning_dir(tmp_path, study_name="unit-study")
        assert candidates[0].validation_macro_f1_calibrated is not None
        assert candidates[0].validation_macro_f1_calibrated == pytest.approx(0.81)

    def test_missing_trials_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Trials directory"):
            load_candidates_from_tuning_dir(tmp_path, study_name="unit-study")

    def test_no_valid_trials_raises(self, tmp_path: Path) -> None:
        (tmp_path / "trials").mkdir()
        with pytest.raises(ValueError, match="No valid candidates"):
            load_candidates_from_tuning_dir(tmp_path, study_name="unit-study")

    def test_config_snapshot_populated(self, tmp_path: Path) -> None:
        trials_dir = tmp_path / "trials"
        self._build_trial_dir(trials_dir, 0, 0.80, with_resolved_config=True)

        candidates = load_candidates_from_tuning_dir(tmp_path, study_name="unit-study")
        assert "model" in candidates[0].config_snapshot
