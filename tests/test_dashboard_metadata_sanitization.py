from __future__ import annotations

from pathlib import Path

from streamlit_app import _sanitize_metadata_payload


def test_sanitize_metadata_payload_strips_absolute_paths() -> None:
    payload: dict[str, object] = {
        "bundle_dir": "C:\\Users\\rohit\\Documents\\bundle",
        "checkpoint_path": "/tmp/work/checkpoints/best.ckpt",
        "nested": {
            "model_directory": "\\\\server\\share\\models\\champion",
            "files": [
                "/opt/app/artifacts/checkpoint.ckpt",
                "C:\\models\\tokenizer\\tokenizer.json",
                "artifacts/model/bundle/checksums.json",
            ],
        },
    }

    sanitized = _sanitize_metadata_payload(payload, root=Path("/workspace"))

    assert sanitized["bundle_dir"] == "bundle"
    assert sanitized["checkpoint_path"] == "best.ckpt"
    nested = sanitized["nested"]
    assert isinstance(nested, dict)
    assert nested["model_directory"] == "champion"

    files = nested["files"]
    assert isinstance(files, list)
    assert files[0] == "checkpoint.ckpt"
    assert files[1] == "tokenizer.json"
    assert files[2] == "artifacts/model/bundle/checksums.json"


def test_sanitize_metadata_payload_prefers_relative_paths_within_root() -> None:
    workspace_root = Path("/workspace/project")
    payload: dict[str, object] = {
        "bundle_dir": "/workspace/project/artifacts/model/bundle",
        "evaluation_dir": "/workspace/project/artifacts/evaluation_full_run",
    }

    sanitized = _sanitize_metadata_payload(payload, root=workspace_root)

    assert sanitized["bundle_dir"] == "artifacts/model/bundle"
    assert sanitized["evaluation_dir"] == "artifacts/evaluation_full_run"
