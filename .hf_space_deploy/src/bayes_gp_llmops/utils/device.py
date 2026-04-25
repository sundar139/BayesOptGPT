from __future__ import annotations

import torch


def resolve_device(preference: str) -> torch.device:
    """Resolve an execution device from preference while handling availability."""

    normalized = preference.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device preference requested but CUDA is unavailable.")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device preference: {preference}")
