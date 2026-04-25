from __future__ import annotations

import ntpath
import re
from collections.abc import Mapping
from pathlib import Path

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
_PATH_KEY_TOKENS = (
    "path",
    "dir",
    "directory",
    "bundle",
    "checkpoint",
)


def sanitize_path_value(path_value: str | Path, *, root: Path | None = None) -> str:
    """Convert path-like values to safe display/storage strings.

    Policy:
    1. If a root is provided and the path can be represented relative to that root,
       return a normalized forward-slash relative path.
    2. For absolute paths outside the root, return only the basename.
    3. For relative paths, normalize separators to forward slashes.
    """

    raw_value = str(path_value).strip()
    if not raw_value:
        return raw_value

    normalized_root = root.resolve() if root is not None else None
    path = Path(raw_value)
    looks_absolute = _looks_absolute_path(raw_value)
    is_cross_platform_absolute = (
        looks_absolute
        and not path.is_absolute()
        and (
            bool(_WINDOWS_ABSOLUTE_PATH.match(raw_value))
            or raw_value.startswith("\\\\")
            or raw_value.startswith("//")
        )
    )

    can_attempt_relative = normalized_root is not None and not is_cross_platform_absolute
    if can_attempt_relative:
        try:
            candidate = path if path.is_absolute() else (normalized_root / path)
            relative = candidate.resolve().relative_to(normalized_root)
            return _normalize_relative_path(relative.as_posix())
        except (OSError, RuntimeError, ValueError):
            pass

    if looks_absolute:
        basename = _path_basename(raw_value)
        return basename if basename else "."

    return _normalize_relative_path(raw_value)


def sanitize_metadata_mapping(
    payload: Mapping[str, object],
    *,
    root: Path | None = None,
) -> dict[str, object]:
    """Recursively sanitize metadata maps for safe external exposure."""

    return {
        str(key): sanitize_metadata_value(value, key=str(key), root=root)
        for key, value in payload.items()
    }


def sanitize_metadata_value(
    value: object,
    *,
    key: str,
    root: Path | None = None,
) -> object:
    """Recursively sanitize metadata values using key-aware path handling."""

    if isinstance(value, Mapping):
        return {
            str(nested_key): sanitize_metadata_value(
                nested_value,
                key=str(nested_key),
                root=root,
            )
            for nested_key, nested_value in value.items()
        }

    if isinstance(value, list):
        return [
            sanitize_metadata_value(item, key=key, root=root)
            if not isinstance(item, str)
            else _sanitize_string_value(item, key=key, root=root)
            for item in value
        ]

    if isinstance(value, str):
        return _sanitize_string_value(value, key=key, root=root)

    return value


def _sanitize_string_value(value: str, *, key: str, root: Path | None) -> str:
    if _is_path_like_key(key) or _looks_absolute_path(value):
        return sanitize_path_value(value, root=root)
    return value


def _is_path_like_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _PATH_KEY_TOKENS)


def _looks_absolute_path(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    return (
        bool(_WINDOWS_ABSOLUTE_PATH.match(stripped))
        or stripped.startswith("/")
        or stripped.startswith("\\\\")
        or stripped.startswith("//")
    )


def _path_basename(path_value: str) -> str:
    normalized = path_value.replace("\\", "/")
    basename = normalized.rsplit("/", maxsplit=1)[-1]
    windows_basename = ntpath.basename(path_value)
    return windows_basename or basename


def _normalize_relative_path(path_value: str) -> str:
    normalized = path_value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized or "."
