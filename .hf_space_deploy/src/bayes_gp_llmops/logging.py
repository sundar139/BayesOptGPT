from __future__ import annotations

import logging


def configure_logging(level_name: str) -> None:
    """Configure application logging once with consistent formatting."""

    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
