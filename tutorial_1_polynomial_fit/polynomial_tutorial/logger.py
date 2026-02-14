import logging
import sys
from typing import Optional


def configure_logging(
    *,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> None:
    """
    Configure application-wide logging.

    Safe to call multiple times (idempotent).
    """
    if fmt is None:
        fmt = (
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s:%(lineno)d | %(message)s"
        )

    root = logging.getLogger()

    # Prevent duplicate handlers if called multiple times
    if root.handlers:
        return

    root.setLevel(level)

    formatter = logging.Formatter(fmt)

    # INFO and below -> stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda r: r.levelno <= logging.INFO)
    stdout_handler.setFormatter(formatter)

    # WARNING and above -> stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)
