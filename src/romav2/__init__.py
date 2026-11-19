"""RoMaV2 public API; load neural inference dependencies only on demand."""

import logging as _logging
from typing import TYPE_CHECKING

from .logging import configure_logger as configure_logger
from .logging import logger as _logger

if TYPE_CHECKING:
    from .romav2 import RoMaV2 as RoMaV2

if not any(not isinstance(h, _logging.NullHandler) for h in _logger.handlers):
    configure_logger()

__all__ = ["RoMaV2", "configure_logger"]


def __getattr__(name):
    if name == "RoMaV2":
        from .romav2 import RoMaV2

        globals()[name] = RoMaV2
        return RoMaV2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
