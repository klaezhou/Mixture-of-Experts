from __future__ import annotations
import os
import datetime
from codename import codename
import warnings
from pathlib import Path
try:
    from tensorboardX import SummaryWriter as _TensorboardSummaryWriter
except Exception:  # pragma: no cover - optional dependency
    _TensorboardSummaryWriter = None

if 'Timetxt' not in locals() and 'Timetxt' not in globals():
    Timetxt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
else:
    Timetxt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    warnings.warn("Timetxt has been updated", stacklevel=2)

if 'runcode' not in locals() and 'runcode' not in globals():
    runcode = codename(separator='-')
else:
    runcode = codename(separator='-')
    warnings.warn("runcode has been updated", stacklevel=2)


def get_logs_root(default: str = "logs") -> str:
    """Return the TensorBoard logs root, honoring the ``LOGS_ROOT`` env var."""

    candidate = os.environ.get("LOGS_ROOT", default)
    if not candidate:
        return default
    return candidate


class NullSummaryWriter:
    """Fallback writer that drops all events when TensorBoard is unavailable."""

    def __init__(self, logdir: str | None = None) -> None:
        self.logdir = logdir

    def add_scalar(self, *args: object, **kwargs: object) -> None:  # noqa: D401
        return None

    def add_text(self, *args: object, **kwargs: object) -> None:
        return None

    def add_figure(self, *args: object, **kwargs: object) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def create_summary_writer(log_dir: Path | str) -> object:
    """Instantiate a TensorBoard writer or a no-op fallback.

    If tensorboardX is missing or cannot create its multiprocessing queue (common
    in restricted sandboxes), a ``NullSummaryWriter`` is returned instead so
    training can continue without telemetry.
    """

    path_str = str(log_dir)

    if _TensorboardSummaryWriter is None:
        warnings.warn(
            "tensorboardX not available; TensorBoard logging disabled.",
            stacklevel=2,
        )
        return NullSummaryWriter(path_str)

    try:
        return _TensorboardSummaryWriter(logdir=path_str)
    except Exception as exc:  # pragma: no cover - runtime dependent
        warnings.warn(
            f"TensorBoard writer unavailable ({exc!s}); falling back to in-memory logging.",
            stacklevel=2,
        )
        try:
            return _TensorboardSummaryWriter(logdir=path_str, write_to_disk=False)
        except Exception:
            return NullSummaryWriter(path_str)