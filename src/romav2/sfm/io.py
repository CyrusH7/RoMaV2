"""Atomic artifacts and stage/failure records for reconstructible SfM runs."""

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger(__name__)


def write_json(path, value):
    path = Path(path)
    # Serialize first so invalid/nonfinite values cannot leave a partial artifact.
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as f:
        temporary = Path(f.name)
        try:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunRecord:
    def __init__(self, output):
        self.output = Path(output)
        self.data = {
            "status": "running",
            "stage": "preflight",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "timings_seconds": {},
        }
        self.started = time.perf_counter()
        self.save()

    def save(self):
        write_json(self.output / "run.json", self.data)

    @contextmanager
    def stage(self, name):
        self.data["stage"] = name
        self.save()
        started = time.perf_counter()
        LOG.info("Starting %s", name)
        try:
            yield
        finally:
            self.data["timings_seconds"][name] = time.perf_counter() - started
            self.save()

    def finish(self, error=None):
        self.data["status"] = "failed" if error is not None else "complete"
        self.data["elapsed_seconds"] = time.perf_counter() - self.started
        if error is not None:
            self.data["error"] = {"type": type(error).__name__, "message": str(error)}
        self.save()
