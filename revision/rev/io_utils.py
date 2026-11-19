"""Crash-safe file helpers.

* ``atomic_write_*``: write to a temporary file in the same directory, flush,
  fsync, then ``os.replace`` it over the target.  After a power cut the target
  is either the complete old file or the complete new file, never a partial one.
* ``EvalCache``: append-only JSON-lines log of objective evaluations.  A
  truncated last line (power cut during a write) is ignored when reloading.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


def _fsync_dir(path: Path) -> None:
    # Directory fsync makes the rename durable on POSIX; not available on Windows.
    if os.name != "nt":
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _json_default(o: Any):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, default=_json_default))


def atomic_joblib_dump(obj: Any, path: Path) -> None:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    atomic_write_bytes(path, buf.getvalue())


def read_json(path: Path) -> Optional[Any]:
    """Return parsed JSON or None if the file is missing or unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


class EvalCache:
    """Persistent cache of objective values, keyed by a canonical parameter string.

    Because every tuner is seeded and deterministic given the sequence of
    objective values, replaying a tuner against this cache after a crash
    reproduces exactly the same search path, and the already-evaluated points
    cost nothing.  The search therefore resumes where it stopped.
    """

    def __init__(self, path: Path, fsync_every: int = 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync_every = max(1, int(fsync_every))
        self.values: Dict[str, float] = {}
        self._pending = 0
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self.values[rec["k"]] = float(rec["v"])
                    except (ValueError, KeyError, TypeError):
                        # truncated / corrupted line from an interrupted write
                        continue
        self.n_loaded = len(self.values)
        self._fh = open(self.path, "a", encoding="utf-8")

    def get(self, key: str) -> Optional[float]:
        return self.values.get(key)

    def put(self, key: str, value: float) -> None:
        self.values[key] = float(value)
        self._fh.write(json.dumps({"k": key, "v": float(value)}) + "\n")
        self._fh.flush()
        self._pending += 1
        if self._pending >= self.fsync_every:
            os.fsync(self._fh.fileno())
            self._pending = 0

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except (OSError, ValueError):
            pass
        try:
            self._fh.close()
        except OSError:
            pass
