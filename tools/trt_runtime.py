"""Helpers shared by the RaCo extractor and its benchmark driver.

These three lived in both ``raco_extract.py`` and ``benchmark_cusfm_extractors.py``
as byte-identical copies (only the docstrings differed), which is exactly how the
two files' engine-cache behaviour drifted apart. One owner each.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from cuda.bindings import runtime as cudart


def check_cuda(result: tuple[Any, ...] | Any, operation: str) -> tuple[Any, ...]:
    """Raise on a failed cuda-python runtime call and return its payload.

    Every ``cudart`` entry point returns ``(status, *values)``, so the status has
    to be stripped before the payload is usable. Silently dropping it turns a
    failed allocation into a null pointer used several lines later.
    """
    values: tuple[Any, ...] = result if isinstance(result, tuple) else (result,)
    if values[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {values[0]}")
    return values[1:]


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all at once.

    ``hashlib.file_digest`` (3.11+) does the chunked read itself; this project
    pins 3.12, so the hand-rolled loop it replaces was pure duplication.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    """Linearly interpolated percentile, matching ``np.percentile``'s default.

    ``fraction`` is in ``[0, 1]`` rather than percent, which is why this wrapper
    exists at all -- the callers all think in fractions.
    """
    return float(np.percentile(values, 100.0 * fraction))
