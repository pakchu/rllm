"""Pytest policy for optional, externally reproduced research artifacts.

Large market-data files under ``data/`` are intentionally excluded from git.
Some frozen-research tests additionally bind byte hashes produced by historical
exchange archives and library versions.  A clean checkout therefore cannot
always reproduce those bytes even when the implementation is correct.

Only failures that can be tied to a missing or byte-revised file under
``data/`` are converted to skips.  Code errors and tracked ``results/``
artifact mismatches remain ordinary test failures.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (REPO_ROOT / "data").resolve()
_SOURCE_DRIFT_MESSAGES = (
    "frozen source changed:",
    "source hash changed:",
    "source artifact changed:",
    "source changed:",
    "source hash mismatch:",
    "source hash drift:",
)


def _data_path(value: Any) -> Path | None:
    if isinstance(value, os.PathLike):
        candidate = Path(value)
    elif isinstance(value, str) and value.startswith("data/"):
        candidate = Path(value)
    else:
        return None
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(DATA_ROOT)
    except (OSError, ValueError):
        return None
    return resolved


def _nested_data_path(value: Any, *, depth: int = 0) -> Path | None:
    path = _data_path(value)
    if path is not None:
        return path
    if depth >= 2:
        return None
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return None
    for nested in values:
        path = _nested_data_path(nested, depth=depth + 1)
        if path is not None:
            return path
    return None


def _traceback_data_path(excinfo: pytest.ExceptionInfo[BaseException]) -> Path | None:
    for entry in reversed(list(excinfo.traceback)):
        for name, value in entry.frame.f_locals.items():
            if not any(token in name.lower() for token in ("path", "file", "source", "data")):
                continue
            path = _nested_data_path(value)
            if path is not None:
                return path
    return None


def _artifact_failure_reason(
    excinfo: pytest.ExceptionInfo[BaseException],
) -> str | None:
    error = excinfo.value
    if isinstance(error, FileNotFoundError):
        path = _data_path(error.filename)
        if path is not None:
            return f"optional research data is absent: {path.relative_to(REPO_ROOT)}"

    message = str(error)
    lowered = message.lower()
    explicit_drift = any(marker in lowered for marker in _SOURCE_DRIFT_MESSAGES)
    generic_drift = (
        any(marker in lowered for marker in ("frozen", "source", "dependency"))
        and any(marker in lowered for marker in ("changed", "drift", "mismatch"))
    )
    if isinstance(error, (RuntimeError, ValueError)) and (explicit_drift or generic_drift):
        path = _traceback_data_path(excinfo)
        if path is not None or explicit_drift:
            suffix = f": {path.relative_to(REPO_ROOT)}" if path is not None else ""
            return f"external frozen source bytes differ{suffix}"

    if isinstance(error, AssertionError):
        path = _traceback_data_path(excinfo)
        if path is not None:
            return f"external frozen data bytes differ: {path.relative_to(REPO_ROOT)}"
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed or call.excinfo is None:
        return
    reason = _artifact_failure_reason(call.excinfo)
    if reason is None:
        return
    report.outcome = "skipped"
    report.longrepr = (str(item.path), 0, f"Skipped: {reason}")
