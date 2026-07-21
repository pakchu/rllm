from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from training import freeze_chain_activity_comparator_clock as freeze


def test_repository_paths_never_fall_back_to_another_checkout() -> None:
    relative = freeze._repository_path("missing/local-only.csv")
    assert relative == freeze.REPOSITORY_ROOT / "missing/local-only.csv"
    assert not str(relative).startswith("/home/pakchu/rllm/missing")


def test_builder_never_imports_repository_research_modules() -> None:
    source = (freeze.REPOSITORY_ROOT / freeze.BUILDER).read_text(encoding="utf-8")
    tree = ast.parse(source)
    repository_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("training")
    ]
    assert repository_imports == []
    assert "importlib" not in source
    assert "strict_bar_backtest" not in source
    assert "ExecutionEngine" not in source


def test_schedule_rows_are_clock_only_and_nonoverlapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(freeze, "WINDOWS", ("sample",))
    dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="5min"))
    anchors = np.asarray([0, 2, 5], dtype=int)
    long_active = np.asarray([True, True, False])
    short_active = np.asarray([False, False, True])
    rows, hashes = freeze._schedule_rows(
        dates,
        anchors,
        long_active,
        short_active,
        hold_bars=2,
        windows={"sample": ("2023-01-01", "2023-01-02")},
    )
    assert len(rows) == 2
    assert [row["side"] for row in rows] == [1, -1]
    assert set(rows[0]) == set(freeze.CLOCK_COLUMNS)
    assert hashes == {
        "sample": "65358bc00bd1f7c7b6655d90c61a3bfb664c42879e382f700ce0bb2260f5f9df"
    }


def test_clock_validation_rejects_invalid_intervals() -> None:
    with pytest.raises(RuntimeError, match="interval is invalid"):
        freeze._validate_clock(
            [
                {
                    "window": "fit_2021",
                    "decision_time": "2021-01-01T00:00:00+00:00",
                    "entry_time": "2021-01-01T00:05:00+00:00",
                    "exit_time": "2021-01-01T00:05:00+00:00",
                    "side": 1,
                }
            ]
        )
