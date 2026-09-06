from __future__ import annotations

import gzip
import inspect
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v3 as v3,
)
from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v4 as v4,
)


SCRIPT = Path(
    "training/evaluate_options_led_volatility_expansion_premium_relay_economics_v4.py"
)


def test_v4_preserves_the_frozen_v3_accounting_engine() -> None:
    assert inspect.getsource(v4.simulate) == inspect.getsource(v3.simulate)
    assert inspect.getsource(v4.evaluate_primary) == inspect.getsource(v3.evaluate_primary)
    assert v4.BASE_COST == v3.BASE_COST
    assert v4.STRESS_COST == v3.STRESS_COST
    assert v4.LEVERAGE == v3.LEVERAGE
    assert v4.STAGES == v3.STAGES


def test_gzip_prefix_reader_consumes_rows_inside_the_open_stream(tmp_path: Path) -> None:
    source = tmp_path / "market.csv.gz"
    with gzip.open(source, "wt") as handle:
        handle.write("date,open\n")
        handle.write("2023-07-01T00:00:00Z,100\n")
        handle.write("2023-07-01T00:05:00Z,101\n")

    frame = v4._stream(
        source,
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2023-07-01T00:05:00Z"),
        ("date", "open"),
        "date",
        True,
    )

    assert frame["open"].tolist() == ["100", "101"]
    assert frame["date"].tolist() == [
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2023-07-01T00:05:00Z"),
    ]


def test_direct_cli_preflight_reaches_verifier_without_opening_outcomes() -> None:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [
            str(Path(".venv/bin/python")),
            "-B",
            str(SCRIPT),
            "--stage",
            "train",
            "--verify-only",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(completed.stdout) == {
        "stage": "train",
        "verified": True,
        "outcomes_opened": False,
    }
