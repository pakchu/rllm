from __future__ import annotations

import gzip
from pathlib import Path

from training import build_cboe_volatility_term_rotation_support as support


def test_support_is_outcome_blind_and_passes_source_floor(tmp_path: Path) -> None:
    output = tmp_path / "support.json"
    ledger = tmp_path / "clocks.csv.gz"
    report = support.build_support(output_path=output, ledger_path=ledger)
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["market_rows_loaded"] == 0
    assert report["funding_rows_loaded"] == 0
    assert report["support_passed"] is True
    assert report["advance_to_stage1_outcomes"] is True
    assert all(report["support_checks"].values())
    assert report["clocks"]["distributions"]["primary"]["stage1"]["events"] == 281
    assert report["clocks"]["distributions"]["primary"]["2023"]["events"] == 101
    with gzip.open(ledger, "rt") as handle:
        assert handle.readline().startswith("control,observation_date,signal_time")


def test_support_ledger_is_deterministic(tmp_path: Path) -> None:
    left = tmp_path / "left.csv.gz"
    right = tmp_path / "right.csv.gz"
    support.build_support(output_path=tmp_path / "left.json", ledger_path=left)
    support.build_support(output_path=tmp_path / "right.json", ledger_path=right)
    assert left.read_bytes() == right.read_bytes()
