import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_g9_september_clock_inputs as b


def test_archive_oi_shifts_and_rejects_off_grid():
    raw = pd.DataFrame({"create_time": ["2026-09-01 00:00:00"], "sum_open_interest": [123.0]})
    out = b.archive_oi(raw)
    assert out.loc[0, "date"] == pd.Timestamp("2026-09-01 00:05:00")
    assert out.loc[0, "open_interest"] == 123.0
    with pytest.raises(ValueError, match="off-grid"):
        b.archive_oi(pd.DataFrame({"create_time": ["2026-09-01 00:01:00"], "sum_open_interest": [123.0]}))


def test_overlay_preserves_prearchive_db_and_hard_fails_missing_prearchive():
    enriched = pd.DataFrame({
        "date": pd.date_range("2026-04-30 23:55", periods=3, freq="5min"),
        "open_interest": [10.0, 20.0, np.nan],
    })
    archive = pd.DataFrame({"date": [pd.Timestamp("2026-05-01 00:05")], "open_interest": [99.0]})
    out = b.overlay_official_oi(enriched, archive, asof=pd.Timestamp("2026-05-01T00:10:00Z"))
    assert out.loc[0, "open_interest"] == 10.0
    assert out.loc[2, "open_interest"] == 99.0
    bad = enriched.copy(); bad.loc[0, "open_interest"] = np.nan
    with pytest.raises(RuntimeError, match="pre-archive DB open-interest source"):
        b.overlay_official_oi(bad, archive, asof=pd.Timestamp("2026-05-01T00:10:00Z"))


def test_report_contract_has_root_optimizer_shape(tmp_path):
    report = {
        "sleeves": {"x": {"trades": [{"entry_date": "2026-06-01T00:05:00", "exit_date": "2026-06-01T01:00:00", "side": "LONG", "exit_price": 1.0, "exit_kind": "open"}]}},
        "market_csv": str((tmp_path / "market.csv").resolve()),
        "funding_csv": str((tmp_path / "funding.csv").resolve()),
        "receipts": {},
    }
    assert set(report) >= {"sleeves", "market_csv", "funding_csv", "receipts"}
    trade = report["sleeves"]["x"]["trades"][0]
    assert set(trade) >= {"entry_date", "exit_date", "side", "exit_price", "exit_kind"}
    assert trade["exit_kind"] in {"open", "barrier"}
