import json

import numpy as np
import pandas as pd

from training import build_ten_alpha_context as b


def _tiny_source(periods=6):
    dates = pd.date_range("2024-01-01", periods=periods, freq="5min").to_numpy()
    targets = np.arange(periods * 6, dtype=float).reshape(periods, 6)
    events = (targets % 3 == 0)
    barriers = np.full((periods, 6), np.nan)
    barriers[2, 1] = 123.0
    return {
        "open": np.arange(periods, dtype=float) + 100,
        "end": np.arange(periods, dtype=float) + 101,
        "high": np.arange(periods, dtype=float) + 102,
        "low": np.arange(periods, dtype=float) + 99,
        "funding": np.zeros(periods),
        "date": dates,
        "end_date": (pd.DatetimeIndex(dates) + pd.Timedelta("5min")).to_numpy(),
        "targets": targets,
        "events": events,
        "barriers": barriers,
        "sleeve_names": np.array(b.SOURCE_SIX, dtype="U"),
        "weights": np.array([1, 1.5, .2, .8, 1, 1.0], dtype=float),
        "weight_label": np.array("g9_macro1", dtype="U"),
        "feature_names": np.array(["x"], dtype="U"),
        "feature_date": np.array([], dtype="datetime64[ns]"),
        "feature_next5m_date": np.array([], dtype="datetime64[ns]"),
        "features": np.empty((0, 1)),
        "feature_row_for_5m": np.full(periods, -1, dtype=np.int64),
    }


def test_fixed_and_barrier_trade_array_semantics():
    n = 5
    t = np.zeros(n); e = np.zeros(n, bool); bar = np.full(n, np.nan)
    b._set_fixed_trade(t, e, bar, entry=1, exit_=3, side=-1, n=n, barrier=False)
    assert t.tolist() == [0, -1, -1, 0, 0]
    assert e.tolist() == [False, True, False, True, False]
    assert np.isnan(bar).all()

    t = np.zeros(n); e = np.zeros(n, bool); bar = np.full(n, np.nan)
    b._set_fixed_trade(t, e, bar, entry=1, exit_=3, side=-1, n=n, barrier=True, exit_price=88.0)
    assert t.tolist() == [0, -1, -1, -1, 0]
    assert e.tolist() == [False, True, False, False, True]
    assert bar[3] == 88.0


def test_write_preserves_first_six_and_schema(monkeypatch, tmp_path):
    src = _tiny_source()
    targets = np.column_stack([src["targets"], np.ones((6, 4))])
    events = np.column_stack([src["events"], np.zeros((6, 4), dtype=bool)])
    barriers = np.column_stack([src["barriers"], np.full((6, 4), np.nan)])
    monkeypatch.setattr(b, "build_context", lambda: {
        "periods": {"2024": {"source_window": "2024", "source": src, "targets": targets, "events": events, "barriers": barriers}},
        "validation": {"2024": {"first_six_byte_exact": True}},
        "counts": {"2024": {"bounds_by_sleeve": {}}},
        "receipts": {},
    })
    monkeypatch.setattr(b, "WINDOWS", {"2024": ("2024", "2024-01-01", "2024-01-01T00:30:00")})
    monkeypatch.setattr(b, "SOURCE", tmp_path / "src")
    b.SOURCE.mkdir()
    np.savez_compressed(b.SOURCE / "2024_context.npz", **src)
    (b.SOURCE / "report.json").write_text("{}")
    monkeypatch.setattr(b, "sha256_file", lambda p: "sha")

    report = b.write_artifacts(tmp_path / "out")
    loaded = np.load(tmp_path / report["artifacts"]["2024"]["path"])

    assert loaded["targets"].shape == (6, 10)
    assert loaded["events"].shape == (6, 10)
    assert loaded["barriers"].shape == (6, 10)
    assert loaded["sleeve_names"].tolist() == b.NAMES
    assert loaded["names"].tolist() == b.NAMES
    assert loaded["weights"].tolist() == b.WEIGHTS.tolist()
    assert loaded["targets"][:, :6].tobytes() == np.ascontiguousarray(src["targets"]).tobytes()
    assert loaded["events"][:, :6].tobytes() == np.ascontiguousarray(src["events"]).tobytes()
    assert np.array_equal(loaded["barriers"][:, :6], src["barriers"], equal_nan=True)
    assert report["arrays"] == ["open", "end", "high", "low", "funding", "date", "end_date", "targets", "events", "barriers", "names"]
