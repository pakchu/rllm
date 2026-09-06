import json

import numpy as np
import pandas as pd

from training import build_short_complement_context as b


def _tiny_data(start="2024-01-01", periods=4):
    dates = pd.date_range(start, periods=periods, freq="5min").to_numpy()
    return {
        "date": dates,
        "end_date": (pd.DatetimeIndex(dates) + pd.Timedelta("5min")).to_numpy(),
        "open": np.arange(periods, dtype=float) + 100,
        "end": np.arange(periods, dtype=float) + 101,
        "high": np.arange(periods, dtype=float) + 102,
        "low": np.arange(periods, dtype=float) + 99,
        "funding": np.zeros(periods),
    }


def test_array_schema_and_feature_alignment(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=4, freq="5min")
    features = pd.DataFrame({"mom6": [1.0]}, index=pd.DatetimeIndex([dates[0] - pd.Timedelta("5min")]))
    targets = np.zeros((4, 6)); events = np.zeros((4, 6), dtype=bool); barriers = np.full((4, 6), np.nan)
    baseline_row = {
        "weights_notional": dict(zip(b.NAMES, b.WEIGHTS[0].tolist())),
        "return_pct": 1.0,
        "cagr_pct": 2.0,
        "mdd_pct": 3.0,
        "calmar": 4.0,
        "entry_episodes": 5,
        "orders": 6,
        "turnover": 7.0,
        "fees_pct_initial": 8.0,
        "funding_pct_initial": 9.0,
        "max_open_net_exposure": 10.0,
        "mean_open_net_exposure": 11.0,
        "cap_interventions": 12,
        "insolvent": False,
    }

    monkeypatch.setattr(b, "build_context", lambda: {
        "registration": {},
        "receipt": {},
        "execution_receipt": {},
        "trade_receipts": {},
        "trade_counts": {"2024": {}},
        "baseline": {"2024": baseline_row},
        "periods": {"2024": {
            "data": _tiny_data(),
            "targets": targets,
            "events": events,
            "barriers": barriers,
            "feature_names": np.array(["mom6"]),
            "feature_date": features.index.to_numpy(),
            "feature_next5m_date": (features.index + pd.Timedelta("5min")).to_numpy(),
            "features": features.to_numpy(float),
            "feature_row_for_5m": np.array([0, -1, -1, -1]),
        }},
    })
    monkeypatch.setattr(b.hist, "WINDOWS", {"2024": ("2024-01-01", "2025-01-01")})
    baseline_path = tmp_path / "baseline" / "report.json"
    baseline_path.parent.mkdir()
    baseline_path.write_text(json.dumps({"reports": {"2024": {str(b.COST): {b.WEIGHT_LABEL: baseline_row}}}}))
    monkeypatch.setattr(b.hist, "OUT", baseline_path.parent)

    report = b.write_artifacts(tmp_path / "out")
    artifact = tmp_path / report["artifacts"]["2024"]["path"]
    loaded = np.load(artifact)

    assert loaded["targets"].shape == (4, 6)
    assert loaded["events"].shape == (4, 6)
    assert loaded["barriers"].shape == (4, 6)
    assert loaded["feature_names"].tolist() == ["mom6"]
    assert loaded["feature_next5m_date"][0] == loaded["date"][0]
    assert loaded["feature_row_for_5m"].tolist() == [0, -1, -1, -1]
    assert report["baseline_checks"]["2024"]["matched"] is True


def test_report_declares_2024_selection_only(monkeypatch, tmp_path):
    monkeypatch.setattr(b, "build_context", lambda: {
        "registration": {}, "receipt": {}, "execution_receipt": {}, "trade_receipts": {},
        "trade_counts": {"2024": {}}, "baseline": {"2024": {"weights_notional": dict(zip(b.NAMES, b.WEIGHTS[0].tolist()))}},
        "periods": {"2024": {
            "data": _tiny_data(), "targets": np.zeros((4, 6)), "events": np.zeros((4, 6), bool),
            "barriers": np.full((4, 6), np.nan), "feature_names": np.array([], dtype="U"),
            "feature_date": np.array([], dtype="datetime64[ns]"), "feature_next5m_date": np.array([], dtype="datetime64[ns]"),
            "features": np.empty((0, 0)), "feature_row_for_5m": np.full(4, -1),
        }},
    })
    monkeypatch.setattr(b.hist, "WINDOWS", {"2024": ("2024-01-01", "2025-01-01")})
    baseline_path = tmp_path / "baseline" / "report.json"
    baseline_path.parent.mkdir()
    baseline_path.write_text(json.dumps({"reports": {"2024": {str(b.COST): {b.WEIGHT_LABEL: {"weights_notional": dict(zip(b.NAMES, b.WEIGHTS[0].tolist()))}}}}}))
    monkeypatch.setattr(b.hist, "OUT", baseline_path.parent)

    report = b.write_artifacts(tmp_path / "out")

    assert report["selection_windows"] == ["2024"]
    assert report["report_windows"] == ["2025", "2026H1"]
    assert report["weights"] == {b.WEIGHT_LABEL: [1.0, 1.5, 0.2, 0.8, 1.0, 1.0]}
