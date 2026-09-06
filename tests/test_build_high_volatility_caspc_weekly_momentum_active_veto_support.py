from __future__ import annotations

import pandas as pd

from training import build_high_volatility_caspc_weekly_momentum_active_veto_support as support
from training import preregister_high_volatility_caspc_weekly_momentum_active_veto as prereg


def _frame(values: dict[str, int], hold: str = "8h") -> pd.DataFrame:
    decision = pd.to_datetime(list(values), utc=True)
    return pd.DataFrame({"candidate": "x", "control": "primary", "split": "train", "decision_time": decision, "feature_available_time": decision, "entry_time": decision + pd.Timedelta("5m"), "exit_time": decision + pd.Timedelta("5m") + pd.Timedelta(hold), "side": list(values.values())})


def test_frozen_inputs_verify() -> None:
    assert prereg.sha256_file(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    assert set(support.verify_inputs()) == set(prereg.COMPONENT_IDS)


def test_exact_opposite_side_veto(monkeypatch) -> None:
    d0, d1, d3 = "2023-07-01T01:00:00Z", "2023-07-02T01:00:00Z", "2023-07-06T01:00:00Z"
    first, second = prereg.ACTION_IDS
    clocks = {
        first: _frame({d0: 1, d1: -1, d3: 1}),
        second: _frame({d1: 1, d3: 1}, "72h"),
    }
    monkeypatch.setattr(support, "_stage", lambda _: "train")
    clock, counts = support.build_clock(clocks)
    assert list(clock["decision_time"]) == [pd.Timestamp(d0), pd.Timestamp(d3)]
    assert list(clock["side"]) == [1, 1]
    assert list(clock["selected_action"]) == [first, first]
    assert counts["primary_decisions"] == 3
    assert counts["veto_absent"] == 1
    assert counts["veto_opposite_side"] == 1
    assert counts["veto_same_side"] == 1
    assert counts["emitted_decisions"] == 2


def test_real_run_is_deterministic_and_sealed(tmp_path) -> None:
    clock, result = tmp_path / "clock.csv.gz", tmp_path / "result.json"
    first = support.run(clock, result)
    first_bytes = (clock.read_bytes(), result.read_bytes())
    second = support.run(clock, result)
    assert (clock.read_bytes(), result.read_bytes()) == first_bytes
    assert second == first
    assert first["combined_postentry_returns_or_pnl_opened"] is False
    assert first["gross9_comparator_rows_opened"] is False
    assert first["support_passed"] == (
        all(first["support_checks"].values())
        and all(first["operator_activation_checks"].values())
    )


def test_manifest_hash_is_canonical(tmp_path) -> None:
    report = support.run(tmp_path / "clock.csv.gz", tmp_path / "result.json")
    assert report["manifest_hash"] == prereg.canonical_hash({key: value for key, value in report.items() if key != "manifest_hash"})
