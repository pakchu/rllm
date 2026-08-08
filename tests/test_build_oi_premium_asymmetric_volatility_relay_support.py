import pandas as pd

from training import build_oi_premium_asymmetric_volatility_relay_support as support


def test_clock_skips_conflict_and_reserves_globally(monkeypatch):
    frame = pd.DataFrame({"date": pd.date_range("2023-07-01", periods=6, freq="5min", tz="UTC")})
    monkeypatch.setattr(support, "state_signals", lambda _frame, _control="primary": ([True, True, False, False, False, False], [True, False, False, False, False, False]))
    clock = support.build_clock(frame)
    assert len(clock) == 1
    assert clock.iloc[0].side == 1
    assert clock.iloc[0].entry_time == frame.date.iloc[1] + pd.Timedelta(minutes=5)


def test_direction_flip_changes_only_side(monkeypatch):
    frame = pd.DataFrame({"date": pd.date_range("2023-07-01", periods=2, freq="5min", tz="UTC")})
    monkeypatch.setattr(support, "state_signals", lambda _frame, _control="primary": ([True, False], [False, False]))
    primary = support.build_clock(frame, "primary")
    flipped = support.build_clock(frame, "direction_flip")
    assert primary.entry_time.tolist() == flipped.entry_time.tolist()
    assert primary.side.tolist() == [-x for x in flipped.side.tolist()]
