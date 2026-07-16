from __future__ import annotations

from dataclasses import replace

import pandas as pd

from training import freeze_crrc_control_clocks as freeze
from training import qualify_cross_venue_radial_refill_compression as qualify


def _inputs() -> tuple[dict, dict, pd.Series]:
    raw = {}
    thresholds = {}
    for venue in ("um", "cm"):
        for side in ("m", "p"):
            for metric in ("add", "withdraw", "net", "flicker"):
                raw[(venue, side, metric)] = pd.Series([2.0, 2.0])
                thresholds[(venue, side, metric)] = pd.Series([1.0, 1.0])
                if metric == "flicker":
                    raw[(venue, side, metric)] = pd.Series([0.5, 0.5])
    return raw, thresholds, pd.Series([True, True])


def test_um_only_ignores_cm_but_primary_style_controls_do_not() -> None:
    raw, thresholds, complete = _inputs()
    for venue in ("um", "cm"):
        raw[(venue, "p", "add")].iloc[0] = 0.0
    raw[("cm", "m", "add")].iloc[0] = 0.0
    dates = pd.Series(pd.date_range("2023-01-01", periods=2, freq="5min"))
    cfg = replace(qualify.Config(), hold_bars=1, entry_delay_bars=1)
    um = freeze.control_signal(dates, raw, thresholds, complete, "um_only", cfg)
    both = freeze.control_signal(
        dates, raw, thresholds, complete, "inner_add_only", cfg
    )
    assert um.loc[0, "side"] == 1
    assert both.loc[0, "side"] == 0


def test_without_credibility_ignores_net_and_flicker() -> None:
    raw, thresholds, complete = _inputs()
    for venue in ("um", "cm"):
        raw[(venue, "p", "add")] = pd.Series([0.0, 0.0])
        raw[(venue, "m", "net")] = pd.Series([0.0, 0.0])
        raw[(venue, "m", "flicker")] = pd.Series([100.0, 100.0])
    dates = pd.Series(pd.date_range("2023-01-01", periods=2, freq="5min"))
    signal = freeze.control_signal(
        dates,
        raw,
        thresholds,
        complete,
        "without_credibility",
        qualify.Config(),
    )
    assert signal["side"].tolist() == [1, 1]


def test_control_specs_are_exactly_the_preregistered_diagnostics() -> None:
    assert set(freeze.CONTROL_SPECS) == {
        "um_only",
        "cm_only",
        "without_credibility",
        "inner_add_only",
        "outer_withdraw_only",
    }
