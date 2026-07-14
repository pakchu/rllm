from __future__ import annotations

import numpy as np
import pandas as pd

from training import preregister_refill_inference_flow_topology as rift


def _frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
            "perp_quarantined": False,
            "spot_quarantined": False,
            "spot_close_vs_centroid_mid_bp": 10.0,
            "spot_micro_log_return": 0.01,
            "micro_log_return": 0.01,
            "spot_signed_quote_notional": 100.0,
            "signed_quote_notional": 100.0,
            "signed_event_imbalance": 0.5,
            "spot_minute_price_path_efficiency": 0.9,
            "spot_minute_flow_path_efficiency": 0.9,
            "spot_minute_flow_price_alignment": 1.0,
            "spot_minute_flow_sign_flip_rate": 0.0,
            "event_notional_hhi": 0.1,
            "interarrival_burstiness": 0.8,
            "agg_trade_count": 100,
        }
    )


def test_setup_score_rewards_mark_path_and_crowd_jointly() -> None:
    frame = _frame()
    values = rift._components(frame)
    baseline = rift._scores(values)["primary"].iloc[-1]
    for column in (
        "spot_close_vs_centroid_mid_bp",
        "spot_minute_price_path_efficiency",
        "event_notional_hhi",
    ):
        weakened = frame.copy()
        weakened.loc[39, column] *= 0.25
        score = rift._scores(rift._components(weakened))["primary"].iloc[-1]
        assert score < baseline


def test_signal_requires_previous_setup_and_current_confirmation() -> None:
    frame = _frame()
    cfg = rift.Config(baseline_bars=4, baseline_min_periods=2)
    signal, controls, _ = rift.classify_sequences(frame, cfg, quantile=0.5)
    assert controls["primary"].iloc[-1]
    assert signal["side"].iloc[-1] == 1

    broken_setup = frame.copy()
    broken_setup.loc[38, "spot_signed_quote_notional"] = -100.0
    _, controls, _ = rift.classify_sequences(broken_setup, cfg, quantile=0.5)
    assert not controls["primary"].iloc[-1]

    broken_confirmation = frame.copy()
    broken_confirmation.loc[39, "spot_micro_log_return"] = -0.01
    _, controls, _ = rift.classify_sequences(
        broken_confirmation, cfg, quantile=0.5
    )
    assert not controls["primary"].iloc[-1]


def test_current_setup_alone_cannot_create_same_bar_signal() -> None:
    frame = _frame()
    frame.loc[:38, "spot_signed_quote_notional"] = -100.0
    cfg = rift.Config(baseline_bars=4, baseline_min_periods=2)
    _, controls, _ = rift.classify_sequences(frame, cfg, quantile=0.5)
    assert controls["same_bar_static"].iloc[-1]
    assert not controls["primary"].iloc[-1]


def test_signal_fails_closed_on_either_venue_quarantine() -> None:
    cfg = rift.Config(baseline_bars=4, baseline_min_periods=2)
    for column in ("spot_quarantined", "perp_quarantined", "quarantined"):
        frame = _frame()
        frame.loc[39, column] = True
        if column != "quarantined":
            frame.loc[39, "quarantined"] = True
        _, controls, _ = rift.classify_sequences(frame, cfg, quantile=0.5)
        assert not controls["primary"].iloc[-1]


def test_stale_setup_controls_do_not_use_current_setup() -> None:
    frame = _frame()
    cfg = rift.Config(baseline_bars=4, baseline_min_periods=2)
    _, original, _ = rift.classify_sequences(frame, cfg, quantile=0.5)
    changed = frame.copy()
    changed.loc[39, "spot_close_vs_centroid_mid_bp"] = 1_000.0
    changed.loc[39, "spot_signed_quote_notional"] = -1_000.0
    _, altered, _ = rift.classify_sequences(changed, cfg, quantile=0.5)
    assert original["stale_setup_1h"].iloc[-1] == altered[
        "stale_setup_1h"
    ].iloc[-1]


def test_spot_only_control_does_not_depend_on_perp_features() -> None:
    frame = _frame()
    cfg = rift.Config(baseline_bars=4, baseline_min_periods=2)
    _, original, _ = rift.classify_sequences(frame, cfg, quantile=0.5)
    altered = frame.copy()
    altered["perp_quarantined"] = True
    altered["quarantined"] = True
    for column in (
        "micro_log_return",
        "signed_quote_notional",
        "signed_event_imbalance",
        "event_notional_hhi",
        "interarrival_burstiness",
    ):
        altered[column] = np.nan
    _, changed, _ = rift.classify_sequences(altered, cfg, quantile=0.5)
    assert original["spot_only"].equals(changed["spot_only"])


def test_component_ablations_do_not_depend_on_removed_fields() -> None:
    frame = _frame()
    cfg = rift.Config(baseline_bars=4, baseline_min_periods=2)
    _, original, _ = rift.classify_sequences(frame, cfg, quantile=0.5)

    no_path = frame.copy()
    for column in (
        "spot_minute_price_path_efficiency",
        "spot_minute_flow_path_efficiency",
        "spot_minute_flow_price_alignment",
        "spot_minute_flow_sign_flip_rate",
    ):
        no_path[column] = np.nan
    _, changed, _ = rift.classify_sequences(no_path, cfg, quantile=0.5)
    assert original["no_path_quality"].equals(changed["no_path_quality"])

    no_crowd = frame.copy()
    no_crowd["event_notional_hhi"] = np.nan
    no_crowd["interarrival_burstiness"] = np.nan
    _, changed, _ = rift.classify_sequences(no_crowd, cfg, quantile=0.5)
    assert original["no_derivatives_crowd"].equals(
        changed["no_derivatives_crowd"]
    )

    no_centroid = frame.copy()
    no_centroid["spot_close_vs_centroid_mid_bp"] = np.nan
    _, changed, _ = rift.classify_sequences(no_centroid, cfg, quantile=0.5)
    assert original["centroid_free_momentum"].equals(
        changed["centroid_free_momentum"]
    )


def test_simple_momentum_uses_only_two_bar_returns_and_cleanliness() -> None:
    frame = _frame()
    frame.loc[38:, "spot_close_vs_centroid_mid_bp"] = -10.0
    frame.loc[38:, "spot_signed_quote_notional"] = -100.0
    frame.loc[38:, "signed_quote_notional"] = -100.0
    frame.loc[38:, "signed_event_imbalance"] = -0.5
    cfg = rift.Config(baseline_bars=4, baseline_min_periods=2)
    _, controls, _ = rift.classify_sequences(frame, cfg, quantile=0.5)
    assert controls["simple_two_bar_momentum"].iloc[-1]
    assert not controls["primary"].iloc[-1]


def test_direction_flip_action_is_explicitly_short() -> None:
    assert rift.CONTROL_ACTIONS["primary"] == 1
    assert rift.CONTROL_ACTIONS["direction_flip"] == -1
    assert set(rift.CONTROL_ACTIONS.values()) == {-1, 1}


def test_support_requires_long_only_clock_and_every_period() -> None:
    schedule = pd.DataFrame(
        {
            "entry_date": [
                "2020-01-01",
                "2021-01-01",
                "2022-01-01",
                "2023-02-01",
                "2023-08-01",
            ],
            "side": [1, 1, 1, 1, 1],
        }
    )
    cfg = rift.Config(
        minimum_nonoverlap_total=5,
        minimum_nonoverlap_per_year=1,
        minimum_nonoverlap_per_2023_half=1,
    )
    assert rift._support(schedule, cfg)["passes_count_support"]
    schedule.loc[0, "side"] = -1
    assert not rift._support(schedule, cfg)["passes_count_support"]


def test_jaccard_handles_empty_and_partial_overlap() -> None:
    assert rift._jaccard(pd.Series([False]), pd.Series([False])) == 0.0
    assert np.isclose(
        rift._jaccard(
            pd.Series([True, True, False]),
            pd.Series([True, False, True]),
        ),
        1.0 / 3.0,
    )
