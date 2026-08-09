import numpy as np
import pandas as pd

from training import build_commodity_currency_relative_stress_relay_support as support


def frame():
    return pd.DataFrame(
        {
            "signal_valid": [True] * 6,
            "source_valid": [True] * 6,
            "btc_valid": [True] * 6,
            "usdcad_return": [0.02, -0.01, 0.03, 0.01, -0.02, 0.02],
            "usdaud_return": [0.01, 0.01, -0.01, 0.00, -0.01, 0.01],
            "relative_stress": [0.01, -0.02, 0.04, 0.01, -0.01, 0.01],
            "absolute_stress_rank": [0.80, 0.90, 0.60, 0.80, 0.90, 0.90],
            "btc_realized_variation": [0.02] * 6,
            "btc_realized_variation_rank": [0.70, 0.80, 0.90, 0.40, 0.90, 0.90],
        }
    )


def test_primary_uses_both_frozen_gates_and_inverse_relative_stress_side():
    active, side = support.conditions(frame(), "primary")
    assert active.tolist() == [True, True, False, False, True, True]
    assert side[active].tolist() == [-1, 1, 1, -1]


def test_controls_are_frozen_and_diagnostic_only():
    assert support.CONTROLS == (
        "no_stress_tail",
        "no_variation_gate",
        "usdcad_only",
        "usdaud_only",
        "one_session_stale_stress",
        "direction_flip",
        "same_clock_forced_long",
    )
    assert support.conditions(frame(), "no_stress_tail")[0].tolist() == [True, True, True, False, True, True]
    assert support.conditions(frame(), "no_variation_gate")[0].tolist() == [True, True, False, True, True, True]
    active, side = support.conditions(frame(), "direction_flip")
    assert side[active].tolist() == [1, -1, -1, 1]
    active, side = support.conditions(frame(), "same_clock_forced_long")
    assert side[active].tolist() == [1, 1, 1, 1]


def test_component_controls_preserve_primary_clock_and_frozen_orientation():
    primary_active, _ = support.conditions(frame(), "primary")
    cad_active, cad_side = support.conditions(frame(), "usdcad_only")
    aud_active, aud_side = support.conditions(frame(), "usdaud_only")
    assert cad_active.equals(primary_active)
    assert aud_active.equals(primary_active)
    assert cad_side[primary_active].tolist() == [-1, 1, 1, -1]
    assert aud_side[primary_active].tolist() == [1, 1, -1, 1]


def test_stale_control_shifts_only_fx_stress_and_keeps_current_variation_gate():
    active, side = support.conditions(frame(), "one_session_stale_stress")
    assert active.tolist() == [False, True, True, False, True, True]
    assert side[active].tolist() == [-1, 1, -1, 1]


def test_causal_rank_excludes_current_and_uses_frozen_history():
    values = pd.Series(np.arange(127, dtype=float))
    rank = support.strict_prior_midrank(values)
    assert np.isnan(rank.iloc[125])
    assert rank.iloc[126] == 1.0


def test_pair_session_and_preregistration_binding_are_frozen():
    assert support.SYMBOLS == ("USDCAD", "USDAUD")
    assert "extract(hour from ts)>=13" in support.QUERY
    assert "extract(hour from ts)<21" in support.QUERY
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
