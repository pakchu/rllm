import json

import numpy as np
import pandas as pd

from training import build_debt_public_supply_liquidity_relay_support as support


def candidate_frame():
    return pd.DataFrame({
        "signal_valid": [True] * 5,
        "standardized_public_supply_change": [1.0, -1.0, 0.5, 1.0, -0.2],
        "standardized_total_public_debt_change": [1.2, -0.8, 0.4, 0.7, -0.3],
        "standardized_intragovernmental_change": [0.8, -1.2, 0.6, 1.3, -0.1],
        "absolute_magnitude_rank": [0.8, 0.9, 0.69, 0.8, 0.8],
        "btc_realized_variation_rank": [0.7, 0.8, 0.9, 0.4, 0.9],
    })


def test_primary_fades_extreme_public_debt_supply_change():
    active, side = support.conditions(candidate_frame(), "primary")
    assert active.tolist() == [True, True, False, False, True]
    assert side[active].tolist() == [-1, 1, 1]


def test_controls_are_frozen_diagnostic_transformations():
    frame = candidate_frame()
    assert support.CONTROLS == (
        "total_public_debt_change", "intragovernmental_change", "no_magnitude_tail",
        "no_volatility_gate", "one_observation_stale_supply_change", "direction_flip",
    )
    assert support.conditions(frame, "no_magnitude_tail")[0].tolist() == [True, True, True, False, True]
    assert support.conditions(frame, "no_volatility_gate")[0].tolist() == [True, True, False, True, True]
    active, side = support.conditions(frame, "direction_flip")
    assert side[active].tolist() == [1, -1, -1]


def test_causal_statistics_exclude_current_value():
    values = pd.Series(np.arange(61, dtype=float))
    zscore = support.causal_z(values)
    expected = (60 - values.iloc[:60].mean()) / values.iloc[:60].std(ddof=1)
    assert np.isclose(zscore.iloc[60], expected)
    rank = support.strict_prior_midrank(values)
    assert rank.iloc[60] == 1.0


def test_official_api_fields_and_exact_debt_identity_are_parsed():
    row = {
        "record_date": "2026-01-02", "debt_held_public_amt": "31.25",
        "intragov_hold_amt": "7.50", "tot_pub_debt_out_amt": "38.75", "src_line_nbr": "1",
    }
    payload = json.dumps({"data": [row], "meta": {"total-count": 1, "total-pages": 1},
                          "links": {}}).encode()
    frame, metadata = support.parse_response(payload)
    assert frame.source_day.iloc[0] == pd.Timestamp("2026-01-02T00:00:00Z")
    assert frame.debt_held_public_amt.iloc[0] == 31.25
    assert frame.intragov_hold_amt.iloc[0] == 7.50
    assert frame.tot_pub_debt_out_amt.iloc[0] == 38.75
    assert metadata["total-count"] == 1


def test_builder_binds_preregistration_and_seals_outcomes():
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
