import numpy as np
import pandas as pd

from training import build_treasury_parallel_yield_shock_relay_support as support


def candidate_frame():
    return pd.DataFrame({
        "signal_valid": [True] * 5,
        "parallel_factor": [1.0, -1.0, 0.5, 1.0, -0.2],
        "z_2y": [1.2, -0.8, 0.4, 0.7, -0.3],
        "z_10y": [0.8, -1.2, 0.6, 1.3, -0.1],
        "absolute_factor_rank": [0.8, 0.9, 0.69, 0.8, 0.8],
        "btc_realized_variation_rank": [0.7, 0.8, 0.9, 0.4, 0.9],
    })


def test_primary_fades_extreme_parallel_yield_shock():
    active, side = support.conditions(candidate_frame(), "primary")
    assert active.tolist() == [True, True, False, False, True]
    assert side[active].tolist() == [-1, 1, 1]


def test_controls_are_frozen_diagnostic_transformations():
    frame = candidate_frame()
    assert support.CONTROLS == (
        "two_year_only", "ten_year_only", "no_factor_tail", "no_volatility_gate",
        "one_observation_stale_factor", "direction_flip",
    )
    assert support.conditions(frame, "no_factor_tail")[0].tolist() == [True, True, True, False, True]
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


def test_official_xml_fields_are_parsed_without_imputation():
    payload = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"><entry><content>
      <m:properties><d:NEW_DATE>2026-01-02T00:00:00</d:NEW_DATE>
      <d:BC_2YEAR>3.47</d:BC_2YEAR><d:BC_10YEAR>4.19</d:BC_10YEAR></m:properties>
      </content></entry></feed>'''
    frame = support.parse_xml(payload)
    assert frame.source_day.iloc[0] == pd.Timestamp("2026-01-02T00:00:00Z")
    assert frame.yield_2y.iloc[0] == 3.47
    assert frame.yield_10y.iloc[0] == 4.19


def test_builder_binds_preregistration_and_seals_outcomes():
    assert support.sha(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA
    source = support.Path(support.__file__).read_text()
    assert '"postentry_return_pnl_execution_price_opened": False' in source
    assert '"gross9_rows_opened": False' in source
