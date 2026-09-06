import pandas as pd

from training import build_volatility_gated_stablecoin_quote_flow_consensus_relay_support as support


def frame() -> pd.DataFrame:
    decision = pd.Timestamp("2024-07-01T08:00:00Z")
    return pd.DataFrame(
        {
            "source_hour_start": [decision - pd.Timedelta(hours=2), decision - pd.Timedelta(hours=1)],
            "decision_time": [decision - pd.Timedelta(hours=1), decision],
            "source_valid": [True, True],
            "vol_valid": [True, True],
            "z_usdt": [0.0, 0.2],
            "z_usdc": [0.0, 1.0],
            "z_fdusd": [0.0, 0.8],
            "alt_share": [0.6, 0.6],
            "prior_alt_share_q50": [0.5, 0.5],
            "bvol_close": [70.0, 70.0],
            "prior_bvol_q60": [60.0, 60.0],
            "dvol_close": [65.0, 65.0],
            "prior_dvol_q60": [60.0, 60.0],
        }
    )


def test_vgsqf_clock_uses_current_consensus_with_exact_entry_and_hold():
    clocks = support.clock(frame())
    assert len(clocks) == 1
    assert clocks.iloc[0].side == 1
    assert clocks.iloc[0].source_hour_start == pd.Timestamp("2024-07-01T07:00:00Z")
    assert clocks.iloc[0].entry_time == pd.Timestamp("2024-07-01T08:05:00Z")
    assert clocks.iloc[0].exit_time - clocks.iloc[0].entry_time == pd.Timedelta(hours=6)


def test_vgsqf_rejects_disagreement_usdt_catchup_and_low_volatility():
    disagreement = frame()
    disagreement.loc[1, "z_fdusd"] = -0.8
    assert support.clock(disagreement).empty
    usdt_caught_up = frame()
    usdt_caught_up.loc[1, "z_usdt"] = 0.5
    assert support.clock(usdt_caught_up).empty
    low_volatility = frame()
    low_volatility.loc[1, "dvol_close"] = 50.0
    assert support.clock(low_volatility).empty


def test_vgsqf_stale_flow_is_a_separate_non_promotable_clock_variant():
    stale = frame()
    stale.loc[0, ["z_usdt", "z_usdc", "z_fdusd"]] = [0.1, -1.0, -0.8]
    stale.loc[1, ["z_usdt", "z_usdc", "z_fdusd"]] = [0.0, 0.0, 0.0]
    assert support.clock(stale).empty
    stale_clock = support.clock(stale, "one_hour_stale_flow")
    assert len(stale_clock) == 1
    assert stale_clock.iloc[0].side == -1
    assert stale_clock.iloc[0].control == "one_hour_stale_flow"
