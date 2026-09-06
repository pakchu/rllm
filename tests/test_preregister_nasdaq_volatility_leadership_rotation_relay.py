import json
import subprocess
import sys
from pathlib import Path

from training import preregister_nasdaq_volatility_leadership_rotation_relay as prereg


EXPECTED_HASHES = {
    "vix_panel": "42eb1093f5167aec9c71a4733ab3451e40807c81dc7cb49568a6a0c634267ba0",
    "vxn": "a856a08bbec6c5fabd23c98d4b75bc28b93c24370ef8780e3e75243c22f66716",
    "gvz": "eaf949af798669fb6a0d5eb0ee5a3d148a9f2abf4679450e7a9d85c6a5e1bcbb",
    "ovx": "77f872f1e069cc93554fe6d80dc6f9d44d0a798ad0a906202a570ad81f73417a",
}


def test_nvlrr_formulas_clock_and_side_are_frozen():
    result = prereg.build()
    features = result["features"]
    policy = result["policy"]

    assert "exact intersection" in features["common_source_dates"]
    assert "positive finite close" in features["common_source_dates"]
    assert "previous exact common source date" in features["index_changes"]
    assert features["index_zscores"]["formula"] == (
        "z_i(S)=(change_i(S)-mean(H_i(S)))/sample_std(H_i(S))"
    )
    assert "strictly prior" in features["index_zscores"]["history"]
    assert "current change excluded" in features["index_zscores"]["history"]
    assert "ddof=1" in features["index_zscores"]["validity"]
    assert features["leadership_residual"] == (
        "z_VXN - median(z_VIX,z_GVZ,z_OVX); strict nonzero"
    )
    assert features["absolute_leadership_rank"]["midrank_formula"] == (
        "(count(prior_abs<current_abs)+0.5*count(prior_abs==current_abs))/N"
    )
    assert features["absolute_leadership_rank"]["eligible"] == "rank>=0.70"
    assert "next exact common Cboe source date D" in features["availability"]
    assert "exact 1440 unique BTCUSDT 1m rows" in features["btc_variation"]
    assert features["btc_variation_rank"]["eligible"] == "rank>=0.65"
    assert policy["zscore_history_observations"] == 252
    assert policy["zscore_minimum_history_observations"] == 126
    assert policy["absolute_leadership_rank_min"] == 0.70
    assert policy["variation_rank_min"] == 0.65
    assert result["clock"]["side"] == (
        "-sign(leadership_residual); no BTC direction confirmation"
    )
    assert result["clock"]["entry"] == (
        "exact BTCUSDT D 09:35 America/New_York 5m open"
    )
    assert result["clock"]["hold"] == "12 elapsed hours"
    assert result["clock"]["reservation"] == "global half-open; exit first on equal open"
    assert result["clock"]["split_crossing_action"] == "skip"
    assert result["clock"]["gross_exposure"] == 0.5


def test_nvlrr_hashes_gates_controls_and_boundary_are_frozen():
    result = prereg.build()
    prereg.validate(result)

    assert result["policy_id"] == "NVLRR-12"
    assert result["slug"] == "nasdaq_volatility_leadership_rotation_relay"
    assert result["as_of_date"] == "2026-08-10"
    assert result["singleton"] is True
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert {name: source["sha256"] for name, source in prereg.SOURCES.items()} == (
        EXPECTED_HASHES
    )
    assert prereg.SOURCES["vix_panel"]["value_column"] == "VIX"
    assert result["source_support_gates"] == {
        "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
        "minority_side_share_min": 0.20,
        "max_month_share": 0.45,
    }
    assert result["novelty_gates"]["must_pass_before_economics"] is True
    assert result["economic_gates"]["mean_gross_underlying_min_bp"] == 20.0
    assert result["economic_gates"]["future_can_rank_repair_or_reselect"] is False
    assert result["rv20_stress_slice"]["entry_filter"] is False
    assert "only after all" in result["rv20_stress_slice"]["future_use"]
    assert result["diagnostic_controls"]["names"] == [
        "no_btc_variation_gate",
        "no_leadership_tail",
        "vxn_minus_vix_raw",
        "one_session_stale_leadership",
        "direction_flip",
        "forced_long",
    ]
    assert result["diagnostic_controls"]["cannot_be_promoted"] is True
    boundary = result["research_boundary"]
    assert boundary["prior_nvxcr_outcomes_known"] is True
    assert boundary["prior_cross_asset_breadth_outcomes_known"] is True
    assert boundary["prior_equity_commodity_residual_outcomes_known"] is True
    assert boundary[
        "exact_standardized_vxn_vs_three_index_leadership_residual_previously_tested"
    ] is False
    assert boundary["prior_outcomes_used_to_set_formula_rank_side_hold_or_clock"] is False
    assert boundary["candidate_count"] == 1
    assert boundary["grid"] is False
    assert boundary["repair_of_prior_candidate"] is False
    assert boundary["promoted_prior_control"] is False


def test_nvlrr_manifest_round_trip_and_deterministic_cli(tmp_path):
    result = prereg.build()
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    prereg.validate(decoded)
    assert decoded == result
    assert result["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
    assert prereg.DEFAULT_OUTPUT == Path(
        "results/nasdaq_volatility_leadership_rotation_relay_preregistration_2026-08-10.json"
    )

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    module = "training.preregister_nasdaq_volatility_leadership_rotation_relay"
    subprocess.run(
        [sys.executable, "-m", module, "--output", str(first)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "-m", module, "--output", str(second)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text()) == result
