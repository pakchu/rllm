import json
import subprocess
import sys
from pathlib import Path

from training import preregister_emerging_growth_volatility_rotation_relay as prereg


def test_egvrr_formula_clock_and_side_are_frozen():
    result = prereg.build()
    features = result["features"]
    assert "exact intersection" in features["common_source_dates"]
    assert features["relative_volatility"] == (
        "log(VXEEM_close/VXN_close) on exact common source date S"
    )
    assert "previous exact common source date" in features["relative_volatility_change"]
    assert features["absolute_change_rank"]["eligible"] == "rank>=0.70"
    assert result["clock"]["side"] == (
        "-sign(relative_volatility_change); no BTC direction confirmation"
    )
    assert result["clock"]["entry"] == "exact BTCUSDT D 09:35 America/New_York 5m open"
    assert result["clock"]["hold"] == "12 elapsed hours"
    assert result["clock"]["reservation"] == "global half-open; exit first on equal open"
    assert result["clock"]["gross_exposure"] == 0.5


def test_egvrr_hashes_gates_controls_and_boundary_are_frozen():
    result = prereg.build()
    prereg.validate(result)
    assert result["policy_id"] == "EGVRR-12"
    assert result["slug"] == "emerging_growth_volatility_rotation_relay"
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert prereg.SOURCES["vxeem_panel"]["sha256"] == (
        "7d332f56676bd40c2b7bc816b432ffa2e31c113dbf4f5605fbe8eb4d0dca18ea"
    )
    assert prereg.SOURCES["vxn"]["sha256"] == (
        "a856a08bbec6c5fabd23c98d4b75bc28b93c24370ef8780e3e75243c22f66716"
    )
    assert result["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert result["novelty_gates"]["must_pass_before_economics"] is True
    assert result["economic_gates"]["future_can_rank_repair_or_reselect"] is False
    assert result["diagnostic_controls"]["cannot_be_promoted"] is True
    boundary = result["research_boundary"]
    assert boundary["prior_emvcr_outcomes_known"] is True
    assert boundary["prior_nvxcr_outcomes_known"] is True
    assert boundary["prior_nvlrr_outcomes_known"] is True
    assert boundary["exact_vxeem_vxn_relative_volatility_change_previously_tested"] is False
    assert boundary["candidate_count"] == 1
    assert boundary["grid"] is False
    assert boundary["repair_of_prior_candidate"] is False


def test_egvrr_manifest_round_trip_and_deterministic_cli(tmp_path):
    result = prereg.build()
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    prereg.validate(decoded)
    assert decoded == result
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    module = "training.preregister_emerging_growth_volatility_rotation_relay"
    subprocess.run([sys.executable, "-m", module, "--output", str(first)], check=True)
    subprocess.run([sys.executable, "-m", module, "--output", str(second)], check=True)
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text()) == result
    assert prereg.DEFAULT_OUTPUT == Path(
        "results/emerging_growth_volatility_rotation_relay_preregistration_2026-08-10.json"
    )
