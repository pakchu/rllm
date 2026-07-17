from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from training import preregister_sofr_rate_dislocation as prereg
from training import sofr_rate_dislocation_clock as exact_clock


def test_manifest_is_deterministic_singleton_and_outcome_blind() -> None:
    first = prereg.build_manifest()
    assert first == prereg.build_manifest()
    assert first["outcomes_opened"] is False
    assert first["selection_protocol"]["candidate_count"] == 1
    assert first["support_verification"]["varying_parameters"] == []
    assert first["causal_feature_contract"]["price_signal_columns"] == []
    assert first["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in first.items() if key != "manifest_hash"}
    )
    prereg.validate_manifest(first)


def test_policy_freezes_rank_tails_clock_and_hold() -> None:
    policy = asdict(prereg.Policy())
    assert policy["delta_rank_lookback_observations"] == 120
    assert policy["lower_tail_quantile"] == 0.15
    assert policy["upper_tail_quantile"] == 0.85
    assert policy["execution_delay_bars"] == 1
    assert policy["hold_bars"] == 1440
    assert policy["leverage"] == 0.5


def test_feature_contract_excludes_quarterly_summary_and_all_crypto_inputs() -> None:
    manifest = prereg.build_manifest()
    feature = manifest["causal_feature_contract"]
    assert feature["allowed_source_columns"] == [
        "effective_date",
        "sofr_available_at_utc",
        "sofr_percent",
    ]
    assert "summary_available_at_utc" in feature["forbidden_source_columns"]
    assert "volume_usd_billions" in feature["forbidden_source_columns"]
    assert "excludes t" in feature["strict_prior_rank"]
    excluded = manifest["novelty_boundary"]["excluded_inputs"]
    assert "OHLC_or_price_return" in excluded
    assert "perpetual_funding_premium_basis_or_open_interest" in excluded


def test_support_counts_match_performance_gates_and_preflight_is_disclosed() -> None:
    manifest = prereg.build_manifest()
    support = manifest["support_verification"]
    gates = manifest["selection_protocol"]["gates"]
    for left, right in (
        ("minimum_nonoverlap_train", "train_trades_min"),
        ("minimum_2021", "2021_trades_min"),
        ("minimum_2022", "2022_trades_min"),
        ("minimum_2023", "2023_trades_min"),
        ("minimum_2023_h1", "2023_h1_trades_min"),
        ("minimum_2023_h2", "2023_h2_trades_min"),
    ):
        assert support[left] == gates[right]
    assert support["expected_preflight_counts"]["train"] == 48
    assert support["expected_preflight_counts"]["2023"] == 40
    disclosure = manifest["research_history_boundary"]
    assert disclosure["source_only_density_preflight_seen_through_2023"] is True
    assert disclosure["exact_sfrd_1_post_entry_outcomes_opened"] is False
    assert disclosure["candidate_class"] == "source-only-screened exploratory singleton"
    assert support["pre_freeze_source_only_screen"]["tail_quantile_grid"] == [
        0.80,
        0.825,
        0.85,
        0.875,
        0.90,
        0.925,
        0.95,
    ]
    assert support["clock_ledger_events_full_source"] == 158
    assert len(support["clock_ledger_sha256"]) == 64
    for left, right in (
        ("minimum_train_each_side", "train_each_side_trades_min"),
        ("minimum_2023_each_side", "2023_each_side_trades_min"),
        ("maximum_single_month_share_train", "train_single_month_share_max"),
        ("maximum_single_month_share_2023", "2023_single_month_share_max"),
    ):
        assert support[left] == gates[right]


def test_exact_clock_replays_manifest_counts_and_ledger_hash() -> None:
    manifest = prereg.build_manifest()
    support = manifest["support_verification"]
    events = exact_clock.build_events(exact_clock.read_source())
    assert events == exact_clock.read_event_ledger(support["clock_ledger"])
    windows = {
        "train": ("2021-01-01", "2023-01-01"),
        "2021": ("2021-01-01", "2022-01-01"),
        "2022": ("2022-01-01", "2023-01-01"),
        "2023": ("2023-01-01", "2024-01-01"),
        "2023_h1": ("2023-01-01", "2023-07-01"),
        "2023_h2": ("2023-07-01", "2024-01-01"),
    }
    expected = support["expected_preflight_counts"]
    for name, (start, end) in windows.items():
        assert exact_clock.event_summary(events, start, end)["count"] == expected[name]
    train = exact_clock.event_summary(events, *windows["train"])
    stage2 = exact_clock.event_summary(events, *windows["2023"])
    assert (train["long"], train["short"]) == (
        expected["train_long"],
        expected["train_short"],
    )
    assert (stage2["long"], stage2["short"]) == (
        expected["2023_long"],
        expected["2023_short"],
    )
    assert train["max_single_month_share"] == expected[
        "train_max_single_month_share"
    ]
    assert stage2["max_single_month_share"] == expected[
        "2023_max_single_month_share"
    ]
    assert hashlib.sha256(Path(support["clock_ledger"]).read_bytes()).hexdigest() == support[
        "clock_ledger_sha256"
    ]


def test_direction_execution_statistics_and_rejection_are_frozen() -> None:
    manifest = prereg.build_manifest()
    assert "tightening state +1 -> fixed SHORT" in manifest[
        "causal_feature_contract"
    ]["action"]
    assert "five calendar days" in manifest["execution_contract"]["exit"]
    assert manifest["selection_protocol"]["gates"][
        "train_and_2023_cagr_to_strict_mdd_min"
    ] == 3.0
    assert manifest["selection_protocol"]["statistical_test_contract"][
        "draws"
    ] == 20_000
    assert "without" in manifest["rejection_contract"]


def test_frozen_sources_and_comparators_match() -> None:
    manifest = prereg.build_manifest()
    source = manifest["source_contract"]
    for key, hash_key in (
        ("sofr", "sofr_sha256"),
        ("sofr_manifest", "sofr_manifest_sha256"),
        ("market", "market_sha256"),
        ("market_manifest", "market_manifest_sha256"),
        ("funding", "funding_sha256"),
        ("funding_manifest", "funding_manifest_sha256"),
    ):
        assert hashlib.sha256(Path(source[key]).read_bytes()).hexdigest() == source[
            hash_key
        ]
    for item in manifest["orthogonality_after_standalone_pass"][
        "comparator_universe"
    ].values():
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item[
            "sha256"
        ]


def test_validate_rejects_opened_outcomes_policy_drift_and_summary_use() -> None:
    opened = prereg.build_manifest()
    opened["outcomes_opened"] = True
    opened["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in opened.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="outcomes opened"):
        prereg.validate_manifest(opened, verify_sources=False)

    drifted = prereg.build_manifest()
    drifted["policy"]["hold_bars"] = 288
    drifted["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in drifted.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="policy differs"):
        prereg.validate_manifest(drifted, verify_sources=False)

    leaked = prereg.build_manifest()
    leaked["causal_feature_contract"]["allowed_source_columns"].append(
        "volume_usd_billions"
    )
    leaked["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in leaked.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="column boundary"):
        prereg.validate_manifest(leaked, verify_sources=False)


def test_written_preregistration_artifact_replays() -> None:
    payload = json.loads(Path(prereg.DEFAULT_OUTPUT).read_text())
    assert payload == prereg.build_manifest()
    prereg.validate_manifest(payload)
