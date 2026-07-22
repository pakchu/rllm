from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import preregister_address_funding_divergence_relay as prereg


def test_contract_is_singleton_and_causal() -> None:
    prereg.validate_contract()
    cfg = prereg.FROZEN_CONFIG
    assert cfg.address_change_days == 7
    assert cfg.funding_window_hours == 72
    assert cfg.required_funding_settlements == 9
    assert cfg.funding_publication_delay_minutes == 5
    assert cfg.maximum_funding_slot_offset_ms == 60_000
    assert cfg.maximum_latest_funding_age_hours == 8
    assert cfg.lower_rank == 0.25
    assert cfg.upper_rank == 0.75
    assert cfg.entry_delay_minutes == cfg.bar_minutes == 5
    assert cfg.hold_bars * cfg.bar_minutes == 72 * 60


def test_payload_is_row_and_outcome_blind() -> None:
    payload = prereg.preregistration_payload()
    boundary = payload["outcome_boundary"]
    assert boundary["outcomes_opened"] is False
    assert boundary["address_numeric_rows_parsed"] == 0
    assert boundary["funding_numeric_rows_parsed"] == 0
    assert boundary["comparator_rows_parsed"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["post_2023_rows_read"] == 0
    assert payload["sources"]["numeric_rows_parsed_during_preregistration"] == 0

    unhashed = dict(payload)
    manifest_hash = unhashed.pop("manifest_hash")
    assert manifest_hash == prereg.canonical_hash(unhashed)


def test_payload_forbids_mark_price_during_source_support() -> None:
    payload = prereg.preregistration_payload()
    funding = payload["sources"]["funding"]
    assert funding["signal_allowed_columns"] == list(prereg.FUNDING_SIGNAL_COLUMNS)
    assert funding["exact_physical_columns"] == list(prereg.FUNDING_PHYSICAL_COLUMNS)
    assert "settlement_mark_price" in funding["forbidden_during_support"]
    assert "funding_time_utc + 5 minutes" == funding["availability"]


def test_payload_freezes_no_repair_sequence_and_economic_target() -> None:
    payload = prereg.preregistration_payload()
    sequence = payload["one_way_sequence"]
    assert sequence["stop_on_source_or_novelty_failure"] is True
    assert sequence["train_before_selection_outcome_transport"] is True
    assert sequence["llm_or_rl_rescue_forbidden"] is True
    assert payload["economic_gates"]["minimum_cagr_to_strict_mdd"] == 3.0
    assert payload["economic_gates"]["maximum_strict_mdd"] == 0.15
    weekly = payload["economic_gates"]["weekly_cluster_test"]
    assert weekly["draws"] == 100_000
    assert weekly["seed"] == 20_260_720
    assert weekly["cluster_order"] == "ascending UTC ISO (year, week)"
    assert weekly["run_separately_in_every_opened_split"] is True
    assert payload["strict_mdd_contract"]["global_high_water_mark"] is True
    assert payload["strict_mdd_contract"][
        "virtual_adverse_exit_cost_at_every_held_bar"
    ] is True
    assert payload["support_contract"]["all_concentration_checks_are_per_split"]


def test_payload_freezes_executable_novelty_and_control_contracts() -> None:
    payload = prereg.preregistration_payload()
    novelty = payload["novelty_contract"]
    assert novelty["minimum_common_support"] == {
        "candidate_events": 10,
        "comparator_events": 5,
        "failure_action": "fail closed, never mark not-applicable",
    }
    assert novelty["signed_exposure"]["overlap"] == (
        "any within-member overlap fails closed"
    )
    assert novelty["signed_exposure"]["zero_variance"] == "fail closed"
    assert payload["policy"]["controls"] == prereg.CONTROL_CONTRACTS
    assert payload["policy"]["controls"]["deterministic_random_side"][
        "material"
    ] == "AFDR-864|20260720|<primary_entry_time_utc>"


def test_preregistration_source_has_no_row_loader() -> None:
    source = Path(prereg.PREREGISTRATION_SOURCE).read_text(encoding="utf-8")
    forbidden = (
        "import pandas",
        "read_csv",
        "read_parquet",
        "gzip.open",
        "urlopen",
        "BTCUSDT_5m",
        "strict_bar_backtest",
    )
    assert all(token not in source for token in forbidden)


def test_all_frozen_comparators_are_hash_bound() -> None:
    for comparator in prereg.COMPARATORS:
        assert prereg.sha256_file(comparator["path"]) == comparator["sha256"]
        assert comparator["capability"] in {
            "directional_interval",
            "timestamp_only",
        }
        assert comparator["format"] in {"csv", "json_comparator_event_bundle"}


def test_write_once_is_idempotent_but_rejects_drift(tmp_path: Path) -> None:
    output = tmp_path / "prereg.json"
    prereg.write_once_json({"a": 1}, output)
    prereg.write_once_json({"a": 1}, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prereg.write_once_json({"a": 2}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 1}


def test_frozen_preregistration_artifact_matches_contract() -> None:
    path = Path(
        "results/address_funding_divergence_relay_preregistration_2026-07-20.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "f76a82881aab7560015c15b181e73b17d09e173fdfbb7a3001b1234e8be66220"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_hash = payload.pop("manifest_hash")
    assert manifest_hash == prereg.canonical_hash(payload)
    assert manifest_hash == (
        "f0174310c503f7a0428ff12a030e77c56e33ce07eb48f81f37850d62698b35e1"
    )
    assert payload["outcome_boundary"]["outcomes_opened"] is False
    assert payload["outcome_boundary"]["address_numeric_rows_parsed"] == 0
    assert payload["outcome_boundary"]["funding_numeric_rows_parsed"] == 0
