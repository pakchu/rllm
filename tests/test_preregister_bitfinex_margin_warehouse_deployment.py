from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from training import preregister_bitfinex_margin_warehouse_deployment as prereg


def test_contract_freezes_four_variants_and_direction() -> None:
    prereg.validate_contract()
    assert len(prereg.VARIANTS) == 4
    assert {variant.warehouse_hours for variant in prereg.VARIANTS} == {12, 24}
    assert {variant.deployment_hours for variant in prereg.VARIANTS} == {3, 6}
    assert {variant.hold_bars for variant in prereg.VARIANTS} == {144}
    assert prereg.FROZEN_POLICY.usd_side == 1
    assert prereg.FROZEN_POLICY.btc_side == -1


def test_payload_is_outcome_blind_and_self_hashed() -> None:
    payload = prereg.preregistration_payload()
    assert payload["decision"] == (
        "freeze_source_support_before_source_values_or_any_outcome"
    )
    boundary = payload["outcome_boundary"]
    assert boundary["outcomes_opened"] is False
    assert boundary["source_numeric_rows_opened"] is False
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["post_2023_rows_read"] == 0
    manifest_hash = payload.pop("manifest_hash")
    assert manifest_hash == prereg.canonical_hash(payload)


def test_payload_freezes_strict_prior_features_and_stopping_rules() -> None:
    payload = prereg.preregistration_payload()
    policy = payload["policy"]
    assert "strictly prior 1440" in policy["features"]["standardization"]
    assert policy["direction"] == "fUSD -> LONG (+1); fBTC -> SHORT (-1)"
    assert payload["one_way_sequence"]["failed_candidate_repair_forbidden"] is True
    assert payload["one_way_sequence"]["llm_rescue_forbidden"] is True
    assert payload["economic_gates"]["minimum_cagr_to_strict_mdd"] == 3.0
    assert payload["economic_gates"]["maximum_strict_mdd"] == 0.15


def test_write_once_refuses_different_payload(tmp_path) -> None:
    output = tmp_path / "prereg.json"
    prereg.write_once_json({"a": 1}, output)
    prereg.write_once_json({"a": 1}, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prereg.write_once_json({"a": 2}, output)
    assert json.loads(output.read_text()) == {"a": 1}


def test_preregistration_module_has_no_source_or_outcome_loader() -> None:
    text = Path(prereg.PREREGISTRATION_SOURCE).read_text()
    forbidden = (
        "pandas",
        "read_csv",
        "read_parquet",
        "urlopen",
        "strict_bar_backtest",
        "BTCUSDT_5m",
    )
    assert all(token not in text for token in forbidden)


def test_frozen_preregistration_artifact_matches_contract() -> None:
    path = Path(
        "results/bitfinex_margin_warehouse_deployment_preregistration_2026-07-20.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "6e478bac6becb58d282867f4ee612d9d13e803d01985474477d6e3073cd49e58"
    )
    payload = json.loads(path.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert manifest_hash == prereg.canonical_hash(payload)
    assert manifest_hash == (
        "9b5d78e506393340c912e6d0fc965a9f2e129594c242d6072fb92c7ed84fb81d"
    )


def test_frozen_novelty_comparators_are_hash_bound_before_source_access() -> None:
    path = Path(
        "results/bitfinex_margin_warehouse_deployment_comparator_freeze_2026-07-20.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "37ee403d33b5361c752b84ef94d46a05d991d82e1b0a77338b41a6c49e8410de"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["preregistration_sha256"] == (
        "6e478bac6becb58d282867f4ee612d9d13e803d01985474477d6e3073cd49e58"
    )
    assert payload["outcome_boundary"] == {
        "bitfinex_numeric_rows_read": 0,
        "btc_market_rows_read": 0,
        "comparator_rows_read": 0,
        "funding_rows_read": 0,
        "outcomes_opened": False,
        "post_2023_rows_read": 0,
        "return_or_pnl_fields_read": 0,
    }
    for comparator in payload["comparators"]:
        comparator_path = Path(comparator["path"])
        assert comparator["control"] == "primary"
        assert hashlib.sha256(comparator_path.read_bytes()).hexdigest() == comparator[
            "sha256"
        ]
