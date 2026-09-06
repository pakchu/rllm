import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_sha256_chain_fee_migration_relay_support_2026-08-10.json"
)
SOURCE = Path(
    "data/high_volatility_sha256_chain_fee_migration_relay_sources_2023_2026/manifest.json"
)


def test_terminal_source_support_artifact_is_sealed() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "2703d730a8d715f6900b612636b23f77706aa2f827be6263837f365b4b5cab72"
    )
    payload = json.loads(RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert payload["manifest_hash"] == expected
    assert payload["support_passed"] is False
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["support_checks"]["eval_minimum_events"] is False
    assert payload["support"]["eval"]["events"] == 5
    assert payload["oos_postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_complete_pair_source_is_sealed() -> None:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == (
        "5cc255700209c7232769cfc95eb970a817b230bd8cd4c5c5f6892d56ebf88ff2"
    )
    payload = json.loads(SOURCE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert payload["manifest_hash"] == expected
    assert payload["transport"]["raw_rows"] == 2616
    assert payload["pair_panel"]["rows"] == 1308
    assert payload["feature_panel"]["valid_rows"] == 1308
    assert payload["oos_postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
