import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_dydx_auto_deleveraging_handoff_relay_support_2026-08-10.json"
)
SOURCE = Path(
    "data/high_volatility_dydx_auto_deleveraging_handoff_relay_sources_2023_2026/manifest.json"
)


def test_terminal_source_support_artifact_is_sealed() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "9272a66db86c6591f42e0fa80a3e8dae4f3375974227620f2cee2b3deb1e89bf"
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
    assert payload["support_checks"]["train_side_balance"] is False
    assert payload["support_checks"]["train_month_concentration"] is False
    assert payload["support"]["train"]["deleveraged_events"] == 0
    assert payload["oos_postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_complete_cursor_replay_and_source_boundaries_are_sealed() -> None:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == (
        "b2104e476ff80d3463edc10000694ce7e37c82de098f3a85c0fc1f3f1c729061"
    )
    payload = json.loads(SOURCE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert payload["manifest_hash"] == expected
    assert payload["page_manifest"]["pages"] == 44841
    assert payload["page_manifest"]["all_trade_ids_seen"] == 44840849
    assert payload["forced_trades"]["rows"] == 158835
    assert payload["panel"]["valid_rows"] == 23778
    assert payload["oos_postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
