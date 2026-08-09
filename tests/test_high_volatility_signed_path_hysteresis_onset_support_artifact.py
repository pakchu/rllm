import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_signed_path_hysteresis_onset_support_2026-08-10.json")


def test_terminal_source_support_artifact_is_sealed() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "1249203ba295462dd7785c22e6cd7fb6d2acdb84ff321b74ba4d6fbebd919f0b"
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
    assert payload["support_checks"]["final_month_concentration"] is False
    assert payload["oos_postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
