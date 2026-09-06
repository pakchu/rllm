from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v2 as e,
)


RECEIPT = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "train_economic_attempt_terminal_failure_v2_2026-08-08.json"
)


def test_v2_failure_receipt_proves_no_outcome_or_metric_was_opened() -> None:
    assert hashlib.sha256(RECEIPT.read_bytes()).hexdigest() == (
        "fc933789698833ad5f8f7528555512b6cd6e68aa1124d31763567523ee6ad55a"
    )
    payload = json.loads(RECEIPT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == e.chash(core) == (
        "008e2ca94bf178007d1154222e45cac3f84750822eaabf24bc93086c9cc709f3"
    )
    assert payload["economic_metrics_computed"] is False
    assert payload["candidate_pass_fail_observed"] is False
    assert payload["btc_execution_source_opened"] is False
    assert payload["attempt_disposition"] == "TERMINAL_FAILURE_NO_RETRY_UNDER_V2"
