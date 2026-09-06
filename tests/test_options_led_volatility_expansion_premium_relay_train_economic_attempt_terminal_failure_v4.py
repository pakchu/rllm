from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v4 as e,
)


RECEIPT = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "train_economic_attempt_terminal_failure_v4_2026-08-08.json"
)


def test_v4_failure_receipt_proves_no_metric_or_candidate_verdict_opened() -> None:
    assert hashlib.sha256(RECEIPT.read_bytes()).hexdigest() == (
        "9c6a5d71212f00974930fcedbec176c60b7f9a70fc7a83fbc1d49be2bd8558d4"
    )
    payload = json.loads(RECEIPT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == e.chash(core) == (
        "5e06ceab638cc104a09e499775adf7c058c3547543cf71557f0dfc6f3caa9c4b"
    )
    assert payload["economic_metrics_computed"] is False
    assert payload["candidate_pass_fail_observed"] is False
    assert payload["later_windows_opened"] is False
    assert payload["attempt_disposition"] == "TERMINAL_FAILURE_NO_RETRY_UNDER_V4"
