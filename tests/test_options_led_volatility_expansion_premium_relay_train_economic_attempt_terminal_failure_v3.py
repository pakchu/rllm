from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v3 as e,
)


RECEIPT = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "train_economic_attempt_terminal_failure_v3_2026-08-08.json"
)


def test_v3_failure_receipt_proves_no_numeric_outcome_row_or_metric_opened() -> None:
    assert hashlib.sha256(RECEIPT.read_bytes()).hexdigest() == (
        "a721a0566e703c25b7221037125af6c0149df204ff55de64742d1e84ad498372"
    )
    payload = json.loads(RECEIPT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == e.chash(core) == (
        "d310bbc1f4986ef04d0cce376f558caae239d335d19d730c656c55f1b63ecda6"
    )
    assert payload["economic_metrics_computed"] is False
    assert payload["btc_execution_numeric_rows_opened"] is False
    assert payload["funding_rows_opened"] is False
    assert payload["attempt_disposition"] == "TERMINAL_FAILURE_NO_RETRY_UNDER_V3"
