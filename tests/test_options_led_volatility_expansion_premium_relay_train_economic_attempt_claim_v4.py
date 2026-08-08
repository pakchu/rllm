from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v4 as e,
)


CLAIM = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "train_economic_attempt_claim_v4_2026-08-08.json"
)


def test_v4_train_claim_is_hash_bound_and_later_windows_remain_sealed() -> None:
    assert hashlib.sha256(CLAIM.read_bytes()).hexdigest() == (
        "4c3eeaf194b54f9e283797ff3b30f5c587e7f813944c74338df114b2b5410175"
    )
    payload = json.loads(CLAIM.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == e.chash(core) == (
        "55913163fe50aee739f936b3ed90ac86515f8a03169202ba7c05c830f2ca4fe0"
    )
    assert payload["temporary_gzip_row_regression_passed"] is True
    assert payload["preflight_outcomes_opened"] is False
    assert payload["authoritative_attempts_allowed"] == 1
    assert payload["sealed_later_windows"] == [
        "test_2024",
        "eval_2025",
        "final_2026H1",
    ]
