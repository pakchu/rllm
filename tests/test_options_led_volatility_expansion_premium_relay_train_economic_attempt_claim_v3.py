from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v3 as e,
)


CLAIM = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "train_economic_attempt_claim_v3_2026-08-08.json"
)


def test_v3_train_claim_is_hash_bound_and_later_windows_remain_sealed() -> None:
    assert hashlib.sha256(CLAIM.read_bytes()).hexdigest() == (
        "1dc97d0a8b7c2bc6c8addd2a992a2fbfdcd0e87cec7cc19322d1f399872620e1"
    )
    payload = json.loads(CLAIM.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == e.chash(core) == (
        "4a29577c0f14c89d05dc9103802f4e0f86536503d092907ed70674e7197252f0"
    )
    assert payload["direct_cli_preflight_passed"] is True
    assert payload["preflight_outcomes_opened"] is False
    assert payload["authoritative_attempts_allowed"] == 1
    assert payload["sealed_later_windows"] == [
        "test_2024",
        "eval_2025",
        "final_2026H1",
    ]
