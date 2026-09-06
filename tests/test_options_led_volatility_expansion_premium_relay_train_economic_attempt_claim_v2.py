from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v2 as e,
)


CLAIM = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "train_economic_attempt_claim_v2_2026-08-08.json"
)


def test_v2_train_claim_is_hash_bound_and_keeps_later_windows_sealed() -> None:
    assert hashlib.sha256(CLAIM.read_bytes()).hexdigest() == (
        "d8485dc01573c04a6a6e8a95a81bece594eb9a0c3e766e18ce60278796e1e593"
    )
    payload = json.loads(CLAIM.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == e.chash(core) == (
        "cf7a0e3bfea02d93848d1e8a8ad5d5a7b8fc0c8ab71f22feaf1f58a2ac16063f"
    )
    assert payload["authoritative_attempts_allowed"] == 1
    assert payload["outcomes_opened"] is False
    assert payload["sealed_later_windows"] == [
        "test_2024",
        "eval_2025",
        "final_2026H1",
    ]
