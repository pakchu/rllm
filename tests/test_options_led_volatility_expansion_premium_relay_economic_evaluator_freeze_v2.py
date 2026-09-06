from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v2 as e,
)


FREEZE = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "economic_evaluator_freeze_v2_2026-08-08.json"
)


def test_v2_evaluator_freeze_is_hash_bound_and_outcome_blind() -> None:
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == (
        "b61ad6a5a38d0c01a4a1e2cfa1a66ca6ce1e183306e9359abf83fcf5c7dd166a"
    )
    payload = json.loads(FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == e.chash(core) == (
        "7de5f06ccca47fd371c03ac91127384e32476f70aa033ad5c3adb5c00da24e46"
    )
    assert payload["evaluator"]["sha256"] == (
        "bb436eeabff1ff3d623d59f3f4c410d61cf0ec2313948a567371329381016460"
    )
    assert payload["v1_terminal_evidence"]["economic_metrics_computed"] is False
    assert payload["candidate_contract_change_from_v1"] is False
    assert payload["outcomes_opened"] is False
