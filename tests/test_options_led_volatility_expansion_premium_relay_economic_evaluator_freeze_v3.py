from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v3 as e,
)


FREEZE = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "economic_evaluator_freeze_v3_2026-08-08.json"
)


def test_v3_freeze_is_hash_bound_outcome_blind_and_accounting_identical() -> None:
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == (
        "b0bfa10fdb6af6f979ed4ce7e1fc6c724ff4b094de93f2b8a0904989a9d03f98"
    )
    payload = json.loads(FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == e.chash(core) == (
        "7c3027e872bd5914995780dc0ecbafe6e11e95f97a6dbbc25fa6ce54c77031d3"
    )
    assert payload["evaluator"]["sha256"] == (
        "f9a9e5bad82f3e914868e1beea1909aa967509c455463fa671105ad8486b9498"
    )
    assert payload["candidate_contract_change_from_v2"] is False
    assert payload["economic_contract_inherited_exactly_from_v2"] is True
    assert payload["outcomes_opened"] is False
