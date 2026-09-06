from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v4 as e,
)


FREEZE = Path(
    "results/options_led_volatility_expansion_premium_relay_"
    "economic_evaluator_freeze_v4_2026-08-08.json"
)


def test_v4_freeze_is_hash_bound_outcome_blind_and_accounting_identical() -> None:
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == (
        "fc9a245ade957f84f1e65114d10bc8fada95a0e7020cda918854e118f848ccb3"
    )
    payload = json.loads(FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["manifest_hash"] == e.chash(core) == (
        "69bc4405113563bba7c94fbb0db0cadfda2070e76390b5703b24b17d8ec6218a"
    )
    assert payload["evaluator"]["sha256"] == (
        "c353d6fd1a75fe0574a035c9adf8ac11f628e9be1a6f743a9b0efd568659cd17"
    )
    assert payload["candidate_contract_change_from_v3"] is False
    assert payload["economic_contract_inherited_exactly_from_v3"] is True
    assert payload["outcomes_opened"] is False
