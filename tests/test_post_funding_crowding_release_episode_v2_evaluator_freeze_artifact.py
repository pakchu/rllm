from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import freeze_post_funding_crowding_release_episode_v2_evaluator as freeze


FREEZE = Path(
    "results/post_funding_crowding_release_episode_v2_evaluator_freeze_2026-07-17.json"
)
EXPECTED_SHA256 = "624d688509451a845c10c3e39d8493fb168194c55b1dc2b8b31607a0ed83a205"


def test_pfcr2_evaluator_freeze_artifact_is_locked_pre_outcome() -> None:
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(FREEZE.read_text())
    freeze.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["clock_rows"] == 82
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["mutable_parameters"] == []
