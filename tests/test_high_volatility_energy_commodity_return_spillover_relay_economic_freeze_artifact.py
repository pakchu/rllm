import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_energy_commodity_return_spillover_relay_economics as e


FREEZE = Path("results/high_volatility_energy_commodity_return_spillover_relay_economic_evaluator_freeze_2026-08-11.json")


def test_freeze_is_outcome_blind_and_bound():
    value = json.loads(FREEZE.read_text())
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    assert value["manifest_hash"] == e.canonical_hash(core)
    assert value["outcomes_opened"] is False
    assert value["evaluator"]["sha256"] == hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest()
    assert value["authorization"]["sha256"] == e.NOVELTY_SHA
    assert value["empty_diagnostic_controls_handled_before_outcomes"] is True
