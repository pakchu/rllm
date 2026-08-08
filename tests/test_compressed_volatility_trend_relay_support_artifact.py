import hashlib
import json
from pathlib import Path

from training import build_compressed_volatility_trend_relay_support as support


RESULT = Path("results/compressed_volatility_trend_relay_support_2026-08-08.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cvtr_support_is_frozen_terminal_before_novelty_and_economics():
    assert digest(RESULT) == "0367628adce5fc7eb66326d5d5ff1674bbdf624acb98c98a027589434265d2ef"
    report = json.loads(RESULT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core) == "a5c8c10d18bfe39e3ae3bee8986b0098ee7d3d85594a16f4d965fc61832dee6d"
    assert report["clock"]["sha256"] == digest(support.CLOCK)
    assert report["support_checks"]["train_minimum_events"] is False
    assert report["support_checks"]["final_minimum_events"] is False
    assert report["support_checks"]["final_side_balance"] is False
    assert report["support_passed"] is False
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
