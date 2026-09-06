import hashlib
import json
from pathlib import Path

from training import build_cross_venue_disagreement_absorption_support as support


RESULT = Path("results/cross_venue_disagreement_absorption_relay_support_2026-08-08.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cvdar_support_is_frozen_terminal_only_on_eval_count():
    assert digest(RESULT) == "37a88ecdf9cbcbafe611d2cef1fbe2ab27606e6aa586f8bdf3a1a4089506ea47"
    report = json.loads(RESULT.read_text()); core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core) == "ca9be5138b7713059642065fea87baa2aba614fb12783ddc6d07ad20941b222d"
    assert report["clock"]["sha256"] == digest(support.CLOCK)
    failed = [name for name, passed in report["support_checks"].items() if not passed]
    assert failed == ["eval_minimum_events"]
    assert report["support_passed"] is False
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
