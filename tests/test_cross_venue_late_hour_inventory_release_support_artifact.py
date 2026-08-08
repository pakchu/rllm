import hashlib
import json

from training import build_cross_venue_late_hour_inventory_release_support as support


def test_cvlir_support_is_frozen_terminal_before_novelty_and_economics():
    assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest() == "883f25050597d9e570be7672b307f9466e23e4f95330ea0564a2c5c7dde41975"
    report = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core) == "12f6f0f517cdfe8320601a78912a66345d6afd7abb252a753525df4e3df9db91"
    assert report["clock"]["sha256"] == hashlib.sha256(support.CLOCK.read_bytes()).hexdigest()
    assert report["support_passed"] is False
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
