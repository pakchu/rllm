import hashlib
import json
from pathlib import Path

from training import build_options_led_intrahour_absorption_support as support
from training import materialize_options_led_intrahour_absorption_sources as sources


SOURCE_MANIFEST = Path("data/options_led_intrahour_absorption_sources_2023_2026/manifest.json")
RESULT = Path("results/options_led_intrahour_absorption_handoff_support_2026-08-08.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_oliah_source_snapshot_is_feature_only_and_frozen():
    assert digest(SOURCE_MANIFEST) == "1ca33578613d084801c02fce5f15213daa6c5a419ad621b056648b2f01e276e2"
    report = json.loads(SOURCE_MANIFEST.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == sources.canonical_hash(core) == "4a4cc319371c932275d92a234bdaea4ae5f3e67f21526db0d98ca26bae5b8fcd"
    assert report["post_entry_return_pnl_or_execution_price_opened"] is False
    assert report["candidate_incidence_opened"] is False


def test_oliah_support_is_terminal_on_frozen_eval_count():
    assert digest(RESULT) == "36624b35677b392191cc2a777c644b82872490604454623608ae94cd19384c7f"
    report = json.loads(RESULT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core) == "dc3b702b528e760ddbbf919dc9aeeb9b5fda312ded42f8f53eaac3367aa25698"
    assert report["clock"]["sha256"] == digest(support.CLOCK)
    assert report["support"]["eval"]["events"] == 11
    assert report["support_checks"]["eval_minimum_events"] is False
    assert report["support_passed"] is False
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
