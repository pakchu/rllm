import hashlib
import json
from pathlib import Path

from training.preregister_options_crowding_deleveraging_relay import canonical_hash


P = Path("results/options_crowding_deleveraging_relay_source_support_veto_2026-08-08.json")


def test_v1_source_veto_is_hash_bound_before_incidence_or_outcomes() -> None:
    assert hashlib.sha256(P.read_bytes()).hexdigest() == (
        "bce9029503503977f1586e7e6428a741f92ddd7f2707e8c78257b219b4c839f5"
    )
    report = json.loads(P.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == canonical_hash(core)
    assert report["candidate_incidence_opened"] is False
    assert report["economic_outcomes_opened"] is False
    assert report["decision"] == "TERMINAL_SOURCE_SUPPORT_REJECT_NO_RETRY_UNDER_V1"
