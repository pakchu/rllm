import hashlib
import json
from pathlib import Path

from training import preregister_options_crowding_deleveraging_relay as p


ARTIFACT = Path(
    "results/options_crowding_deleveraging_relay_preregistration_2026-08-08.json"
)


def test_preregistration_artifact_is_frozen_before_candidate_incidence() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "90287c37c861c1c8e2b690f0bfb2768cc6dfa490bd6677a2d2c6654929296be0"
    )
    report = json.loads(ARTIFACT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == p.canonical_hash(core)
    assert report["research_boundary"]["ocdr_incidence_opened"] is False
    assert report["research_boundary"]["ocdr_price_or_return_rows_opened"] is False
