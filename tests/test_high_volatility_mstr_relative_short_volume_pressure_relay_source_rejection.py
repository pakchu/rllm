import hashlib
import json
from pathlib import Path


ARTIFACT = Path("results/high_volatility_mstr_relative_short_volume_pressure_relay_source_rejection_2026-08-10.json")


def test_transport_rejection_is_terminal_before_market_or_outcomes() -> None:
    payload = json.loads(ARTIFACT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert payload["manifest_hash"] == expected
    assert payload["decision"]["status"] == "terminal_source_transport_rejection"
    assert payload["decision"]["repair_authorized"] is False
    assert payload["access_boundary"]["candidate_incidence_derived"] is False
    assert payload["access_boundary"]["postgres_connected"] is False
    assert payload["access_boundary"]["economic_outcomes_opened"] is False
