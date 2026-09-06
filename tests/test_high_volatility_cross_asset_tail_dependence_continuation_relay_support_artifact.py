import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_cross_asset_tail_dependence_continuation_relay_support_2026-08-10.json")


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_source_rejection_artifact_is_hash_bound_and_terminal() -> None:
    payload = json.loads(RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert payload["manifest_hash"] == expected
    assert payload["source"]["state"]["sha256"] == sha(payload["source"]["state"]["path"])
    assert payload["clock"]["sha256"] == sha(payload["clock"]["path"])
    for binding in payload["controls"].values():
        assert binding["sha256"] == sha(binding["path"])
        assert binding["promotion_authorized"] is False
    assert payload["support_passed"] is False
    assert payload["decision"] == "terminal_source_support_reject"
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
