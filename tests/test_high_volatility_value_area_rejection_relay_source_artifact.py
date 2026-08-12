import hashlib
import json

from training import build_high_volatility_value_area_rejection_relay_support as source


def test_hvvar_source_pass_is_frozen_and_reproduced() -> None:
    assert hashlib.sha256(source.RESULT.read_bytes()).hexdigest() == (
        "84fdc25d3284495ae44efebd92798486f4e54f81f74c510f6dda1e2351af9734"
    )
    result = json.loads(source.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == source.canonical_hash(core) == (
        "4c6d92cd3e76731d7ad5fdce4a921de1c54fa129e7b2310102b160a322285647"
    )
    assert result["support_passed"] is True
    assert all(result["support_checks"].values())
    assert result["clock"]["sha256"] == (
        "4080e24e99f0871bd2cfded81bd37518f3354af16a490558d5551fbf9a2e708d"
    )
    assert result["source_manifest"]["sha256"] == (
        "3d95e330529d969c8cdc5a465b92ebb7fb9156aa7f405c3f65f395aad1f9036d"
    )
    assert result["advance_to_gross9_novelty"] is True
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["funding_opened"] is False
    assert result["gross9_rows_opened"] is False
