import hashlib
import json

from training import build_high_volatility_oi_conditioned_late_variation_ignition_relay_support as s


def test_hvoilvi_source_pass_is_frozen_and_reproduced():
    assert hashlib.sha256(s.RESULT.read_bytes()).hexdigest() == "48c52efa0582373f88d44c0f9b57f818816e6a2edc8ccfe3270315d9a1702348"
    result = json.loads(s.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == s.prereg.canonical_hash(core) == "1de9e67782044d7a041c3dea6bc05bedbd100d3648a9950c1a35b4cb40471763"
    assert result["support_passed"] is True and all(result["support_checks"].values())
    assert result["clock"]["sha256"] == "e4c36e8fffb4805dc397808a246f9cddf54d80cc40153f019e8d773fcdecc719"
    assert result["advance_to_gross9_novelty"] is True
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
