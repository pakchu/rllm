import hashlib,json
from pathlib import Path
from training import build_dollar_factor_shock_relay_support as support


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dfsr_source_support_pass_is_outcome_sealed():
    assert sha(support.RESULT)=="57018248d49fa0fa2a142adbffe2b42ae1d12fb55b762f84e4334cb8f21d46b1"
    result=json.loads(support.RESULT.read_text())
    assert result["policy_id"]=="DFSR-12"
    assert result["support_passed"] is True
    assert result["decision"]=="pass_to_novelty"
    assert result["advance_to_gross9_novelty"] is True
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train","test","eval","final")]==[14,31,31,17]


def test_dfsr_hashes_and_controls_are_bound():
    result=json.loads(support.RESULT.read_text())
    assert result["manifest_hash"]==support.chash({key:value for key,value in result.items() if key!="manifest_hash"})
    assert result["clock"]["sha256"]==sha(Path(result["clock"]["path"]))
    for control in result["controls"].values():
        assert control["promotion_authorized"] is False
        assert control["sha256"]==sha(Path(control["path"]))
