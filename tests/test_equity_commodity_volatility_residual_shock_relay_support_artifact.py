import hashlib,json
from pathlib import Path
from training import build_equity_commodity_volatility_residual_shock_relay_support as support


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ecvrs_source_support_pass_is_outcome_sealed():
    assert sha(support.RESULT)=="ca5080d27fd0f98f079f31c2c86f5e1797c310aec5071916881d8e3510bbf9bb"
    result=json.loads(support.RESULT.read_text())
    assert result["policy_id"]=="ECVRS-12"
    assert result["support_passed"] is True
    assert result["decision"]=="pass_to_novelty"
    assert result["advance_to_gross9_novelty"] is True
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train","test","eval","final")]==[9,45,27,23]


def test_ecvrs_hashes_and_controls_are_bound():
    result=json.loads(support.RESULT.read_text())
    assert result["manifest_hash"]==support.canonical_hash({key:value for key,value in result.items() if key!="manifest_hash"})
    assert result["clock"]["sha256"]==sha(Path(result["clock"]["path"]))
    assert result["source_manifest"]["sha256"]==sha(Path(result["source_manifest"]["path"]))
    for control in result["controls"].values():
        assert control["promotion_authorized"] is False
        assert control["sha256"]==sha(Path(control["path"]))
