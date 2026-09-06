import hashlib,json
from pathlib import Path
from training import build_equity_commodity_volatility_divergence_relay_support as support


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ecvdr_source_rejection_is_sealed():
    assert sha(support.RESULT)=="1cb4be281d3cc682b935f5866531ef79876d4b94559b3830b2c8646f503c18bb"
    result=json.loads(support.RESULT.read_text())
    assert result["policy_id"]=="ECVDR-12"
    assert result["support_passed"] is False
    assert result["decision"]=="terminal_source_support_reject"
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train","test","eval","final")]==[2,12,9,9]


def test_ecvdr_hashes_and_controls_are_bound():
    result=json.loads(support.RESULT.read_text())
    assert result["manifest_hash"]==support.canonical_hash({key:value for key,value in result.items() if key!="manifest_hash"})
    assert result["clock"]["sha256"]==sha(Path(result["clock"]["path"]))
    assert result["source_manifest"]["sha256"]==sha(Path(result["source_manifest"]["path"]))
    for control in result["controls"].values():
        assert control["promotion_authorized"] is False
        assert control["sha256"]==sha(Path(control["path"]))
