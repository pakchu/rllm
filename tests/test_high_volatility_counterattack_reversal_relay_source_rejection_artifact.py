import hashlib,json
from pathlib import Path
from training import build_high_volatility_counterattack_reversal_relay_support as support

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def test_counterattack_source_rejection_is_terminal_and_sealed()->None:
    result=json.loads(support.RESULT.read_text());core={k:v for k,v in result.items() if k!="manifest_hash"}
    assert result["manifest_hash"]==support.canonical_hash(core)
    assert result["support_passed"] is False and result["decision"]=="terminal_source_support_reject"
    assert result["advance_to_gross9_novelty"] is False and result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False and result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train","test","eval","final")]==[0,0,0,1]
    assert sha(support.RESULT)=="fc4b4430a5ba780292c2bd7dbae53e74b7eae163bde49ae17ec6ca3ecc908415"
    assert result["clock"]["sha256"]==sha(Path(result["clock"]["path"]))
