import hashlib,json
from pathlib import Path
from training import build_high_volatility_rogers_satchell_excursion_asymmetry_reversal_relay_support as s
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_hvrsar_source_pass_is_reproducible_and_outcomes_sealed():
 x=json.loads(s.RESULT.read_text());core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==s.canonical_hash(core)
 assert x["policy_id"]=="HVRSAR-8" and x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False and x["decision"]=="pass_to_novelty"
 assert x["postentry_return_pnl_execution_price_opened"] is False and x["funding_values_opened"] is False and x["gross9_rows_opened"] is False
 assert [x["support"][k]["events"] for k in ("train","test","eval","final")]==[42,86,79,35] and all(x["support_checks"].values())
 assert x["clock"]["rows"]==242 and x["clock"]["sha256"]==sha(Path(x["clock"]["path"]))
 assert sha(s.RESULT)=="79c1b9d092087e49aadc85dc9b156006a286fee98431484daf24b82773a9b046"
