import hashlib,json
from pathlib import Path
R=Path("results/cftc_micro_standard_positioning_segmentation_relay_support_2026-08-11.json")
def test_frozen_support_pass():
 assert hashlib.sha256(R.read_bytes()).hexdigest()=="ef20ff133ddaf9845472a1c053f61525960235fbef373dd3b76d6f33557f79c2";x=json.loads(R.read_text());assert x["support_passed"] is True and x["advance_to_gross9_novelty"] is True and x["advance_to_economic_outcomes"] is False;assert x["postentry_return_pnl_execution_price_opened"] is False and x["gross9_rows_opened"] is False;assert [x["support"][n]["events"] for n in ("train","test","eval","final")]==[12,42,26,18];assert all(x["support_checks"].values())
