import hashlib,json
from pathlib import Path
R=Path("results/high_volatility_flow_impact_convexity_relay_support_2026-08-13.json")
def test_terminal_source_artifact():
 x=json.loads(R.read_text());assert x["support_passed"] is False;assert x["advance_to_gross9_novelty"] is False;assert x["advance_to_economic_outcomes"] is False;assert x["support"]["train"]["events"]==1;assert x["support"]["test"]["events"]==8;assert x["gross9_rows_opened"] is False;assert hashlib.sha256(R.read_bytes()).hexdigest()=="3c17b613cde2ef3e3ceb3378b1344e91eaaa1f517e973264a52ef06da4f34c8a"
