import json
from pathlib import Path


def test_source_support_passes_without_outcomes():
 v=json.loads(Path("results/options_risk_peak_leverage_handoff_continuation_support_2026-08-11.json").read_text())
 assert v["policy_id"]=="ORPLHC-6" and v["support_passed"] is True
 assert {k:x["events"] for k,x in v["support"].items()}=={"train":29,"test":35,"eval":47,"final":30}
 assert v["postentry_return_pnl_execution_price_opened"] is False and v["gross9_rows_opened"] is False
 assert all(not x["promotion_authorized"] for x in v["controls"].values())
