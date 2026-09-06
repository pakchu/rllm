import json
from training import evaluate_high_volatility_epu_hedge_relay_economics as economics
def test_train_economics_is_terminal_and_later_stages_are_closed():
 p=json.loads(economics.OUTPUTS["train"].read_text());assert p["policy_id"]=="HVEPUH-24";assert p["stage"]=="train";assert not p["passed"];assert p["decision"]=="terminal_reject_no_repair";assert not p["later_stage_outcomes_opened"];assert not p["advance_to_next_stage"];assert p["primary"]["base"]["absolute_return_pct"]<0;assert p["primary"]["base"]["mean_gross_underlying_bp"]<20;assert not p["checks"]["each_calendar_half_positive"];h=p.pop("manifest_hash");assert economics.canonical_hash(p)==h
