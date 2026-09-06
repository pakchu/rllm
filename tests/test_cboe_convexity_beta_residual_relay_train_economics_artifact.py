import json
def test_ccbrr_train_economics_is_terminal_and_later_stages_closed():
 r=json.load(open("results/cboe_convexity_beta_residual_relay_train_economics_2026-08-08.json"));assert r["stage"]=="train";assert r["passed"] is False;assert r["decision"]=="terminal_reject_no_repair";assert r["advance_to_next_stage"] is False;assert r["later_stage_outcomes_opened"] is False;assert r["primary"]["base"]["absolute_return_pct"]<0;assert r["primary"]["stress"]["absolute_return_pct"]<0
