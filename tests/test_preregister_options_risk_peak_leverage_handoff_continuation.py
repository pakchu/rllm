from training import preregister_options_risk_peak_leverage_handoff_continuation as p


def test_frozen_contract():
 v=p.build();p.validate(v)
 assert v["policy_id"]=="ORPLHC-6" and v["outcomes_opened"] is False
 assert v["clock"]["side"]=="cooling-hour premium displacement sign"
 assert v["policy"]["prior_dvol_body_rank_min"]==.75
 assert v["diagnostic_controls"]["cannot_be_promoted"] is True
