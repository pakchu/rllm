from training import preregister_cftc_asset_manager_option_position_second_week_relay as p


def test_frozen_contract():
 v=p.build();p.validate(v)
 assert v["policy_id"]=="CAMOP2W-168" and v["outcomes_opened"] is False
 assert v["features"]["conservative_availability"].startswith("report_date+7")
 assert v["clock"]["hold"]=="168 elapsed hours"
 assert v["diagnostic_controls"]["cannot_be_promoted"] is True
