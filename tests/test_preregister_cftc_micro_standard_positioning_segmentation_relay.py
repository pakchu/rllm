from training import preregister_cftc_micro_standard_positioning_segmentation_relay as p
def test_preregistration_is_sealed_and_distinct():
 x=p.build();p.validate(x);assert x==p.build();assert x["policy_id"]=="CFTCMSPS-168";assert x["features"]["standard_contract"]["code"]=="133741";assert x["features"]["micro_contract"]["code"]=="133742";assert x["policy"]["variation_rank_min"]==.5;assert x["clock"]["hold"]=="168 elapsed hours";assert not x["outcomes_opened"] and not x["source_incidence_opened"] and not x["gross9_rows_opened"]
