from training import audit_family_champion_alphas as s

def test_report_gate_is_fixed_and_strict():
 g=s.DESIGN['report_gate'];assert g['positive_each_period'];assert g['combined_10bp_return_positive'];assert g['combined_mdd_max']==15.;assert g['combined_entries_min']==30
 assert s.DESIGN['no_report_rerank'] and not s.DESIGN['live_enabled']
