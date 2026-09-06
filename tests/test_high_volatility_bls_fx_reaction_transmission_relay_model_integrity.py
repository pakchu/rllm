import hashlib,json
from training import evaluate_high_volatility_bls_fx_reaction_transmission_relay_model_integrity as e

def test_calendar_is_official_selected_unique_and_frozen():
 assert e.sha(e.CALENDAR)==e.CALENDAR_SHA
 x=__import__('pandas').read_csv(e.CALENDAR);assert len(x)==175;assert x.release_date.nunique()==175;assert x.release_time_et.eq('08:30 AM').all();assert x.source_url.str.startswith('https://www.bls.gov/schedule/').all()
def test_terminal_history_floor_failure_is_outcome_blind():
 r=e.run();assert r['model_integrity_passed'] is False;assert r['feasibility']=={'selected_releases_before_train':30,'selected_releases_before_train_end':60,'maximum_possible_ranked_train_events':0};assert r['decision']=='terminal_preregistered_history_floor_failure';assert not r['candidate_incidence_opened'] and not r['fx_or_btc_feature_values_opened'] and not r['postentry_outcomes_opened'] and not r['gross9_rows_opened'];h=r.pop('manifest_hash');assert e.chash(r)==h
