from training import evaluate_oi_divergence_fresh_v2 as v2
from training import evaluate_oi_divergence_fresh as v1

def test_only_key_aliases_change():
 assert v2.DESIGN['version']==2;assert v2.DESIGN['candidate_config']==v1.DESIGN['candidate_config'];assert v2.DESIGN['signal_window']==v1.DESIGN['signal_window'];assert v2.DESIGN['costs']==v1.DESIGN['costs']
