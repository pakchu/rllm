from training import evaluate_regional_trend_fresh_v2 as v2
from training import evaluate_regional_trend_fresh as v1

def test_v2_is_receipt_only_correction():
 assert v2.DESIGN['version']==2
 assert v2.DESIGN['correction'].startswith('availability receipt')
 assert v2.DESIGN['candidate']==v1.DESIGN['candidate']
 assert v2.DESIGN['fresh_window']==v1.DESIGN['fresh_window']
 assert v2.DESIGN['costs']==v1.DESIGN['costs']
