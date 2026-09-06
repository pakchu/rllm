import numpy as np
from training import build_high_volatility_spot_perpetual_sign_transition_leadership_relay_support as subject

def test_transition_metric_is_strict_and_causal_shape():
 s=np.array([1.,2.,1.,2.,1.]);p=np.array([1.,1.5,1.,1.5,1.]);x=subject.transition_metrics(s,p);assert x["spot_transition_pairs"]==3 and x["perp_transition_pairs"]==3;assert "leadership_advantage" in x
def test_clock_contract():
 assert subject.COLUMNS[0:4]==("candidate","control","split","decision_time");assert subject.PREREG_SHA=="083d31c6caa3286d4bc3f150cd694ea3a05d10c30073b8e724078c5e38798d7f"
