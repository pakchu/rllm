from __future__ import annotations
import copy
import pytest
from training import preregister_gross9_overlap_portfolio as p

def rehash(x):
 c={k:v for k,v in x.items() if k!='manifest_hash'};x['manifest_hash']=p.canonical_hash(c);return x

def test_prereg_freezes_universe_overlap_and_search():
 x=p.build();p.validate(x)
 assert x['immutable_universe']['canonical_sleeves']==71
 assert x['overlap_policy']['inter_sleeve_positions_allowed'] is True
 assert x['overlap_policy']['gross_risk_nets_opposite_positions'] is False
 assert x['search_grammar']['proxy_candidate_cap']==12000
 assert x['selection_windows']['holdout_opened_by_preregistration'] is False

def test_prereg_rejects_universe_and_boundary_drift():
 x=p.build();x['immutable_universe']['sha256']='0'*64
 with pytest.raises(RuntimeError,match='universe binding'):p.validate(rehash(x))
 x=p.build();x['selection_windows']['holdout_opened_by_preregistration']=True
 with pytest.raises(RuntimeError,match='holdout boundary'):p.validate(rehash(x))
