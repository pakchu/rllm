import numpy as np
from training import search_long_complement_shorts as s


def test_hold_has_no_future_influence():
    p,_=s.pulse_hold([False,True,False,False,False],3)
    assert p.tolist()==[0,-1,-1,-1,0]
    p2,_=s.pulse_hold([False,True,False,False,True],3)
    assert np.array_equal(p[:4],p2[:4])


def test_specs_short_complement_no_standalone_gate():
    specs=s.candidate_specs()
    assert len(specs)==148
    assert len({x['name'] for x in specs})==148
    assert specs[0]['coefficient']==0
    assert 'standalone profitability not required' in s.DESIGN['objective']
