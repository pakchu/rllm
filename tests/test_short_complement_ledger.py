import numpy as np
from training import short_complement_ledger as hedge
from training import g9_joint_net_ledger as base
from tests.test_g9_joint_net_ledger import data


def test_zero_hedge_is_identical_to_existing_ledger():
    d=data();p=np.array([[1.,0.],[1.,0.]]);e=np.array([[1,0],[0,0]],bool);b=np.full((2,2),np.nan);w=np.array([[1.,0.]])
    old,path=base.simulate(d,p,e,b,w,['parent','hedge'])
    new,newpath=hedge.simulate(d,p,e,b,w,['parent','hedge'],hedge_targets=-np.ones((2,1)),hedge_events=np.ones((2,1),bool))
    assert np.array_equal(path,newpath)
    for key in old[0]:assert new[0][key]==old[0][key]


def test_hedge_cannot_open_short_without_parent_long():
    d=data();p=np.zeros((2,2));e=np.ones((2,2),bool);b=np.full((2,2),np.nan)
    rows,path=hedge.simulate(d,p,e,b,np.ones((1,2)),['parent','hedge'],hedge_targets=-np.ones((2,1)),hedge_events=e[:,:1])
    assert rows[0]['hedge_entries']==0 and np.all(path==1)


def test_oversized_hedge_offsets_only_parent_long():
    d=data();p=np.array([[1.,0.],[1.,0.]]);e=np.ones((2,2),bool);b=np.full((2,2),np.nan)
    rows,path=hedge.simulate(d,p,e,b,np.array([[1.,4.]]),['parent','hedge'],hedge_targets=-np.ones((2,1)),hedge_events=e[:,:1])
    assert rows[0]['hedge_entries']==1 and np.all(path==1)


def test_hedge_closes_when_parent_barrier_exits():
    d=data();p=np.array([[1.,0.],[0.,0.]]);e=np.array([[1,1],[0,0]],bool);b=np.array([[105.,np.nan],[np.nan,np.nan]])
    rows,path=hedge.simulate(d,p,e,b,np.array([[1.,1.]]),['parent','hedge'],hedge_targets=-np.ones((2,1)),hedge_events=np.array([[1],[0]],bool))
    assert np.all(path==1)


def test_parent_exit_cannot_create_phantom_unhedged_ruin():
    d=data();d['high'][0]=1000.;d['low'][0]=1.
    p=np.array([[1.,0.],[0.,0.]]);e=np.array([[1,1],[0,0]],bool);b=np.array([[105.,np.nan],[np.nan,np.nan]])
    rows,path=hedge.simulate(d,p,e,b,np.array([[1.,1.]]),['parent','hedge'],hedge_targets=-np.ones((2,1)),hedge_events=np.array([[1],[0]],bool))
    assert np.all(path==1)
    assert rows[0]['mdd_pct']==0 and not rows[0]['insolvent']
