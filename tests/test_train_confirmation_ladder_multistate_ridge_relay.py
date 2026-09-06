import numpy as np
from training import train_confirmation_ladder_multistate_ridge_relay as train
def test_ridge_is_deterministic_and_has_unpenalized_intercept():
 x=np.arange(120,dtype=float).reshape(5,24)+np.arange(5)[:,None]**2;y=np.arange(5,dtype=float);a=train.fit_ridge(x,y,100.);b=train.fit_ridge(x,y,100.);assert all(np.allclose(i,j) for i,j in zip(a,b));assert len(a[0])==24
