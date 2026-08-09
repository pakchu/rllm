import numpy as np,pandas as pd
from training import build_high_volatility_equity_adjusted_btc_residual_reversal_support as support
def test_beta_is_prior_linear_projection():
 assert np.isnan(support.beta(pd.DataFrame()))
 x=np.arange(1,61,dtype=float);p=pd.DataFrame({'spy_return':x,'btc_return':2*x+3});assert np.isclose(support.beta(p),2.)
def test_primary_fades_residual_and_controls_are_fixed():
 s=pd.DataFrame([{'session_date':pd.Timestamp('2023-07-03'),'cash_close_time':pd.Timestamp('2023-07-03T20:00:00Z'),'spy_return':.01,'btc_return':.03,'beta':1.,'stale_beta':2.,'residual':.02,'stale_residual':.01,'residual_rank':.8,'variation':.1,'variation_rank':.8,'elapsed_gap_hours':72.}]);assert support.build_clock(s).side.tolist()==[-1];assert support.build_clock(s,'direction_flip').side.tolist()==[1];assert support.build_clock(s,'raw_btc_return_reversal').side.tolist()==[-1];assert support.build_clock(s,'same_clock_forced_long').side.tolist()==[1]
