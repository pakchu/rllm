import numpy as np,pandas as pd
from training import search_calendar_session_alpha_combinations as s

def test_calendar_model_uses_only_mature_prior_years():
 dates=pd.date_range('2020-03-01',periods=24*365*3,freq='1h');x=pd.DataFrame(index=dates);data={'open':np.linspace(100,200,len(x))}
 score,audit=s.calendar_expected(x,data,24)
 first=next(r for r in audit if r['prediction_year']==2021)
 assert pd.Timestamp(first['last_train'])+pd.Timedelta(hours=24,minutes=5)<pd.Timestamp('2021-01-01')
 assert np.isnan(score[dates.year==2020]).all()

def test_design_is_calendar_known_ex_ante():
 assert 'UTC calendar known ex ante' in s.DESIGN['sources']
 assert s.DESIGN['no_frequency_or_fee_ratio_gate']
