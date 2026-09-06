import json

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_usd_index_forecast_threshold_relay_support as b


def _equity(dates, adjusted):
    return pd.DataFrame({"session_date": pd.to_datetime(dates), "adjusted_close": adjusted})


def test_uup_adjusted_source_and_official_close():
    timestamps = [int(pd.Timestamp(d, tz="America/New_York").timestamp()) for d in ("2024-01-02", "2024-01-03")]
    result = {"meta": {"symbol": "UUP", "exchangeTimezoneName": "America/New_York"}, "timestamp": timestamps,
              "indicators": {"quote": [{"volume": [10,20], "close": [101,102], "open": [100,101], "low": [99,100], "high": [102,103]}], "adjclose": [{"adjclose": [91.,92.]}]}}
    stable, frame, _ = b.normalize_yahoo_chart(json.dumps({"chart":{"result":[result],"error":None}}).encode(), "UUP")
    assert frame.adjusted_close.tolist() == [91.,92.]
    assert json.loads(stable)["adjclose"]["adjclose"] == [91.,92.]
    panel = b.build_equity_panel({"UUP": _equity(["2023-07-03","2023-07-05"],[100.,101.])})
    assert panel.cash_close_time.dt.strftime("%Y-%m-%dT%H:%MZ").tolist() == ["2023-07-03T17:00Z","2023-07-05T20:00Z"]


def test_btc_grid_is_exact():
    start=pd.Timestamp("2024-01-01T00:00Z"); end=start+pd.Timedelta(minutes=3)
    raw=pd.DataFrame({"ts":pd.date_range(start,end,freq="1min",inclusive="left"),"open":[1,2,3],"close":[2,3,4]})
    assert len(b.normalize_btc_bars(raw,start,end)) == 3
    with pytest.raises(RuntimeError,match="exact requested 1m grid"):
        b.normalize_btc_bars(raw.iloc[[0,2]],start,end)


def test_var_is_strictly_trailing_and_current_only_enters_forecast():
    n=258
    frame=pd.DataFrame({"btc_return":np.sin(np.arange(n)/7)/100,"uup_return":np.cos(np.arange(n)/11)/200})
    first=b.causal_var_forecasts(frame,trailing=252)
    changed=frame.copy(); changed.loc[n-1,"uup_return"] += 10
    second=b.causal_var_forecasts(changed,trailing=252)
    assert first.iloc[:253].isna().all()
    assert first.iloc[n-2] == second.iloc[n-2]
    assert first.iloc[n-1] != second.iloc[n-1]


def _signal_frame():
    return pd.DataFrame({"btc_forecast":[.2,-.3,-.1,.4],"forecast_magnitude_rank":[.7,.7,.5,.9],
                         "uup_return":[.1,-.2,.3,-.4],"uup_magnitude_rank":[.7,.4,.8,.9],
                         "btc_return":[.2,.1,-.1,.3],"btc_variation_rank":[.9]*4})


def test_threshold_signal_and_controls():
    frame=_signal_frame(); active,side=b._signal(frame,"primary")
    assert active.tolist()==[True,True,False,True]; assert side.tolist()==[1,-1,-1,1]
    stale,_=b._signal(frame,"one_session_stale_forecast"); assert stale.tolist()==[False,True,True,False]
    uup,_=b._signal(frame,"uup_sign_only"); assert uup.tolist()==[True,False,True,True]
    _,flip=b._signal(frame,"direction_flip"); assert flip.tolist()==[-1,1,1,-1]
    _,forced=b._signal(frame,"same_clock_forced_long"); assert forced.tolist()==[1,1,1,1]
    assert len(b.CONTROLS)==6


def test_clock_latency_half_open_and_support_gates():
    frame=_signal_frame().iloc[1:].reset_index(drop=True)
    frame["session_date"]=pd.to_datetime(["2023-07-03","2023-07-04","2023-07-05"])
    frame["cash_close_time"]=pd.to_datetime(["2023-07-03T17:00Z","2023-07-04T17:00Z","2023-07-05T17:00Z"])
    frame["btc_realized_variation"]=[.01]*3
    clock=b.build_clock(frame)
    assert clock.entry_time.dt.strftime("%Y-%m-%dT%H:%MZ").iloc[0]=="2023-07-03T17:10Z"
    assert b.MINIMUM_EVENTS=={"train":8,"test":12,"eval":12,"final":8}
