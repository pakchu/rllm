import numpy as np
import pandas as pd

from training import build_high_volatility_shanghai_forecast_relay_support as b


def test_var_forecast_is_strictly_causal_and_uses_sse_lag():
    count = 270
    sse = np.sin(np.arange(count) / 7.0) / 100
    btc = np.zeros(count)
    for index in range(1, count):
        btc[index] = 0.2 * btc[index - 1] + 0.8 * sse[index - 1]
    frame = pd.DataFrame({"btc_return": btc, "sse_return": sse})
    forecasts = b.causal_var_forecasts(frame, trailing=252)
    assert forecasts.iloc[:253].isna().all()
    assert np.isfinite(forecasts.iloc[253:]).all()
    changed = frame.copy()
    changed.loc[269, "btc_return"] = 99.0
    assert b.causal_var_forecasts(changed, trailing=252).iloc[268] == forecasts.iloc[268]


def test_strict_prior_midrank_excludes_current():
    values = pd.Series([1.0, 2.0, 3.0])
    ranks = b.strict_prior_midrank(values, lookback=2, minimum=2)
    assert ranks.iloc[:2].isna().all()
    assert ranks.iloc[2] == 1.0
