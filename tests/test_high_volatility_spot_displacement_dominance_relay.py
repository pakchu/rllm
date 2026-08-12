import numpy as np
import pandas as pd

from training.build_high_volatility_spot_displacement_dominance_relay_support import half_hour_returns


def test_half_hour_returns_use_six_consecutive_five_minute_equivalents() -> None:
    frame = pd.DataFrame({"open": np.ones(480), "close": np.ones(480)})
    for block in range(16):
        frame.loc[block * 30 + 29, "close"] = np.exp((block + 1) / 1000)
    values = half_hour_returns(frame)
    assert len(values) == 16
    assert np.allclose(values, np.arange(1, 17) / 1000)
