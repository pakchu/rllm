import pandas as pd
import numpy as np

from training.event_candidate_pool_probe import EventPoolConfig, _candidate_rows_for_family


def _market(n: int = 20) -> pd.DataFrame:
    dates = pd.date_range('2024-01-01', periods=n, freq='5min')
    return pd.DataFrame({
        'date': dates,
        'open': np.linspace(100.0, 101.0, n),
        'high': np.linspace(100.5, 101.5, n),
        'low': np.linspace(99.5, 100.5, n),
        'close': np.linspace(100.0, 101.0, n),
    })


def test_candidate_rows_require_positive_above_threshold_strength():
    market = _market()
    cfg = EventPoolConfig(input_csv='dummy.csv', output='dummy.json', window_size=1, hold_bars=1, entry_delay_bars=1, stride_bars=1)
    mask = np.ones(len(market), dtype=bool)
    strength = np.zeros(len(market), dtype=float)
    strength[5] = 0.1
    strength[6] = 0.2
    direction = np.ones(len(market), dtype=float)

    rows = _candidate_rows_for_family(market, strength, direction, family='demo', threshold=0.1, mask=mask, cfg=cfg)

    assert [row['signal_date'] for row in rows] == [str(market.iloc[6]['date'])]


def test_candidate_rows_do_not_trade_zero_strength_when_threshold_collapses_to_zero():
    market = _market()
    cfg = EventPoolConfig(input_csv='dummy.csv', output='dummy.json', window_size=1, hold_bars=1, entry_delay_bars=1, stride_bars=1)
    mask = np.ones(len(market), dtype=bool)
    strength = np.zeros(len(market), dtype=float)
    strength[4] = 0.01
    direction = np.ones(len(market), dtype=float)

    rows = _candidate_rows_for_family(market, strength, direction, family='demo', threshold=0.0, mask=mask, cfg=cfg)

    assert len(rows) == 1
    assert rows[0]['signal_date'] == str(market.iloc[4]['date'])
