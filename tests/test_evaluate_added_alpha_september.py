import numpy as np
import pandas as pd
import pytest
from training import evaluate_added_alpha_september as s


def test_archive_delays_observations_one_bar():
    raw=pd.DataFrame({'create_time':['2026-09-01 00:00:00'],'sum_open_interest':[123.]})
    result=s.archive_oi(raw)
    assert result.date.iloc[0]==pd.Timestamp('2026-09-01 00:05:00')
    assert result.open_interest.iloc[0]==123.


def test_bad_archive_rejected():
    raw=pd.DataFrame({'create_time':['2026-09-01 00:01:00'],'sum_open_interest':[123.]})
    with pytest.raises(ValueError):s.archive_oi(raw)


def test_weights_not_reoptimized():
    assert s.LABELS[:3]==['june_selected_80_20','retrospective_60_40','fixed_50_50']
    assert np.allclose(s.WEIGHTS.sum(axis=1),1)
    assert s.END=='2026-09-05T00:00:00Z'
