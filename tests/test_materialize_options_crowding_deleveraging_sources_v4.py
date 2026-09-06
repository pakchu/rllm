import hashlib
from pathlib import Path

import pandas as pd

from training import materialize_options_crowding_deleveraging_sources_v4 as m


def test_zero_db_mark_uses_exact_event_aligned_frozen_mark(tmp_path: Path, monkeypatch) -> None:
    source=tmp_path/'marks.csv.gz'
    pd.DataFrame({'funding_time_utc':['2023-07-01T00:00:00.001Z'],'funding_rate':[0.001],'settlement_mark_price':[100.0]}).to_csv(source,index=False,compression='gzip')
    monkeypatch.setattr(m,'FROZEN_MARKS',source);monkeypatch.setattr(m,'FROZEN_MARKS_SHA',hashlib.sha256(source.read_bytes()).hexdigest())
    frame=pd.DataFrame({'symbol':['BTCUSDT'],'funding_time':pd.to_datetime(['2023-07-01T00:00:00.001Z']),'funding_rate':[0.001],'mark_price':[0.0]})
    out=m.complete_funding_marks(frame)
    assert out.loc[0,'mark_price']==100.0
    assert out.loc[0,'mark_source']=='frozen_official_binance_mark'
