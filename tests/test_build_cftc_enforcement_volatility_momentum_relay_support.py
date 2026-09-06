import numpy as np,pandas as pd
from training import build_cftc_enforcement_volatility_momentum_relay_support as support
def test_parser_reads_action_row():
 raw=b'''<table><tbody><tr><td><time datetime="2025-06-01T12:30:00Z">06/01/2025</time></td><td><a href="/PressRoom/PressReleases/1234">Action title</a><ul><li><a href="/media/order.pdf">Order</a></li></ul></td></tr></tbody></table>''';rows=support.parse(raw);assert rows==[{"timestamp":"2025-06-01T12:30:00Z","title":"Action title","action_url":"https://www.cftc.gov/PressRoom/PressReleases/1234","document_urls":["https://www.cftc.gov/media/order.pdf"]}]
def test_rank_excludes_current():
 r=support.rank(pd.Series(np.arange(127,dtype=float)));assert np.isnan(r.iloc[125]);assert r.iloc[126]==1.
def test_parallelism_is_bounded():assert support.WORKERS==4
