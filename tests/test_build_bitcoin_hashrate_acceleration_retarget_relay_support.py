import numpy as np,pandas as pd
from training import build_bitcoin_hashrate_acceleration_retarget_relay_support as support
def test_compact_target_known_diff1():assert support.target(0x1d00ffff)==0x00ffff << (8*(0x1d-3))
def test_header_bits_decode_position():
 header=bytearray(80);header[72:76]=(0x170fffff).to_bytes(4,"little");assert int.from_bytes(header[72:76],"little")==0x170fffff
def test_rank_excludes_current():
 r=support.rank(pd.Series(np.arange(27,dtype=float)));assert np.isnan(r.iloc[25]);assert r.iloc[26]==1.
