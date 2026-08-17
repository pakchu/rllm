import pandas as pd
from training import build_high_volatility_btc_factor_residual_flip_continuation_support as s

def sample() -> pd.DataFrame:
    rows=[]
    for i,(btc,alts) in enumerate([(0.06,[.01,.011,.009,.012,.008,.01]),(-.02,[.01,.011,.009,.012,.008,.01])]):
        row={"decision_time":pd.Timestamp("2023-07-01T04:00Z")+pd.Timedelta(hours=8*i),"btc_return":btc,"btc_realized_variation":.1,"variation_rank":.8}
        row.update(dict(zip(s.prereg.ALTS,alts)));rows.append(row)
    return pd.DataFrame(rows)

def test_residual_flip_follows_new_relative_side():
    features=s.prepare(sample());assert features.iloc[1]["eligible"]
    clock=s.build_clock(features);assert len(clock)==1 and clock.iloc[0]["side"]==-1

def test_same_sign_residual_is_not_eligible():
    frame=sample();frame.loc[1,"btc_return"]=.04;assert not s.prepare(frame).iloc[1]["eligible"]

def test_real_run_is_deterministic(tmp_path):
    f=tmp_path/"f.csv.gz";c=tmp_path/"c.csv.gz";r=tmp_path/"r.json";first=s.run(f,c,r);raw=(f.read_bytes(),c.read_bytes(),r.read_bytes());second=s.run(f,c,r)
    assert raw==(f.read_bytes(),c.read_bytes(),r.read_bytes()) and first==second
    assert first["postentry_return_pnl_execution_price_opened"] is False
