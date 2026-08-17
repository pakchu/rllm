import pandas as pd
import pytest
from training import build_high_volatility_variance_acceleration_oi_regime_router_support as s

def branch(decision: str, side: int, oi: float) -> pd.DataFrame:
    d = pd.Timestamp(decision); e = d + pd.Timedelta("5m")
    return pd.DataFrame({"split":["train"],"decision_time":[d],"feature_available_time":[d],"entry_time":[e],"exit_time":[e+pd.Timedelta("8h")],"side":[side],"oi_change":[oi]})

def test_disjoint_branches_are_unioned_without_side_changes():
    out=s.combine({"expansion_continuation":branch("2023-07-01T00:00Z",1,.1),"contraction_reversal":branch("2023-07-01T08:00Z",-1,-.1)})
    assert out["side"].tolist()==[1,-1]
    assert out["branch"].tolist()==["expansion_continuation","contraction_reversal"]

def test_duplicate_decision_is_hard_failure():
    with pytest.raises(RuntimeError,match="duplicate"):
        s.combine({"expansion_continuation":branch("2023-07-01T00:00Z",1,.1),"contraction_reversal":branch("2023-07-01T00:00Z",-1,-.1)})

def test_real_run_is_deterministic(tmp_path):
    clock=tmp_path/"clock.csv.gz";result=tmp_path/"result.json";first=s.run(clock,result);raw=(clock.read_bytes(),result.read_bytes());second=s.run(clock,result)
    assert raw==(clock.read_bytes(),result.read_bytes()) and first==second
    assert first["combined_postentry_returns_or_pnl_opened"] is False
    assert first["support_passed"]==all(first["support_checks"].values())
