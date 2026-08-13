import numpy as np
import pandas as pd
from training import build_confirmation_ladder_witness_migration_sponsorship_relay_support as support


def _minutes(start, periods):
    times = pd.date_range(start, periods=periods, freq="1min")
    close = np.linspace(100, 110, periods)
    opened = np.r_[100.0, close[:-1]]
    return pd.DataFrame({"ts": times, "open": opened, "high": np.maximum(opened, close)+0.1, "low": np.minimum(opened, close)-0.1, "close": close, "duplicate_count": 1})


def test_ceil_and_interval_return():
    assert support.ceil_5m(301) == pd.Timestamp(600, unit="s", tz="UTC")
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    market = support.prepare_minutes(_minutes(start, 10))
    value = support.interval_return(market, int(start.timestamp()), int((start + pd.Timedelta(minutes=10)).timestamp()))
    assert value is not None and value[0] > 0 and value[1] == 10


def test_primary_and_controls():
    features = pd.DataFrame({
        "source_valid": [True, True], "eligible_state": [False, True],
        "late_return": [-0.1, -0.2], "witness_migration": [False, True],
        "serialized_size_migration": [True, True], "late_unanimous": [False, True],
    })
    active, side = support.active_and_side(features)
    assert active.tolist() == [False, True] and side.tolist() == [-1, -1]
    _, flip = support.active_and_side(features, "direction_flip")
    assert flip.tolist() == [1, 1]
    witness, _ = support.active_and_side(features, "witness_migration_only")
    size, _ = support.active_and_side(features, "serialized_size_migration")
    assert witness.tolist() == [False, True]
    assert size.tolist() == [False, True]


def test_bip141_validation():
    rows=[]
    for i in range(7):
        rows.append({"height": 36+i, "id": f"{i+1:064x}", "previousblockhash": f"{i:064x}", "timestamp": 1_700_000_000+600*i, "mediantime": 1_699_999_000+600*i, "tx_count": 100, "size": 1_000_000, "weight": 3_000_000})
    valid=support.validate_blocks(pd.DataFrame(rows))
    assert len(valid)==7
    broken=pd.DataFrame(rows);broken.loc[3,"weight"]=4_000_001
    try: support.validate_blocks(broken)
    except RuntimeError as exc: assert "BIP141" in str(exc)
    else: raise AssertionError("invalid BIP141 row accepted")


def test_outcomes_closed_and_canonical_json():
    source_text = open(support.__file__).read()
    assert '"execution_prices_opened": False' in source_text
    assert '"gross9_rows_opened": False' in source_text
    assert '"rv20_opened": False' in source_text
    assert "ensure_ascii=False" in source_text
