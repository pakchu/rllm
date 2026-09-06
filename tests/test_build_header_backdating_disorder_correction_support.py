import pandas as pd

from training import build_header_backdating_disorder_correction_support as support


def test_ceil_5m_is_conservative():
    assert support.ceil_5m(301) == pd.Timestamp(600, unit="s", tz="UTC")
    assert support.ceil_5m(600) == pd.Timestamp(600, unit="s", tz="UTC")


def test_support_contract_is_frozen():
    assert support.MINIMUM == {"train": 8, "test": 12, "eval": 12, "final": 8}
    assert "tx_count" not in support.BAR_QUERY
    assert "BTCUSDT" in support.BAR_QUERY
    assert support.COLUMNS[-1] == "side_return"


def test_block_validation_rejects_chain_gap():
    frame = pd.DataFrame([
        {"height": 1, "id": "a", "previousblockhash": "z", "timestamp": 1, "mediantime": 1, "tx_count": 1, "size": 1, "weight": 1},
        {"height": 3, "id": "b", "previousblockhash": "a", "timestamp": 2, "mediantime": 2, "tx_count": 1, "size": 1, "weight": 1},
    ])
    try:
        support.validate_blocks(frame)
    except RuntimeError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("gap was accepted")
