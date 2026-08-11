import hashlib
import json
import math

import numpy as np
import pandas as pd

from training import build_high_volatility_last_zero_crossing_trend_relay_support as s


def test_last_passage_uses_terminal_side_and_last_opposed_close(monkeypatch):
    monkeypatch.setitem(s.P, "block_minutes", 6)
    terminal, fraction, occupation = s.last_passage(
        100.0, np.array([99.0, 100.0, 101.0, 102.0, 103.0, 104.0])
    )
    assert terminal == math.log(1.04)
    assert fraction == 2 / 6
    assert occupation == 4 / 6

    terminal, fraction, occupation = s.last_passage(
        100.0, np.array([101.0, 100.0, 99.0, 98.0, 97.0, 96.0])
    )
    assert terminal == math.log(0.96)
    assert fraction == 2 / 6
    assert occupation == 4 / 6


def test_last_passage_rejects_invalid_or_zero_terminal(monkeypatch):
    monkeypatch.setitem(s.P, "block_minutes", 3)
    assert all(math.isnan(value) for value in s.last_passage(100.0, np.array([101.0, 102.0])))
    assert all(math.isnan(value) for value in s.last_passage(100.0, np.array([101.0, 102.0, 100.0])))
    assert all(math.isnan(value) for value in s.last_passage(0.0, np.array([1.0, 2.0, 3.0])))


def test_prior_rank_excludes_current_and_skips_invalid(monkeypatch):
    monkeypatch.setitem(s.P, "minimum_history_blocks", 2)
    monkeypatch.setitem(s.P, "history_blocks", 3)
    rank = s.prior_rank(
        pd.Series([1.0, 2.0, np.nan, 3.0, 4.0]),
        pd.Series([True, True, False, True, True]),
    )
    assert np.isnan(rank.iloc[1])
    assert rank.iloc[3] == 1.0
    assert rank.iloc[4] == 1.0


def test_frozen_controls():
    panel = pd.DataFrame(
        {
            "source_valid": [True] * 5,
            "last_passage_tail": [False, True, True, False, True],
            "variation_tail": [True, True, False, True, True],
            "occupation_share": [0.8, 0.9, 0.8, 0.6, 0.9],
            "eligible": [False, True, False, False, True],
            "entry_side": [1, -1, 1, -1, -1],
        }
    )
    active, side, _ = s.active(panel)
    assert active.tolist() == [False, True, False, False, True]
    assert side[active].tolist() == [-1, -1]

    active, side, _ = s.active(panel, "no_last_passage_gate")
    assert active.tolist() == [True, True, False, True, True]
    assert side[active].tolist() == [1, -1, -1, -1]

    active, _, _ = s.active(panel, "no_variation_gate")
    assert active.tolist() == [False, True, True, False, True]

    active, side, _ = s.active(panel, "occupation_share")
    assert active.tolist() == [True, True, False, False, True]
    assert side[active].tolist() == [1, -1, -1]

    active, side, _ = s.active(panel, "one_block_stale_state")
    assert active.tolist() == [False, False, True, False, False]
    assert side[active].tolist() == [-1]

    active, side, _ = s.active(panel, "direction_flip")
    assert side[active].tolist() == [1, 1]

    active, side, _ = s.active(panel, "forced_long")
    assert side[active].eq(1).all()


def test_source_blind_and_hash_bound():
    assert "bars_binance " in s.PERP_QUERY
    assert "funding" not in s.PERP_QUERY.lower()
    assert "gross9" not in s.PERP_QUERY.lower()
    assert s.PREREG_SHA == hashlib.sha256(s.prereg.DEFAULT_OUTPUT.read_bytes()).hexdigest()
    value = {"한글": "last-passage"}
    expected = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert s.canonical_hash(value) == expected
