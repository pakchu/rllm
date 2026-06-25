import json
from pathlib import Path

import numpy as np

from training.event_candidate_path_gate_walkforward import (
    EventCandidatePathGateWalkForwardCfg,
    _count_signals,
    _selected_feature_names,
    _target,
)


def test_path_targets_penalize_adverse_excursion():
    row = {"reward": {"net_return_pct": 0.03, "mae": 0.01, "mfe": 0.04}}
    assert _target(row, "ret") == 0.03
    assert _target(row, "ret_minus_mae") == 0.019999999999999997
    assert _target(row, "ret_minus_2mae") == 0.009999999999999998
    assert _target(row, "win_stop1") == 1.0

    risky = {"reward": {"net_return_pct": 0.03, "mae": 0.03, "mfe": 0.04}}
    assert _target(risky, "win_stop1") == -1.0
    assert _target(risky, "win_stop2") == -1.0


def test_selected_feature_names_keeps_only_requested_prefixes():
    rows = [
        {
            "feature_snapshot": {
                "pa_ext_36_range_pos": 0.1,
                "rex_36_cur_to_max_pct": 0.2,
                "bb_z": 0.3,
            },
            "state_tokens": {"regime": "x"},
        }
    ]
    nums, cats = _selected_feature_names(rows, "pa_ext_,rex_")
    assert nums == ["pa_ext_36_range_pos", "rex_36_cur_to_max_pct"]
    assert cats == ["tok:regime=x"]


def test_count_signals_deduplicates_candidate_rows():
    rows = [{"signal_pos": 1}, {"signal_pos": 1}, {"signal_pos": 2}]
    assert _count_signals(rows) == 2
