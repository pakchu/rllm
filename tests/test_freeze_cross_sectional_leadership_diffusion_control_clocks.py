from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training import freeze_cross_sectional_leadership_diffusion_control_clocks as freeze
from training import preregister_cross_sectional_leadership_diffusion as cld


def test_frozen_support_dependencies_verify_without_outcomes() -> None:
    support, primary = freeze._load_and_verify_support()
    assert support["all_support_gates_pass"] is True
    assert primary["post_entry_outcomes_opened"] is False
    assert primary["event_count"] == 106


def test_signal_from_mask_uses_only_selected_rows() -> None:
    panel = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2023-01-01 00:00", "2023-01-01 01:00"]
            ),
            "feature_boundary": pd.to_datetime(
                ["2023-01-01 00:05", "2023-01-01 01:05"]
            ),
            "entry_date": pd.to_datetime(
                ["2023-01-01 00:10", "2023-01-01 01:10"]
            ),
            "direction": [1.0, -1.0],
        }
    )
    signal = freeze._signal_from_mask(
        panel, pd.Series([True, False]), branch="control"
    )
    assert signal["side"].tolist() == [1, 0]
    assert signal["branch"].tolist() == ["control", "none"]


def test_control_payload_has_no_outcome_fields_after_freeze() -> None:
    path = freeze.OUTPUT
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    assert payload["post_entry_outcomes_opened"] is False
    assert payload["entry_or_later_ohlc_loaded"] is False
    assert payload["funding_loaded"] is False
    assert payload["controls_are_diagnostics_not_repair_candidates"] is True
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert cld.canonical_hash(body) == payload["manifest_hash"]
