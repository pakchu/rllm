from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import training.audit_gross9_oi_pullback_marginal as module


def _preregistration() -> dict:
    return json.loads(module.PREREGISTRATION.read_text(encoding="utf-8"))


def _metric(
    *,
    absolute: float,
    cagr: float,
    mdd: float,
    ratio: float,
) -> dict:
    return {
        "absolute_return_pct": absolute,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": ratio,
        "trades": 100,
    }


def test_preregistration_freezes_four_weights_and_future_veto() -> None:
    payload = module.load_preregistration(module.PREREGISTRATION)
    assert payload["selection_contract"]["candidate_weight_grid"] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert payload["future_veto_contract"]["future_can_rerank"] is False
    assert payload["future_veto_contract"]["future_can_repair"] is False


def test_preregistration_hash_fails_closed_on_drift(tmp_path: Path) -> None:
    payload = _preregistration()
    payload["selection_contract"]["gross_cap"] = 9.75
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="preregistration hash drifted"):
        module.load_preregistration(drifted)


def test_gate_mask_uses_every_frozen_condition() -> None:
    features = pd.DataFrame(
        {
            "a": [0.0, 2.0, 2.0],
            "b": [0.0, 0.0, -2.0],
        }
    )
    active = module._gate_mask(
        features,
        [
            {"feature": "a", "op": ">=", "threshold": 1.0},
            {"feature": "b", "op": "<=", "threshold": -1.0},
        ],
    )
    assert active.tolist() == [False, False, True]


def test_candidate_feature_contract_fails_closed_on_missing_oi() -> None:
    with pytest.raises(RuntimeError, match="oi_minus_px_4h_z"):
        module.validate_candidate_feature_contract(
            pd.DataFrame({"range_vol": [0.1]}),
            [
                {
                    "feature": "oi_minus_px_4h_z",
                    "op": ">=",
                    "threshold": 0.5,
                }
            ],
        )


def test_candidate_feature_frame_builds_exact_oi_divergence() -> None:
    rows = 400
    close = np.exp(np.linspace(np.log(100.0), np.log(120.0), rows))
    market = pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-01-01", periods=rows, freq="5min"
            ),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.linspace(1.0, 2.0, rows),
            "quote_asset_volume": np.linspace(100.0, 240.0, rows),
            "number_of_trades": np.arange(rows) + 1,
            "open_interest": 1_000.0 + np.arange(rows) * 2.0,
        }
    )
    features = module.build_candidate_feature_frame(market)
    assert "oi_minus_px_4h_z" in features
    assert features["oi_minus_px_4h_z"].notna().sum() > 0
    module.validate_candidate_feature_contract(
        features,
        [
            {
                "feature": "oi_minus_px_4h_z",
                "op": ">=",
                "threshold": 0.5,
            }
        ],
    )


def test_selection_row_requires_both_windows_to_beat_control() -> None:
    baseline = {
        "train": _metric(absolute=100.0, cagr=40.0, mdd=10.0, ratio=4.0),
        "test2024": _metric(
            absolute=100.0, cagr=40.0, mdd=10.0, ratio=4.0
        ),
    }
    candidate = {
        "train": _metric(absolute=102.0, cagr=41.0, mdd=9.5, ratio=4.3),
        "test2024": _metric(
            absolute=103.0, cagr=42.0, mdd=9.5, ratio=4.4
        ),
    }
    control = {
        "train": _metric(
            absolute=101.0, cagr=40.5, mdd=10.0, ratio=4.05
        ),
        "test2024": _metric(
            absolute=101.0, cagr=40.5, mdd=10.0, ratio=4.05
        ),
    }
    standalone = {
        "train": _metric(absolute=1.0, cagr=1.0, mdd=1.0, ratio=1.0),
        "test2024": _metric(
            absolute=1.0, cagr=1.0, mdd=1.0, ratio=1.0
        ),
    }
    row = module.selection_row(
        weight=0.25,
        baseline=baseline,
        candidate=candidate,
        control=control,
        standalone=standalone,
        max_entry_jaccard=0.1,
    )
    assert row["passes"] is True

    failed_control = {
        **control,
        "test2024": _metric(
            absolute=101.0, cagr=40.5, mdd=10.0, ratio=4.5
        ),
    }
    failed = module.selection_row(
        weight=0.25,
        baseline=baseline,
        candidate=candidate,
        control=failed_control,
        standalone=standalone,
        max_entry_jaccard=0.1,
    )
    assert failed["passes"] is False


def test_entry_jaccard_and_rising_edges_are_exact() -> None:
    left = np.asarray([False, True, True, False, True, False])
    right = np.asarray([False, True, False, False, True, True])
    left_entries = module._entries(left)
    right_entries = module._entries(right)
    assert left_entries.tolist() == [1, 4]
    assert right_entries.tolist() == [1, 4]
    assert module._jaccard(left_entries, right_entries) == 1.0


def test_daily_returns_include_zero_filled_calendar_rows() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="12h")
    values = np.asarray([0.1, -0.05, 0.0, 0.0])
    daily = module._daily_returns(values, dates)
    assert daily.iloc[0] == pytest.approx(1.1 * 0.95 - 1.0)
    assert daily.iloc[1] == 0.0


def test_marginal_diagnostics_uses_exact_entries_and_serializes_nan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module.portfolio,
        "SLEEVES",
        (*module.portfolio.SLEEVES, module.CANDIDATE),
    )
    monkeypatch.setitem(
        module.portfolio.FAMILIES, module.CANDIDATE, "oi_divergence"
    )
    sleeves = module.portfolio.SLEEVES
    candidate_index = sleeves.index(module.CANDIDATE)
    baseline_sleeve = "fresh_kimchi_fx"
    baseline_index = sleeves.index(baseline_sleeve)
    shape = (len(sleeves), 4)
    entries = {
        sleeve: np.asarray([], dtype=np.int64) for sleeve in sleeves
    }
    entries[module.CANDIDATE] = np.asarray([1], dtype=np.int64)
    entries[baseline_sleeve] = np.asarray([1], dtype=np.int64)
    data = {
        "R": np.zeros(shape),
        "A": np.zeros(shape),
        "U": np.zeros(shape),
        "L": np.zeros(shape),
        "H": np.zeros(shape),
        "counts": np.zeros(len(sleeves), dtype=np.int64),
        "wins": np.zeros(len(sleeves), dtype=np.int64),
        "dates": pd.date_range("2024-01-01", periods=4, freq="12h"),
        "entry_positions": entries,
    }
    data["counts"][candidate_index] = 1
    data["counts"][baseline_index] = 1
    data["L"][candidate_index, 1] = -0.01
    data["H"][baseline_index, 1] = 0.01
    arrays = {split: data for split in module.SELECTION_SPLITS}
    diagnostics = module.marginal_diagnostics(
        arrays, {baseline_sleeve: 1.0}
    )
    assert diagnostics["max_entry_jaccard"] == 1.0
    assert (
        diagnostics["train"]["daily_mtm_correlation"]["pearson"] is None
    )
    output = tmp_path / "finite.json"
    module._atomic_json(output, {"diagnostics": diagnostics})
    assert json.loads(output.read_text())["diagnostics"]["max_entry_jaccard"] == 1.0


def test_authoritative_gross9_contract_matches_preregistration() -> None:
    preregistration = module.load_preregistration(module.PREREGISTRATION)
    result = module.validate_authoritative_gross9(
        module.Config(), preregistration
    )
    assert result["source_future_used_for_allocation_ranking"] is False
    assert sum(result["frozen_top1_weights"].values()) == 9.0


def test_source_does_not_offer_future_phase_or_threshold_search() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'choices=("pre2025",)' in source
    assert "threshold_grid" not in source
    assert "eval2025" not in module.SELECTION_SPLITS
    assert "ytd2026" not in module.SELECTION_SPLITS
