from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import training.audit_gross9_residual_recovery_extratrees_marginal as mod
import training.portfolio_opt_added_alpha_update as portfolio
from training.portfolio_opt_new_alpha_pool import _event_path


def _market(rows: int = 240) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    opens = 100.0 * np.exp(0.0008 * x + 0.006 * np.sin(x / 8.0))
    highs = opens * (1.0 + 0.003 + 0.001 * np.cos(x / 5.0))
    lows = opens * (1.0 - 0.003 - 0.001 * np.sin(x / 7.0) ** 2)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": opens,
        }
    )


def test_preregistration_is_hash_bound_and_total() -> None:
    payload = mod.load_preregistration(mod.PREREGISTRATION)
    assert payload["physical_selection_cutoff"] == "2025-01-01"
    assert tuple(payload["feature_contract"]["columns"]) == mod.FEATURE_COLUMNS
    assert payload["candidate_universe"]["portfolio_cells"] == 24
    assert set(payload["selection_contract"]["gross9_windows"]) == {
        "train",
        "selection_2024",
    }
    assert payload["future_veto_contract"]["future_can_rerank"] is False
    assert payload["future_veto_contract"]["future_can_repair"] is False


def test_exact_no_stop_targets_match_canonical_event_path() -> None:
    market = _market()
    anchors = np.asarray([15, 37, 91], dtype=np.int64)
    hold = 24
    targets = mod.exact_no_stop_targets(
        market,
        anchors,
        hold=hold,
        cost_rate=0.0006,
        leverage=0.5,
    )
    for row, anchor in enumerate(anchors):
        for side, return_col, adverse_col in (
            ("long", 0, 1),
            ("short", 2, 3),
        ):
            path = _event_path(
                market,
                int(anchor),
                side=side,
                hold=hold,
                cost_rate=0.0006,
                leverage=0.5,
            )
            assert path is not None
            _, adverse, realized = path
            assert np.isclose(targets[row, return_col], realized, atol=1e-12)
            assert np.isclose(
                targets[row, adverse_col],
                max(0.0, -float(np.min(adverse))),
                atol=1e-12,
            )


def test_adjusted_scores_use_population_seed_uncertainty_and_long_tie() -> None:
    predictions = np.asarray(
        [
            [[0.04, 0.02, 0.03, 0.00], [0.01, 0.00, 0.01, 0.00]],
            [[0.02, 0.02, 0.03, 0.02], [0.01, 0.00, 0.01, 0.00]],
            [[0.03, 0.02, 0.03, 0.01], [0.01, 0.00, 0.01, 0.00]],
        ],
        dtype=float,
    )
    score, side = mod.adjusted_scores(predictions)
    long_utility = predictions[:, :, 0] - 0.5 * predictions[:, :, 1]
    short_utility = predictions[:, :, 2] - 0.5 * predictions[:, :, 3]
    long_score = long_utility.mean(axis=0) - 0.5 * long_utility.std(
        axis=0, ddof=0
    )
    short_score = short_utility.mean(axis=0) - 0.5 * short_utility.std(
        axis=0, ddof=0
    )
    expected_side = np.where(long_score >= short_score, 1, -1)
    assert np.allclose(score, np.maximum(long_score, short_score))
    assert np.array_equal(side, expected_side)
    assert side[1] == 1


def test_annual_masks_purge_fit_and_prediction_boundary_crossers() -> None:
    signal_dates = pd.Series(
        pd.to_datetime(
            [
                "2021-12-30",
                "2021-12-31 23:55",
                "2022-01-01",
                "2022-12-31 23:55",
            ],
            format="mixed",
        )
    )
    exit_dates = pd.Series(
        pd.to_datetime(
            [
                "2021-12-31",
                "2022-01-01",
                "2022-01-01 06:00",
                "2023-01-01",
            ],
            format="mixed",
        )
    )
    fit, predict = mod.annual_masks(
        signal_dates,
        exit_dates,
        np.ones(4, dtype=bool),
        np.ones(4, dtype=bool),
        prediction_year=2022,
    )
    assert np.array_equal(fit, [True, False, False, False])
    assert np.array_equal(predict, [False, False, True, False])


def test_event_occupancy_bridges_zero_path_bars_until_exit() -> None:
    ret = np.zeros(8)
    ret[1] = -0.0003
    ret[5] = -0.0003
    event = {
        "entry_positions": [1],
        "ret": ret,
        "adv": np.zeros(8),
        "low": np.zeros(8),
        "high": np.zeros(8),
    }
    occupied = mod.event_occupancy(event, 8)
    assert np.array_equal(
        occupied,
        [False, True, True, True, True, False, False, False],
    )
    same_bar = np.zeros(8)
    same_bar[3] = -0.001
    assert not mod.event_occupancy(
        {"entry_positions": [3], "ret": same_bar}, 8
    ).any()


def test_gross9_drawdown_state_does_not_reset_at_2024(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.to_datetime(
        [
            "2023-12-31 23:45",
            "2023-12-31 23:50",
            "2023-12-31 23:55",
            "2024-01-01 00:00",
            "2024-01-01 00:05",
        ]
    )
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.ones(5),
            "high": np.ones(5),
            "low": np.ones(5),
            "close": np.ones(5),
        }
    )
    masks = {
        "train": np.asarray([True, True, True, False, False]),
        "test2024": np.asarray([False, False, False, True, True]),
    }
    arrays = {
        "train": {
            "R": np.asarray([[0.10, -0.10, 0.0]]),
            "L": np.zeros((1, 3)),
            "H": np.zeros((1, 3)),
        },
        "test2024": {
            "R": np.asarray([[-0.10, 0.0]]),
            "L": np.zeros((1, 2)),
            "H": np.zeros((1, 2)),
        },
    }
    monkeypatch.setattr(portfolio, "SLEEVES", ("base",))
    monkeypatch.setattr(
        portfolio,
        "split_arrays",
        lambda events, supplied_market, supplied_masks: arrays,
    )
    flat, drawdown, meta = mod.gross9_state(
        market, masks, [], {"base": 1.0}
    )
    assert flat.all()
    assert np.isclose(drawdown[3], 0.19)
    assert meta["continuous_clock"]["reset_at_2024"] is False


def test_final_contract_valid_anchor_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market(220)
    hold = 24
    final_valid = len(market) - hold - 2
    active = np.zeros(len(market), dtype=bool)
    active[final_valid] = True
    masks = {"train": np.ones(len(market), dtype=bool)}
    monkeypatch.setattr(portfolio, "SLEEVES", ("candidate",))
    events: list[dict] = []
    counts = mod.append_rrem_mask_policy(
        events,
        market,
        masks,
        name="candidate",
        long_active=active,
        short_active=np.zeros(len(market), dtype=bool),
        hold=hold,
        stride=1,
        cost_rate=0.0006,
    )
    assert counts == {"train": 1}
    assert events[0]["entry_positions"] == [final_valid + 1]


def test_same_gross_formula_and_total_order() -> None:
    combined, comparator = mod.same_gross_weights(
        {"a": 3.0, "b": 6.0}, "candidate", 0.75
    )
    assert combined == {"a": 3.0, "b": 6.0, "candidate": 0.75}
    assert comparator == {"a": 3.25, "b": 6.5}
    rows = [
        {"candidate_name": "z", "selection_key": [1.0, 2.0]},
        {"candidate_name": "a", "selection_key": [1.0, 2.0]},
        {"candidate_name": "m", "selection_key": [1.0, 3.0]},
    ]
    assert [row["candidate_name"] for row in mod.rank_rows(rows)] == [
        "m",
        "a",
        "z",
    ]


def test_cost_stress_preserves_candidate_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market(260)
    active = np.zeros(len(market), dtype=bool)
    active[[155, 191, 227]] = True
    candidates = {
        "candidate": {
            "hold": 24,
            "long_active": active,
            "short_active": np.zeros(len(market), dtype=bool),
        }
    }
    masks = {"train": np.ones(len(market), dtype=bool)}
    monkeypatch.setattr(portfolio, "SLEEVES", ("candidate",))
    normal: list[dict] = []
    stress: list[dict] = []
    normal_counts = mod.append_candidates(
        normal, market, masks, candidates, cost_rate=0.0006
    )
    stress_counts = mod.append_candidates(
        stress, market, masks, candidates, cost_rate=0.001
    )
    assert normal_counts == stress_counts
    assert normal[0]["entry_positions"] == stress[0]["entry_positions"]


def test_candidate_universe_and_ranking_helpers_are_deterministic() -> None:
    assert len(mod.CANDIDATE_NAMES) == 6
    assert len(set(mod.CANDIDATE_NAMES)) == 6
    assert mod.signed_geometric_mean(4.0, 9.0) == 6.0
    assert mod.signed_geometric_mean(-4.0, -9.0) == -6.0
    assert mod.signed_geometric_mean(-4.0, 9.0) == -6.0
    assert mod.signed_geometric_mean(0.0, 9.0) == 0.0


def test_runner_has_pre2025_phase_only_and_no_future_result_loader() -> None:
    source = inspect.getsource(mod.main)
    assert 'choices=("pre2025",)' in source
    run_source = inspect.getsource(mod.run_pre2025)
    assert "2025/2026" not in run_source
    assert "future_opened" in run_source
    assert "future_can_rerank" in run_source
