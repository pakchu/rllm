from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import training.evaluate_cash_open_cross_asset_gap_relay_marginal as cogr


def _market(start: str = "2023-03-10 00:00", periods: int = 1200):
    dates = pd.date_range(start, periods=periods, freq="5min")
    base = 100.0 + np.arange(periods) * 0.01
    return pd.DataFrame(
        {
            "date": dates,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.1,
        }
    )


def _session_market(session_dates: list[str]) -> pd.DataFrame:
    blocks = []
    for index, session_date in enumerate(session_dates):
        signal = cogr.feature_times_utc([session_date])[0]
        dates = pd.date_range(
            signal - pd.Timedelta(minutes=5),
            periods=cogr.HOLD_BARS + 3,
            freq="5min",
        )
        base = 100.0 + index + np.arange(len(dates)) * 0.001
        blocks.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "open": base,
                    "high": base + 0.25,
                    "low": base - 0.25,
                    "close": base + 0.05,
                }
            )
        )
    return (
        pd.concat(blocks, ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


def _features(session_dates: list[str]) -> pd.DataFrame:
    rows = []
    for index, session_date in enumerate(session_dates):
        row = {
            "session_date": pd.Timestamp(session_date),
            "feature_valid": True,
        }
        for column_index, column in enumerate(cogr.FEATURE_COLUMNS):
            row[column] = float(index + column_index / 1000.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _learner_prereg(minimum_fit_rows: int = 1) -> dict:
    return {
        "learner_contract": {
            "minimum_fit_rows": minimum_fit_rows,
            "n_estimators": 4,
            "max_depth": 2,
            "min_samples_leaf": 1,
            "max_features": 0.75,
        }
    }


def _gate_prereg() -> dict:
    standalone = {
        "absolute_return_positive": True,
        "minimum_cagr_to_strict_mdd": -1_000.0,
        "maximum_strict_mdd_pct": 100.0,
        "minimum_trades": 0,
        "minimum_long_share": 0.0,
        "minimum_short_share": 0.0,
        "maximum_month_share": 1.0,
        "maximum_weekday_share": 1.0,
        "candidate_10bp_stress_absolute_return_positive": True,
    }
    portfolio = {
        "maximum_strict_mdd_pct": 100.0,
        "absolute_return_retention_floor_vs_unscaled_gross9": -1_000.0,
        "minimum_cagr_mdd_improvement_vs_same_gross_prorata_gross9": -1_000.0,
        "strict_mdd_reduction_vs_unscaled_gross9": True,
        "maximum_exact_entry_jaccard_vs_any_gross9_sleeve": 0.25,
        "candidate_10bp_stress_portfolio_absolute_return_positive": True,
    }
    return {
        "selection_contract": {
            "selection_2023h2_standalone_requirements": standalone,
            "selection_2023h2_portfolio_requirements": portfolio,
            "selection_2023h2_mechanism_requirements": {
                "primary_cagr_mdd_margin_over_best_of_all_nine_controls": -1_000.0
            },
        }
    }


def _zero_funding(market: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(market["date"].iloc[0]) - pd.Timedelta(hours=1)],
            "funding_rate": [0.0],
        }
    )


def _trade(
    signal_position: int = 9,
    *,
    side: int = 1,
    entry_date: str = "2023-07-03 13:40:00",
):
    return cogr.accounting.Trade(
        signal_position=signal_position,
        entry_position=signal_position + 1,
        exit_position=signal_position + 1 + cogr.HOLD_BARS,
        side=side,
        gross_return=0.02 * side,
        price_factor=1.01,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=1.015,
        adverse_price_factor=0.995,
        entry_date=entry_date,
    )


def _array_data(
    *,
    length: int = 300,
    baseline_return: float = 0.01,
    candidate_return: float = 0.005,
    baseline_entries: tuple[int, ...] = (10,),
    candidate_entries: tuple[int, ...] = (10,),
) -> tuple[dict, dict, dict]:
    cogr._install_candidate_sleeve()
    sleeves = cogr.portfolio.SLEEVES
    rows = len(sleeves)
    dates = pd.date_range("2023-07-01", periods=length, freq="5min")

    def empty() -> dict:
        return {
            "R": np.zeros((rows, length), dtype=float),
            "A": np.zeros((rows, length), dtype=float),
            "U": np.zeros((rows, length), dtype=float),
            "L": np.zeros((rows, length), dtype=float),
            "H": np.zeros((rows, length), dtype=float),
            "counts": np.zeros(rows, dtype=np.int64),
            "wins": np.zeros(rows, dtype=np.int64),
            "dates": dates,
            "entry_positions": {
                sleeve: np.empty(0, dtype=np.int64) for sleeve in sleeves
            },
        }

    baseline = empty()
    candidate = empty()
    baseline_index = sleeves.index("cand_rex_veto_7")
    candidate_index = sleeves.index(cogr.CANDIDATE_SLEEVE)
    baseline["R"][baseline_index, -1] = baseline_return
    baseline["L"][baseline_index, -1] = -0.002
    baseline["H"][baseline_index, -1] = 0.012
    baseline["counts"][baseline_index] = len(baseline_entries)
    baseline["wins"][baseline_index] = len(baseline_entries)
    baseline["entry_positions"]["cand_rex_veto_7"] = np.asarray(
        baseline_entries, dtype=np.int64
    )
    candidate["R"][candidate_index, -2] = candidate_return
    candidate["L"][candidate_index, -2] = -0.001
    candidate["H"][candidate_index, -2] = 0.006
    candidate["counts"][candidate_index] = len(candidate_entries)
    candidate["wins"][candidate_index] = len(candidate_entries)
    candidate["entry_positions"][cogr.CANDIDATE_SLEEVE] = np.asarray(
        candidate_entries, dtype=np.int64
    )
    combined = cogr.merge_array_data(baseline, candidate)
    return baseline, candidate, combined


def test_dst_alignment_entry_prior_bar_and_exact_exit():
    session_dates = [
        "2023-03-10",
        "2023-03-13",
        "2023-07-05",
        "2023-11-06",
    ]
    market = _session_market(session_dates)
    aligned = cogr.align_feature_sessions_to_market(
        _features(session_dates), market
    )
    assert list(aligned["feature_time_utc"].astype(str)) == [
        "2023-03-10 14:35:00",
        "2023-03-13 13:35:00",
        "2023-07-05 13:35:00",
        "2023-11-06 14:35:00",
    ]
    dates = pd.DatetimeIndex(market["date"])
    for _, row in aligned.iterrows():
        signal = int(row["signal_position"])
        entry = int(row["entry_position"])
        exit_position = int(row["exit_position"])
        assert int(row["coordination_position"]) == signal - 1
        assert entry == signal + 1
        assert exit_position == entry + 144
        assert dates[entry] - dates[signal] == pd.Timedelta(minutes=5)
        assert dates[exit_position] - dates[entry] == pd.Timedelta(hours=12)


def test_timing_alignment_fails_closed_when_exact_entry_bar_is_missing():
    market = _session_market(["2023-03-13"])
    entry_time = cogr.feature_times_utc(["2023-03-13"])[0] + pd.Timedelta(
        minutes=5
    )
    market = market.loc[market["date"] != entry_time].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="09:30/09:35/09:40"):
        cogr.align_feature_sessions_to_market(
            _features(["2023-03-13"]), market
        )


def test_latency_bar_market_mutation_does_not_change_targets():
    market = _market("2023-03-13 00:00", 500)
    aligned = cogr.align_feature_sessions_to_market(
        _features(["2023-03-13"]), market
    )
    signal = int(aligned.loc[0, "signal_position"])
    funding = _zero_funding(market)
    before = cogr.exact_targets(market, funding, np.array([signal]))
    mutated = market.copy()
    mutated.loc[signal, ["open", "high", "low", "close"]] = [
        1.0,
        9999.0,
        0.5,
        5000.0,
    ]
    after = cogr.exact_targets(mutated, funding, np.array([signal]))
    np.testing.assert_allclose(after, before, rtol=0, atol=0)


def test_gate_uses_prior_bar_not_signal_latency_bar():
    aligned = pd.DataFrame(
        {
            "session_date": [pd.Timestamp("2023-07-03")],
            "signal_position": [10],
            "coordination_position": [9],
            "entry_position": [11],
            "exit_position": [155],
        }
    )
    fold = {
        "rows": np.array([0]),
        "score": np.array([1.0]),
        "side": np.array([1]),
        "threshold": 0.0,
    }
    flat = np.zeros(200, dtype=bool)
    flat[9] = True
    flat[10] = False
    drawdown = np.zeros(200)
    accepted = cogr.accepted_schedule(
        aligned, fold, "gross9_flat_at_signal", flat, drawdown
    )
    assert len(accepted) == 1
    flat[10] = True
    assert (
        cogr.accepted_schedule(
            aligned, fold, "gross9_flat_at_signal", flat, drawdown
        )
        == accepted
    )


def test_fit_and_eval_purge_require_full_path_containment():
    market = _market("2023-12-31 00:00", 1000)
    aligned = cogr.align_feature_sessions_to_market(
        _features(["2023-12-31", "2024-01-01", "2024-01-02"]), market
    )
    dates = pd.DatetimeIndex(pd.to_datetime(market["date"]))
    fit = cogr.fit_mask(aligned, dates, "2024-01-01")
    assert not fit[0]
    prediction = cogr.prediction_mask(
        aligned, dates, "2024-01-02", "2024-01-02 18:00"
    )
    assert not prediction[2]


def test_selection_fold_predictions_never_call_eval_fold_or_target_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    sessions = [
        "2021-01-04",
        "2022-01-03",
        "2023-01-03",
        "2023-03-01",
        "2023-07-03",
        "2023-09-01",
    ]
    market = _session_market(sessions)
    features = _features([*sessions, "2024-02-01"])
    aligned = cogr.align_feature_sessions_to_market(features, market)
    called_signals: list[np.ndarray] = []

    def fake_targets(_market, _funding, signals, **_kwargs):
        called_signals.append(np.asarray(signals, dtype=np.int64).copy())
        return np.zeros((len(signals), 4), dtype=np.float64)

    monkeypatch.setattr(cogr, "exact_targets", fake_targets)
    predictions, metadata = cogr.fold_predictions(
        aligned,
        market,
        _zero_funding(market),
        _learner_prereg(),
        phase="selection",
    )
    assert tuple(predictions) == cogr.PHASE_FOLDS["selection"]
    assert tuple(metadata["allowed_folds"]) == cogr.PHASE_FOLDS["selection"]
    assert len(called_signals) == 1
    target_dates = pd.DatetimeIndex(market["date"])[called_signals[0]]
    assert target_dates.max() < pd.Timestamp("2024-01-01")
    assert "eval_2024" not in predictions


def test_eval_fold_graph_includes_both_prior_threshold_sources():
    assert cogr.PHASE_FOLDS["eval"] == (
        "calibration_2023h1",
        "selection_2023h2",
        "eval_2024",
    )
    assert (
        cogr.FOLD_BY_NAME["selection_2023h2"]["threshold_source"]
        == "calibration_2023h1"
    )
    assert (
        cogr.FOLD_BY_NAME["eval_2024"]["threshold_source"]
        == "selection_2023h2"
    )


def test_prior_bar_q75_threshold_no_outcomes_needed():
    scores = np.array([-2.0, -1.0, 4.0, 8.0])
    assert float(np.quantile(scores, 0.75)) == 5.0
    predictions = np.zeros((3, 4, 4))
    predictions[:, :, 0] = scores
    predictions[:, :, 2] = scores - 1
    chosen, side, *_ = cogr.adjusted_scores(predictions)
    np.testing.assert_allclose(chosen, scores)
    assert (side == 1).all()


def test_same_gross_units_scale_gross9_not_candidate_notional():
    combined, comparator = cogr.same_gross_weights(
        cogr.BASELINE_WEIGHTS, 0.5
    )
    assert combined[cogr.CANDIDATE_SLEEVE] == 0.5
    assert sum(combined.values()) == 9.5
    assert abs(sum(comparator.values()) - 9.5) < 1e-12
    assert comparator["frozen_annual_rank7"] == (
        cogr.BASELINE_WEIGHTS["frozen_annual_rank7"] * 9.5 / 9.0
    )


def test_delayed_control_containment_and_nonoverlap():
    aligned = pd.DataFrame(
        {
            "session_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "signal_position": [0, 100, 200, 900],
            "coordination_position": [-1, 99, 199, 899],
            "entry_position": [1, 101, 201, 901],
            "exit_position": [145, 245, 345, 1045],
        }
    )
    primary = [
        {
            "session_date": "2024-01-02",
            "signal_position": 0,
            "entry_position": 1,
            "exit_position": 145,
            "side": 1,
            "score": 1.0,
        },
        {
            "session_date": "2024-01-03",
            "signal_position": 100,
            "entry_position": 101,
            "exit_position": 245,
            "side": -1,
            "score": 1.0,
        },
        {
            "session_date": "2024-01-04",
            "signal_position": 200,
            "entry_position": 201,
            "exit_position": 345,
            "side": 1,
            "score": 1.0,
        },
    ]
    dates = pd.date_range("2024-01-01", periods=1000, freq="5min")
    delayed = cogr.delayed_control(
        primary,
        aligned,
        dates,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-04"),
    )
    assert [item["session_date"] for item in delayed] == ["2024-01-03"]
    assert delayed[0]["side"] == 1
    assert delayed[0]["exit_position"] == 245


def test_all_nine_controls_always_enter_best_comparison():
    baseline, _candidate, combined = _array_data()
    controls = {
        name: {"cagr_to_strict_mdd": float(index)}
        for index, name in enumerate(cogr.CONTROL_NAMES)
    }
    jaccards = {name: 0.0 for name in cogr.BASELINE_WEIGHTS}
    row = cogr.row_for_cell(
        "unrestricted",
        0.25,
        normal_data=combined,
        stress_data=combined,
        baseline_data=baseline,
        baseline_weights=cogr.BASELINE_WEIGHTS,
        primary_trades=[_trade()],
        control_metrics=controls,
        jaccards=jaccards,
        window_name="selection_2023h2",
        preregistration=_gate_prereg(),
    )
    assert tuple(row["controls"]) == cogr.CONTROL_NAMES
    assert row["best_control_cagr_mdd"] == 8.0
    with pytest.raises(RuntimeError, match="all nine"):
        cogr.row_for_cell(
            "unrestricted",
            0.25,
            normal_data=combined,
            stress_data=combined,
            baseline_data=baseline,
            baseline_weights=cogr.BASELINE_WEIGHTS,
            primary_trades=[_trade()],
            control_metrics={
                name: controls[name] for name in cogr.CONTROL_NAMES[:-1]
            },
            jaccards=jaccards,
            window_name="selection_2023h2",
            preregistration=_gate_prereg(),
        )


def test_nonempty_gross9_arrays_affect_same_gross_and_jaccard_gate():
    baseline, candidate, combined = _array_data()
    empty_baseline, _unused, candidate_only = _array_data(
        baseline_return=0.0,
        baseline_entries=(),
    )
    start, end = cogr.WINDOWS["selection_2023h2"]
    with_gross9 = cogr.portfolio_metrics(
        combined,
        combined,
        baseline,
        cogr.BASELINE_WEIGHTS,
        0.5,
        start=start,
        end=end,
    )
    without_gross9 = cogr.portfolio_metrics(
        candidate_only,
        candidate_only,
        empty_baseline,
        cogr.BASELINE_WEIGHTS,
        0.5,
        start=start,
        end=end,
    )
    assert (
        with_gross9["same_gross_comparator"]["absolute_return_pct"]
        != without_gross9["same_gross_comparator"]["absolute_return_pct"]
    )

    jaccards = {name: 0.0 for name in cogr.BASELINE_WEIGHTS}
    jaccards["cand_rex_veto_7"] = cogr.entry_jaccard(
        candidate["entry_positions"][cogr.CANDIDATE_SLEEVE],
        baseline["entry_positions"]["cand_rex_veto_7"],
    )
    controls = {
        name: {"cagr_to_strict_mdd": 0.0} for name in cogr.CONTROL_NAMES
    }
    row = cogr.row_for_cell(
        "unrestricted",
        0.5,
        normal_data=combined,
        stress_data=combined,
        baseline_data=baseline,
        baseline_weights=cogr.BASELINE_WEIGHTS,
        primary_trades=[_trade()],
        control_metrics=controls,
        jaccards=jaccards,
        window_name="selection_2023h2",
        preregistration=_gate_prereg(),
    )
    assert row["max_entry_jaccard"] == 1.0
    assert row["checks"]["entry_jaccard"] is False


def test_portfolio_statistical_checks_are_all_enforced():
    requirement = {
        "minimum_active_weeks_each_test": 26,
        "maximum_p_value_each_test": 0.1,
    }
    checks: dict[str, bool] = {}
    cogr.apply_statistical_checks(
        checks,
        {
            "standalone_active_weeks": 30,
            "standalone_sign_flip_p": 0.05,
            "portfolio_active_weeks": 25,
            "portfolio_sign_flip_p": 0.11,
            "portfolio_bootstrap_90pct_lower_mean_log_excess": 0.0,
        },
        requirement,
    )
    assert checks["standalone_active_weeks"] is True
    assert checks["standalone_sign_flip_p"] is True
    assert checks["portfolio_active_weeks"] is False
    assert checks["portfolio_sign_flip_p"] is False
    assert checks["portfolio_bootstrap_lower_positive"] is False
    assert not all(checks.values())


def test_weekly_signflip_and_bootstrap_are_deterministic():
    effects = np.array([0.01, 0.02, -0.005, 0.003])
    assert cogr.sign_flip_pvalue(
        effects, simulations=256, seed=123
    ) == cogr.sign_flip_pvalue(effects, simulations=256, seed=123)
    assert cogr.bootstrap_lower_mean(
        effects, simulations=256, seed=456
    ) == cogr.bootstrap_lower_mean(effects, simulations=256, seed=456)


def test_funding_prefix_stops_before_future_numeric_parse(tmp_path: Path):
    source = tmp_path / "funding.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write("date,symbol,funding_rate,funding_time,mark_price\n")
        handle.write(
            "2023-12-31 16:00:00,BTCUSDT,0.0001,1704038400000,42000\n"
        )
        handle.write(
            "2024-01-01 00:00:00,BTCUSDT,DO_NOT_PARSE,"
            "1704067200000,DO_NOT_PARSE\n"
        )
    frame = cogr.read_funding_prefix(source, cutoff="2024-01-01")
    assert tuple(frame.columns) == ("date", "funding_rate")
    assert len(frame) == 1
    assert frame.loc[0, "funding_rate"] == 0.0001
    assert frame.loc[0, "date"] == pd.Timestamp("2023-12-31 16:00:00")


def test_funding_prefix_rejects_nonexact_source_schema(tmp_path: Path):
    source = tmp_path / "funding-extra-column.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write(
            "date,symbol,funding_rate,funding_time,mark_price,unexpected\n"
        )
        handle.write(
            "2023-12-31 16:00:00,BTCUSDT,0.0001,"
            "1704038400000,42000,x\n"
        )
    with pytest.raises(RuntimeError, match="schema drifted"):
        cogr.read_funding_prefix(source, cutoff="2024-01-01")


def test_self_provenance_is_verified_from_preregistration(tmp_path: Path):
    configured: dict[str, str] = {}
    provenance: dict[str, dict[str, str]] = {}
    evaluator = tmp_path / "evaluate.py"
    for index, name in enumerate(cogr.INPUT_KEYS):
        path = evaluator if name == "cogr_evaluator" else tmp_path / f"{index}.bin"
        path.write_bytes(f"{name}\n".encode())
        configured[name] = str(path)
        provenance[name] = {"path": str(path), "sha256": cogr._sha256(path)}
    preregistration = {"input_provenance": provenance}
    records = cogr.validate_inputs(
        cogr.Config(),
        preregistration,
        evaluator_path=evaluator,
        configured_override=configured,
    )
    assert records["cogr_evaluator"]["sha256"] == cogr._sha256(evaluator)
    evaluator.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cogr_evaluator"):
        cogr.validate_inputs(
            cogr.Config(),
            preregistration,
            evaluator_path=evaluator,
            configured_override=configured,
        )


def _complete_selection_artifact() -> tuple[dict, dict, dict]:
    input_identity = {
        name: {
            "path": f"/frozen/{name}",
            "sha256": f"sha-{name}",
            "validated_against_preregistration": True,
        }
        for name in cogr.INPUT_KEYS
    }
    config = {"preregistration": "/frozen/preregistration.json"}
    rows = []
    for index, (mode, weight) in enumerate(
        (mode, weight)
        for mode in cogr.COORDINATION_MODES
        for weight in cogr.WEIGHTS
    ):
        passing = index == 0
        rows.append(
            {
                "coordination_mode": mode,
                "candidate_weight": weight,
                "passes": passing,
                "checks": {"frozen_gate": passing},
                "standalone": {},
                "stressed_standalone": {},
                "portfolio": {},
                "stressed_portfolio": {},
                "same_gross_comparator": {},
                "unscaled_gross9": {},
                "controls": {},
                "entry_jaccards": {},
            }
        )
    top = rows[0]
    raw = {
        "as_of": cogr.AS_OF,
        "phase": "selection",
        "config": config,
        "preregistration_sha256": "abc",
        "input_identity": input_identity,
        "gross9_selection_disclosure": {
            "candidate_metric_window": "2023H2",
            "future_candidate_data_opened": False,
            "future_gross9_metadata_exposed": False,
        },
        "model_meta": {
            "phase": "selection",
            "variants": {
                name: {
                    "variant": name,
                    "allowed_folds": list(cogr.PHASE_FOLDS["selection"]),
                }
                for name in ("primary", *cogr.FEATURE_CONTROL_NAMES)
            },
        },
        "schedule_meta": {mode: {} for mode in cogr.COORDINATION_MODES},
        "decision": "freeze_top1_for_eval",
        "eval_opened": False,
        "rank2_opened": False,
        "tested_cells": 12,
        "rows": rows,
        "frozen_top1": top,
    }
    return cogr.finalize_payload(raw), input_identity, config


def test_eval_verifies_selection_result_hash_prereg_schema_and_top1():
    selection, input_identity, config = _complete_selection_artifact()
    selection = json.loads(cogr.canonical_json(selection))
    top = selection["frozen_top1"]
    assert cogr.verify_selection_artifact(
        selection,
        preregistration_sha256="abc",
        expected_input_identity=input_identity,
        expected_config=config,
    ) == top
    selection["candidate_tamper"] = True
    with pytest.raises(RuntimeError, match="result_hash"):
        cogr.verify_selection_artifact(
            selection,
            preregistration_sha256="abc",
            expected_input_identity=input_identity,
            expected_config=config,
        )


def test_eval_rejects_forged_selection_even_with_recomputed_self_hash():
    frozen, input_identity, config = _complete_selection_artifact()
    forged = dict(frozen)
    forged["frozen_top1"] = dict(forged["rows"][1])
    forged["frozen_top1"]["passes"] = True
    forged["frozen_top1"]["checks"] = {"frozen_gate": True}
    forged_rows = [dict(row) for row in forged["rows"]]
    forged_rows[0]["passes"] = False
    forged_rows[0]["checks"] = {"frozen_gate": False}
    forged_rows[1] = forged["frozen_top1"]
    forged["rows"] = forged_rows
    forged = cogr.finalize_payload(forged)
    assert cogr.verify_selection_artifact(
        forged,
        preregistration_sha256="abc",
        expected_input_identity=input_identity,
        expected_config=config,
    ) == forged["frozen_top1"]
    with pytest.raises(RuntimeError, match="deterministic selection replay"):
        cogr.verify_reproduced_selection(forged, frozen)


def test_selection_payload_does_not_disclose_2024_gross9_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    identity = {name: {} for name in cogr.INPUT_KEYS}
    context = {
        "aligned": object(),
        "candidate_market": object(),
        "funding": object(),
        "authoritative_anchor_validated": True,
        "gross9_meta": {"counts": {"test2024": 1}},
        "state_meta": {"test2024": {"state_hash": "forbidden"}},
    }
    monkeypatch.setattr(cogr, "load_preregistration", lambda _path: {})
    monkeypatch.setattr(cogr, "_sha256", lambda _path: "prereg-sha")
    monkeypatch.setattr(
        cogr, "validate_inputs", lambda _cfg, _preregistration: identity
    )
    monkeypatch.setattr(
        cogr,
        "build_context",
        lambda _cfg, _preregistration, phase: context,
    )
    monkeypatch.setattr(
        cogr,
        "prediction_variants",
        lambda *_args, **_kwargs: ({}, {"phase": "selection"}),
    )
    monkeypatch.setattr(
        cogr,
        "_mode_rows",
        lambda *_args, **_kwargs: ([], {}),
    )
    payload = cogr.build_selection_payload(cogr.Config())
    assert "gross9_source_meta" not in payload
    assert "gross9_observable_state" not in payload
    assert payload["gross9_selection_disclosure"] == {
        "candidate_metric_window": "2023H2",
        "future_candidate_data_opened": False,
        "future_gross9_metadata_exposed": False,
    }


def test_json_and_docs_outputs_are_byte_deterministic(tmp_path: Path):
    payload = {
        "as_of": cogr.AS_OF,
        "phase": "selection",
        "decision": "reject_no_passing_2023h2_cell",
        "tested_cells": 12,
        "frozen_top1": None,
        "preregistration_sha256": "abc",
    }
    first_json = tmp_path / "first.json"
    first_docs = tmp_path / "first.md"
    second_json = tmp_path / "second.json"
    second_docs = tmp_path / "second.md"
    cogr._write_outputs(payload, first_json, first_docs)
    cogr._write_outputs(payload, second_json, second_docs)
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_docs.read_bytes() == second_docs.read_bytes()
