from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from training import optimize_gross9_overlap_portfolio as opt


def _market(start: str, end: str, price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="5min", tz="UTC")
    return pd.DataFrame({"date": dates, "open": price, "high": price, "low": price})


def _funding() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime([], utc=True), "funding_rate": [], "mark_price": []})


def _clock(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _sleeve(sleeve_id: str, rows: list[dict[str, object]]) -> opt.SleeveClock:
    return opt.SleeveClock(
        sleeve_id=sleeve_id,
        clock_path=None,
        clock_sha256=None,
        clock=opt.normalize_sleeve_clock(
            _clock(rows),
            sleeve_id=sleeve_id,
            start=opt._utc("2023-07-01T00:00:00Z"),
            end=opt._utc("2023-08-01T00:00:00Z"),
        ),
        source={},
    )


def test_inter_sleeve_overlap_allowed_but_gross_risk_does_not_net_execution_nets() -> None:
    long = _sleeve("long", [{"entry_time": "2023-07-01T00:00:00Z", "exit_time": "2023-07-01T08:00:00Z", "side": 1}])
    short = _sleeve("short", [{"entry_time": "2023-07-01T00:00:00Z", "exit_time": "2023-07-01T08:00:00Z", "side": -1}])
    spec = opt.PortfolioSpec(weights=(("long", 0.2), ("short", 0.2)), proxy_score=0.0, proxy_metrics={})
    clock = opt.build_portfolio_clock(spec, {"long": long, "short": short})

    risk = opt.exposure_and_turnover(clock, opt._utc("2023-07-01T00:00:00Z"), opt._utc("2023-07-01T08:00:00Z"))
    assert risk["max_gross_exposure"] == pytest.approx(0.4)
    assert risk["max_abs_net_exposure"] == pytest.approx(0.0)

    ledger = opt.fixed_ledger.simulate_portfolio(
        clock,
        _market("2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z"),
        _funding(),
        opt._utc("2023-07-01T00:00:00Z"),
        opt._utc("2023-07-01T08:00:00Z"),
        cost=0.0006,
    )
    assert ledger["total_fees"] == pytest.approx(0.0)
    assert ledger["final_equity"] == pytest.approx(100_000.0)


def test_intra_sleeve_overlap_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="intra-sleeve overlap"):
        _sleeve(
            "bad",
            [
                {"entry_time": "2023-07-01T00:00:00Z", "exit_time": "2023-07-01T08:00:00Z", "side": 1},
                {"entry_time": "2023-07-01T04:00:00Z", "exit_time": "2023-07-01T12:00:00Z", "side": 1},
            ],
        )


def test_overlap_config_freezes_allowed_overlap_and_holdout_boundary() -> None:
    cfg = opt.build_overlap_allowed_config({"sleeve_weights": {"a": 0.25, "b": 0.25}})
    assert cfg["policy_id"] == "G9-OVERLAP-PORT-1"
    assert cfg["overlap_policy"]["inter_sleeve_overlap_allowed"] is True
    assert cfg["overlap_policy"]["intra_sleeve_overlap_allowed"] is False
    assert cfg["overlap_policy"]["gross_risk_nets_opposite_sides"] is False
    assert cfg["risk_caps"]["gross_risk_does_not_net"] is True
    assert cfg["optimizer"]["december_holdout_unopened"] is True
    assert cfg["evidence_boundary"]["december_holdout_opened_by_selection"] is False
    core = {k: v for k, v in cfg.items() if k != "protocol_hash"}
    assert cfg["protocol_hash"] == opt.canonical_hash(core)
    rejected = opt.build_overlap_allowed_config({"sleeve_weights": {"a": 0.25}, "passed": False})
    assert rejected["status"] == "terminal_train_reject_diagnostic_config_not_live"


def test_beam_search_is_deterministic_and_respects_grid_constraints() -> None:
    idx = pd.to_datetime(["2023-07-03T00:00:00Z", "2023-07-10T00:00:00Z"], utc=True)
    effects = {
        "b": pd.Series([0.02, 0.02], index=idx),
        "a": pd.Series([0.03, -0.005], index=idx),
        "c": pd.Series([-0.01, 0.04], index=idx),
    }
    cfg = opt.OptimizerConfig(weight_grid=(0.05, 0.10), gross_grid=(0.10, 0.15, 0.20), min_gross=0.10, max_gross=0.20, max_sleeves=2, beam_width=4, proxy_candidate_cap=100, exact_finalist_count=4)
    first = opt.beam_search_portfolios(effects, cfg)
    second = opt.beam_search_portfolios(copy.deepcopy(effects), cfg)
    assert [p.weights for p in first] == [p.weights for p in second]
    assert first
    assert all(0.10 <= p.gross <= 0.20 for p in first)
    assert all(len(p.weights) <= 2 for p in first)
    assert all(weight in {0.05, 0.10} for p in first for _, weight in p.weights)


def test_exact_finalist_selection_uses_authoritative_raw_score_and_gates(monkeypatch) -> None:
    base_rows = [{"entry_time": "2023-07-01T00:00:00Z", "exit_time": "2023-07-01T08:00:00Z", "side": 1}]
    sleeves = {sid: _sleeve(sid, base_rows) for sid in ("a", "b", "c")}
    specs = [
        opt.PortfolioSpec(weights=(("a", 0.25),), proxy_score=99.0, proxy_metrics={}),
        opt.PortfolioSpec(weights=(("b", 0.25),), proxy_score=1.0, proxy_metrics={}),
        opt.PortfolioSpec(weights=(("c", 0.25),), proxy_score=0.5, proxy_metrics={}),
    ]

    def fake_primary(clock, market, funding, start, end):
        sid = str(clock.sleeve.iloc[0])
        if sid == "a":
            ret, ratio = -1.0, -1.0
        elif sid == "b":
            ret, ratio = 2.0, 4.0
        else:
            ret, ratio = 1.0, 3.1
        return {
            "base": {"absolute_return_pct": ret, "cagr_to_strict_mdd": ratio, "strict_mdd_pct": 1.0, "intervals": 20, "mean_exposure_weighted_gross_edge_bp": 21.0},
            "stress": {"absolute_return_pct": max(ret - 0.1, -2.0), "cagr_to_strict_mdd": max(ratio - 0.1, -2.0)},
            "calendar_halves": {"first": {"absolute_return_pct": 1.0}, "second": {"absolute_return_pct": 1.0}},
            "cluster_signflip": {"pvalue": 0.05},
        }

    monkeypatch.setattr(opt.fixed_ledger, "evaluate_primary", fake_primary)
    monkeypatch.setattr(opt, "evaluate_monthly_stability", lambda *args: [{"base_return_pct": 1.0, "stress_return_pct": 1.0}] * 5)
    cfg = opt.OptimizerConfig(min_trade_count=1, min_active_weeks=0, exact_finalist_count=3, max_turnover_weight_per_day=999, max_month_share=1.0, max_sleeve_turnover_share=1.0)
    evaluated = opt.evaluate_exact_finalists(specs, sleeves, pd.DataFrame(), pd.DataFrame(), opt._utc("2023-07-01"), opt._utc("2023-07-02"), cfg)
    winner = opt.select_authoritative_rank1(evaluated)
    assert winner["sleeve_weights"] == {"b": 0.25}
    assert winner["proxy_rank"] == 2
    assert evaluated[0]["passed"] is True


def test_load_sleeve_clock_rejects_holdout_and_oos_windows(tmp_path: Path) -> None:
    path = tmp_path / "clock.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("split,entry_time,exit_time,side\n")
        handle.write("train,2023-12-01T00:00:00Z,2023-12-01T08:00:00Z,1\n")
    record = {"sleeve_id": "x", "stage_clocks": {"train": {"path": str(path), "sha256": opt.sha256_file(path)}, "test": {"path": str(path)}}}
    with pytest.raises(RuntimeError, match="December holdout"):
        opt.load_sleeve_clock(record, "train", opt._utc("2023-07-01"), opt._utc("2023-12-02"))
    with pytest.raises(RuntimeError, match="only load train"):
        opt.load_sleeve_clock(record, "test", opt._utc("2023-07-01"), opt._utc("2023-08-01"))


def test_max_t_api_is_deterministic_and_familywise() -> None:
    effects = {"b": [0.1, -0.02, 0.03], "a": [0.05, 0.01, 0.02]}
    first = opt.max_t_signflip_pvalue(effects, draws=500, seed=7)
    second = opt.max_t_signflip_pvalue(effects, draws=500, seed=7)
    assert first == second
    assert first["method"] == "shared_weekly_signflip_max_t"
    assert set(first["adjusted_pvalues"]) == {"a", "b"}
    assert all(0.0 < p <= 1.0 for p in first["adjusted_pvalues"].values())


def test_market_proxy_ignores_cached_outcome_columns() -> None:
    clock = pd.DataFrame(
        {
            "entry_time": [pd.Timestamp("2023-07-01T00:00:00Z")],
            "exit_time": [pd.Timestamp("2023-07-01T08:00:00Z")],
            "side": [1],
            "net_return": [0.50],
        }
    )
    opens = {
        pd.Timestamp("2023-07-01T00:00:00Z"): 100.0,
        pd.Timestamp("2023-07-01T08:00:00Z"): 100.0,
    }
    effect = opt._row_effects(clock, opens)
    assert effect.iloc[0] < 0.0


def test_universe_loader_rejects_wrong_identity_even_with_valid_manifest(tmp_path: Path) -> None:
    inventory = {"manifest_hash": "inventory"}
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    sleeves = [
        {"sleeve_id": f"h{i}", "provenance": {"kind": "historical_gross9_near6h_only_reject"}}
        for i in range(64)
    ] + [
        {"sleeve_id": f"a{i}", "provenance": {"kind": "active_veto_duplicate_only_canonical"}}
        for i in range(7)
    ]
    core = {
        "policy_id": "WRONG",
        "protocol_version": opt.universe_builder.PROTOCOL_VERSION,
        "precanonical_schedule_count": 71,
        "canonical_sleeve_count": 71,
        "historical_novelty_inventory": {
            "path": str(inventory_path),
            "sha256": opt.sha256_file(inventory_path),
            "manifest_hash": "inventory",
        },
        "sleeves": sleeves,
    }
    path = tmp_path / "universe.json"
    path.write_text(json.dumps({**core, "manifest_hash": opt.canonical_hash(core)}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity/count drift"):
        opt.load_universe_manifest(path)


def test_exact_replay_requires_frozen_finalist_count() -> None:
    cfg = opt.OptimizerConfig(exact_finalist_count=2)
    with pytest.raises(RuntimeError, match="requires at least 2 proxy finalists"):
        opt.evaluate_exact_finalists([], {}, pd.DataFrame(), pd.DataFrame(), opt._utc("2023-07-01"), opt._utc("2023-12-01"), cfg)


def test_persistent_selection_requires_preregistration_and_source_receipts(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires frozen preregistration and source receipts"):
        opt.optimize_from_manifest(
            tmp_path / "universe.json",
            pd.DataFrame(),
            pd.DataFrame(),
            output=tmp_path / "result.json",
        )


def test_train_source_receipt_loader_records_hash_bound_inputs(monkeypatch) -> None:
    start = opt._utc(opt.TRAIN_PROXY_WINDOW[0])
    end = opt._utc(opt.TRAIN_PROXY_WINDOW[1])
    market = pd.DataFrame({"date": [start, end]})
    funding = pd.DataFrame({"date": [start, end - pd.Timedelta(hours=8)]})
    monkeypatch.setattr(opt.train_sources, "load_market_hash_bound", lambda a, b: market)
    monkeypatch.setattr(opt.train_sources, "load_train_funding_hash_bound", lambda a, b: funding)
    loaded_market, loaded_funding, receipt = opt.load_bound_selection_sources(start, end)
    assert loaded_market is market and loaded_funding is funding
    assert receipt["market"]["sha256"] == opt.train_sources.econ.v1.MARKET_SHA
    assert receipt["funding"]["sha256"] == opt.train_sources.econ.TRAIN_FUNDING_SHA
