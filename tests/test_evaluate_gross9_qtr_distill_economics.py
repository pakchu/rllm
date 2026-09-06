from __future__ import annotations

import gzip
import json
import math
import types
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_gross9_qtr_distill_economics as qtr


def _market(start="2024-01-01T00:00:00Z", periods=7, opens=None, highs=None, lows=None):
    dates = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    if opens is None:
        opens = [100.0] * periods
    if highs is None:
        highs = [o + 1 for o in opens]
    if lows is None:
        lows = [o - 1 for o in opens]
    return pd.DataFrame({"date": dates, "open": opens, "high": highs, "low": lows, "close": opens})


def _funding(times=None, rates=None, marks=None):
    times = times or ["2024-01-01T00:00:00Z"]
    rates = rates or [0.0] * len(times)
    marks = marks or [100.0] * len(times)
    return pd.DataFrame({"date": pd.to_datetime(times, utc=True), "funding_rate": rates, "mark_price": marks})


def _four_sleeve_clock(rows):
    base = []
    for i, sleeve in enumerate(["a", "b", "c", "d"]):
        base.append({"sleeve": sleeve, "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:25:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:30:00Z"), "side": 1})
    base.extend(rows)
    return pd.DataFrame(base)


def _auth(sleeves=None, episodes=None):
    sleeves = sleeves or [qtr.SleeveSpec(name=n, weight=0.1, clock_path=Path(f"{n}.csv.gz"), clock_sha256="0" * 64) for n in ["a", "b", "c", "d"]]
    return qtr.FrozenAuthorization(
        preregistration={"manifest_hash": "prehash", "active_veto_terminal_artifacts": {"gross9_novelty": {"path": "novelty.json"}}},
        clock_package={"manifest_hash": "clockhash"},
        novelty={"manifest_hash": "novhash"},
        sleeves=sleeves,
        source_signed_episodes_by_split=episodes or {"train": 96, "test": 204, "eval": 192, "final": 98},
        preliminary_train_receipt_support={"commit": "cbb5f8bc"},
    )


def _write_json_with_manifest(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload["manifest_hash"] = qtr.canonical_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("x\n")
            return
        columns = list(rows[0])
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row[column]) for column in columns) + "\n")


def test_frozen_authorization_loader_binds_prereg_clock_package_schedules_and_train_novelty(monkeypatch, tmp_path):
    prereg_payload = _write_json_with_manifest(tmp_path / "prereg.json", {
        "policy_id": qtr.POLICY_ID,
        "preliminary_train_receipt_support": {"commit": "cbb5f8bc"},
    })
    fake_module = types.SimpleNamespace(build=lambda: prereg_payload, validate=lambda value: None)
    monkeypatch.setattr(qtr.importlib, "import_module", lambda name: fake_module)

    builder = tmp_path / "builder.py"; builder.write_text("# builder\n", encoding="utf-8")
    sleeves = {}
    for base in ["a", "b", "c", "d"]:
        clock = tmp_path / f"{base}.csv.gz"
        _write_gzip_csv(clock, [{"split": "train", "entry_time": "2023-07-01T00:00:00Z", "exit_time": "2023-07-01T08:00:00Z", "side": 1}])
        sleeves[base] = {"sleeve_id": base, "weight": 0.1, "clock": {"path": str(clock), "sha256": qtr.sha256_file(clock), "rows": 1}}
    schedules = {}
    for name in ["transitions", "segments", "signed_episodes"]:
        schedule = tmp_path / f"{name}.csv.gz"
        _write_gzip_csv(schedule, [{"x": 1}, {"x": 2}])
        schedules[name] = {"path": str(schedule), "sha256": qtr.sha256_file(schedule), "rows": 2}
    package = _write_json_with_manifest(tmp_path / "clock_package.json", {
        "policy_id": qtr.POLICY_ID,
        "decision": "materialized_shadow_distilled_clock_package",
        "preregistration": {"path": str(tmp_path / "prereg.json"), "sha256": qtr.sha256_file(tmp_path / "prereg.json"), "manifest_hash": prereg_payload["manifest_hash"], "status": "validated_against_committed_preregistration"},
        "implementation": {"builder": {"path": str(builder), "sha256": qtr.sha256_file(builder)}},
        "components": {"base_order": ["a", "b", "c", "d"]},
        "sleeves": sleeves,
        "portfolio_schedules": schedules,
        "portfolio_source_stats": {"splits": {"train": {"signed_episodes": 4}, "test": {"signed_episodes": 12}, "eval": {"signed_episodes": 13}, "final": {"signed_episodes": 8}}},
    })

    novelty_payload = _write_json_with_manifest(tmp_path / "novelty.json", {
        "policy_id": qtr.POLICY_ID,
        "decision": "pass_g9qtr_distill_to_economic_outcomes",
        "advance_to_economic_outcomes": True,
        "gross9_pass": True,
        "preregistration": {
            "path": str(tmp_path / "prereg.json"),
            "sha256": qtr.sha256_file(tmp_path / "prereg.json"),
            "manifest_hash": prereg_payload["manifest_hash"],
        },
        "source_package": {
            "path": str(tmp_path / "clock_package.json"),
            "sha256": qtr.sha256_file(tmp_path / "clock_package.json"),
            "manifest_hash": package["manifest_hash"],
            "predecessor_mutated": False,
        },
    })
    monkeypatch.setattr(qtr, "TRAIN_NOVELTY", tmp_path / "novelty.json")

    auth = qtr.load_frozen_authorization(tmp_path / "prereg.json", tmp_path / "clock_package.json")
    assert [s.name for s in auth.sleeves] == ["a", "b", "c", "d"]
    assert auth.clock_package["manifest_hash"] == package["manifest_hash"]
    assert auth.source_signed_episodes_by_split["test"] == 12
    assert auth.preliminary_train_receipt_support == {"commit": "cbb5f8bc"}

    bad = dict(novelty_payload, advance_to_economic_outcomes=False)
    bad["manifest_hash"] = qtr.canonical_hash({k: v for k, v in bad.items() if k != "manifest_hash"})
    (tmp_path / "novelty.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="train novelty binding drift|train novelty did not authorize"):
        qtr.load_frozen_authorization(tmp_path / "prereg.json", tmp_path / "clock_package.json")


def test_same_timestamp_exits_entries_are_net_costed_and_new_quantities_use_same_pre_equity():
    market = _market(periods=5)
    funding = _funding(times=["2024-01-01T00:00:00Z"], rates=[0.0])
    clock = pd.DataFrame([
        {"sleeve": "a", "weight": 0.5, "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
        {"sleeve": "b", "weight": 0.5, "entry_time": pd.Timestamp("2024-01-01T00:10:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:20:00Z"), "side": 1},
        {"sleeve": "c", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:15:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:20:00Z"), "side": 1},
        {"sleeve": "d", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:15:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:20:00Z"), "side": -1},
    ])
    result = qtr.simulate_portfolio(clock, market, funding, market.date.iloc[0], market.date.iloc[-1], cost=0.001, initial_equity=100_000.0)
    transitions = {row["time"]: row for row in result["transition_rows"]}

    first = transitions[pd.Timestamp("2024-01-01T00:00:00Z")]
    assert first["q_before"] == 0.0
    assert first["q_after"] == pytest.approx(500.0)
    assert first["fee"] == pytest.approx(50.0)

    handoff = transitions[pd.Timestamp("2024-01-01T00:10:00Z")]
    assert handoff["exits"] == ["a"]
    assert handoff["entries"] == ["b"]
    # Exiting + entering same side at same open nets to a tiny fee only for the
    # equity-after-fee sizing difference, not two full notionals.
    assert abs(handoff["delta_q"]) < 1.0
    assert handoff["fee"] < 0.10
    assert handoff["q_after"] == pytest.approx(0.5 * handoff["equity_pre"] / 100.0)


def test_fixed_quantities_do_not_rebalance_after_price_move_and_final_exit_charges_cost():
    market = _market(periods=4, opens=[100, 110, 120, 130], highs=[100, 110, 120, 130], lows=[100, 110, 120, 130])
    funding = _funding()
    clock = _four_sleeve_clock([
        {"sleeve": "a", "weight": 0.5, "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:15:00Z"), "side": 1},
    ])
    # Remove dummy future rows outside market by using exactly four active sleeves inside the window.
    clock = pd.DataFrame([
        {"sleeve": "a", "weight": 0.5, "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:15:00Z"), "side": 1},
        {"sleeve": "b", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
        {"sleeve": "c", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": -1},
        {"sleeve": "d", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
    ])
    result = qtr.simulate_portfolio(clock, market, funding, market.date.iloc[0], market.date.iloc[-1], cost=0.0, initial_equity=100_000.0)
    entry = result["transition_rows"][0]
    assert entry["q_after"] == pytest.approx(500.0)
    # Deterministic value includes b/c/d fixed-quantity intraday sleeves; no interval is resized after later price moves.
    assert result["final_equity"] == pytest.approx(115_954.54545454546, rel=1e-9)


def test_funding_uses_post_transition_aggregate_q_so_exiting_position_excluded_and_new_entry_included():
    market = _market(periods=3)
    funding = _funding(times=["2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z"], rates=[0.01, 0.01], marks=[100.0, 100.0])
    clock = pd.DataFrame([
        {"sleeve": "a", "weight": 0.5, "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
        {"sleeve": "b", "weight": 0.25, "entry_time": pd.Timestamp("2024-01-01T00:10:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z") + pd.Timedelta(minutes=0), "side": 1},
        {"sleeve": "c", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": -1},
        {"sleeve": "d", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
    ])
    # b must be non-zero interval; extend market one more row.
    market = _market(periods=4)
    clock.loc[1, "exit_time"] = pd.Timestamp("2024-01-01T00:15:00Z")
    result = qtr.simulate_portfolio(clock, market, funding, market.date.iloc[0], market.date.iloc[-1], cost=0.0, initial_equity=100_000.0)
    effects = {row["time"]: row for row in result["equity_effect_rows"]}
    assert effects[pd.Timestamp("2024-01-01T00:00:00Z")]["funding_cash"] == pytest.approx(-500.0)
    # At 00:10 a exits before funding and b enters before funding, so only b's post-transition q pays.
    assert effects[pd.Timestamp("2024-01-01T00:10:00Z")]["funding_cash"] == pytest.approx(-248.75, rel=1e-6)


def test_strict_mdd_uses_favorable_then_adverse_global_hwm_with_virtual_liquidation_cost():
    market = _market(periods=3, opens=[100, 100, 100], highs=[120, 100, 100], lows=[80, 100, 100])
    funding = _funding()
    clock = pd.DataFrame([
        {"sleeve": "a", "weight": 1.0, "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
        {"sleeve": "b", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": -1},
        {"sleeve": "c", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
        {"sleeve": "d", "weight": 0.1, "entry_time": pd.Timestamp("2024-01-01T00:05:00Z"), "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"), "side": 1},
    ])
    result = qtr.simulate_portfolio(clock, market, funding, market.date.iloc[0], market.date.iloc[-1], cost=0.001, initial_equity=100_000.0)
    # Peak after favorable high is 119900 (entry fee paid); adverse equity is 79820 (includes virtual cost at low).
    assert result["strict_mdd_pct"] == pytest.approx((1 - 79_820.0 / 119_900.0) * 100.0, rel=1e-6)


def test_cluster_signflip_is_utc_week_deterministic():
    rows = [
        {"time": pd.Timestamp("2024-01-01T00:00:00Z"), "log_effect": 0.01},
        {"time": pd.Timestamp("2024-01-03T00:00:00Z"), "log_effect": 0.02},
        {"time": pd.Timestamp("2024-01-08T00:00:00Z"), "log_effect": -0.01},
    ]
    a = qtr.cluster_signflip(rows, draws=1000, seed=7)
    b = qtr.cluster_signflip(rows, draws=1000, seed=7)
    assert a == b
    assert a["clusters"] == 2
    assert a["draws"] == 1000


def test_train_source_loader_uses_bound_active_veto_source_module(monkeypatch):
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = pd.Timestamp("2024-01-01T00:00:00Z")
    market = _market(start=str(start), periods=2)
    funding = _funding(times=[str(start)])
    monkeypatch.setattr(
        qtr.train_sources,
        "load_market_hash_bound",
        lambda observed_start, observed_end: market,
    )
    monkeypatch.setattr(
        qtr.train_sources,
        "load_train_funding_hash_bound",
        lambda observed_start, observed_end: funding,
    )

    loaded_market, loaded_funding, source = qtr.load_sources("train", start, end)

    assert loaded_market is market
    assert loaded_funding is funding
    assert source["mode"] == "hash_bound_gzip_physical_prefix"


def test_stage_checks_train_reports_legacy_bonferroni_but_oos_p_point_one_authorizes(monkeypatch, tmp_path):
    primary = {
        "base": {"absolute_return_pct": 1, "cagr_to_strict_mdd": 3, "strict_mdd_pct": 10, "mean_exposure_weighted_gross_edge_bp": 21},
        "stress": {"absolute_return_pct": 1, "cagr_to_strict_mdd": 2.5},
        "calendar_halves": {"first": {"absolute_return_pct": 1}, "second": {"absolute_return_pct": 1}},
        "cluster_signflip": {"pvalue": 0.09},
    }
    assert "oos_cluster_signflip_p_max_0_1" not in qtr.stage_checks("train", primary)
    checks = qtr.stage_checks("test", primary, source_signed_episodes=12)
    assert checks["oos_cluster_signflip_p_max_0_1"] is True
    assert checks["source_min_nonzero_signed_episodes"] is True
    primary["cluster_signflip"] = {"pvalue": 0.11}
    assert qtr.stage_checks("test", primary, source_signed_episodes=12)["oos_cluster_signflip_p_max_0_1"] is False
    assert qtr.stage_checks("test", primary, source_signed_episodes=11)["source_min_nonzero_signed_episodes"] is False
    assert qtr.TRAIN_LEGACY_BONFERRONI_P_MAX == pytest.approx(0.1 / 72)


def test_predecessor_hash_pass_gate_blocks_before_loader_opening(monkeypatch, tmp_path):
    called = {"load": False}
    monkeypatch.setattr(qtr, "load_frozen_authorization", lambda: _auth())
    monkeypatch.setattr(qtr, "load_portfolio_clock", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loader opened")))
    outputs = {stage: tmp_path / f"{stage}.json" for stage in qtr.STAGES}
    with pytest.raises(RuntimeError, match="missing predecessor train"):
        qtr.run("test", output=tmp_path / "test.json", sleeves=[], outputs=outputs)

    core = {"protocol_version": qtr.PROTOCOL_VERSION, "policy_id": qtr.POLICY_ID, "stage": "train", "passed": False, "advance_to_next_stage": False}
    outputs["train"].write_text(json.dumps({**core, "manifest_hash": qtr.canonical_hash(core)}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="predecessor did not pass"):
        qtr.run("test", output=tmp_path / "test.json", sleeves=[], outputs=outputs)


def test_run_writes_manifest_and_never_opens_later_stage_without_passed_predecessor(monkeypatch, tmp_path):
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = pd.Timestamp("2024-01-01T00:00:00Z")
    clock = pd.DataFrame([
        {"sleeve": "a", "weight": 0.1, "entry_time": start, "exit_time": start + pd.Timedelta(minutes=5), "side": 1},
        {"sleeve": "b", "weight": 0.1, "entry_time": start, "exit_time": start + pd.Timedelta(minutes=5), "side": 1},
        {"sleeve": "c", "weight": 0.1, "entry_time": start, "exit_time": start + pd.Timedelta(minutes=5), "side": -1},
        {"sleeve": "d", "weight": 0.1, "entry_time": start, "exit_time": start + pd.Timedelta(minutes=5), "side": 1},
    ])
    monkeypatch.setattr(qtr, "load_frozen_authorization", lambda: _auth())
    monkeypatch.setattr(qtr, "load_portfolio_clock", lambda *a, **k: clock)
    monkeypatch.setattr(qtr, "load_sources", lambda stage, s, e: (_market(str(s), periods=2, opens=[100, 110], highs=[110, 110], lows=[100, 110]), _funding(times=[str(s)], rates=[0.0]), {"mode": "unit"}))
    monkeypatch.setattr(qtr, "validate_market", lambda *a, **k: None)
    monkeypatch.setattr(qtr, "validate_funding", lambda *a, **k: None)
    monkeypatch.setattr(qtr, "evaluate_primary", lambda *a, **k: {
        "base": {"absolute_return_pct": 1, "cagr_to_strict_mdd": 3, "strict_mdd_pct": 10, "mean_exposure_weighted_gross_edge_bp": 21},
        "stress": {"absolute_return_pct": 1, "cagr_to_strict_mdd": 2.5},
        "calendar_halves": {"first": {"absolute_return_pct": 1}, "second": {"absolute_return_pct": 1}},
        "cluster_signflip": {"pvalue": 0.5},
    })
    monkeypatch.setattr(qtr, "STAGES", {**qtr.STAGES, "train": ("train", str(start), str(start + pd.Timedelta(minutes=5)))})

    output = tmp_path / "train.json"
    result = qtr.run("train", output=output, sleeves=_auth().sleeves)
    written = json.loads(output.read_text())
    assert written["manifest_hash"] == qtr.canonical_hash({k: v for k, v in written.items() if k != "manifest_hash"})
    assert result["later_stage_outcomes_opened"] is False
    assert result["train_legacy_cluster_diagnostic"]["reported_not_pass_authorizing"] is True
    assert result["status"] == "post_selection_train_shape_shadow"
    assert result["decision"] == "post_selection_train_shape_shadow"
    assert result["formal_legacy_train_pass"] is False
    assert result["frozen_authorization"]["preliminary_train_receipt_support"] == {"commit": "cbb5f8bc"}
