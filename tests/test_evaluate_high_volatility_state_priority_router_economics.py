from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_high_volatility_state_priority_router_economics as evaluator


def _primary(ratio: float = 3.1, absolute_return: float = 10.0, pvalue: float = 0.01) -> dict:
    return {
        "base": {
            "absolute_return_pct": absolute_return,
            "cagr_to_strict_mdd": ratio,
            "strict_mdd_pct": 5.0,
            "mean_gross_underlying_bp": 21.0,
        },
        "stress": {"absolute_return_pct": 1.0, "cagr_to_strict_mdd": 2.6},
        "cluster_signflip": {"pvalue": pvalue},
        "calendar_halves": {
            "first": {"absolute_return_pct": 1.0},
            "second": {"absolute_return_pct": 1.0},
        },
    }


def _manifest(path: Path, **overrides) -> dict:
    winner = evaluator.ELIGIBLE_CANDIDATES[0]
    core = {
        "policy_id": evaluator.POLICY_ID,
        "candidate": winner,
        "stage": "train",
        "passed": True,
        "frozen_train_winner": winner,
        "substitution_authorized": False,
        "predecessor": None,
        "selection": {"raw_train_rank_one": winner},
        **overrides,
    }
    report = {**core, "manifest_hash": evaluator.canonical_hash(core)}
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def test_frozen_bindings_and_four_candidate_authorization() -> None:
    assert evaluator.POLICY_ID == "HVSPR-8"
    assert evaluator.LEVERAGE == 0.5
    assert evaluator.BASE_COST == 0.0006
    assert evaluator.STRESS_COST == 0.0010
    assert evaluator.TRAIN_BONFERRONI_RAW_P_MAX == pytest.approx(0.025)
    assert evaluator.sha256(evaluator.PREREG) == evaluator.PREREG_SHA
    assert evaluator.sha256(evaluator.SUPPORT) == evaluator.SUPPORT_SHA
    assert evaluator.sha256(evaluator.GROSS9) == evaluator.GROSS9_SHA
    for binding in evaluator.CLOCKS.values():
        assert evaluator.sha256(binding["path"]) == binding["sha256"]

    gross9, predecessor = evaluator.verify("train")
    assert predecessor is None
    assert gross9["eligible_routers_for_economics"] == list(evaluator.ELIGIBLE_CANDIDATES)
    assert gross9["advance_to_economic_outcomes"] is True
    assert gross9["evidence_boundary"]["outcomes_opened"] is False


def test_train_bonferroni_is_additional_to_every_normal_gate() -> None:
    assert all(evaluator.economic_checks("test", _primary(pvalue=0.03)).values())
    train = evaluator.economic_checks("train", _primary(pvalue=0.03))
    assert train["cluster_signflip_p_max_0_1"] is True
    assert train["train_bonferroni_raw_weekly_p_max_0_025"] is False
    assert all(evaluator.economic_checks("train", _primary(pvalue=0.025)).values())


def test_raw_ranking_uses_ratio_return_then_frozen_family_order() -> None:
    first, second, third, fourth = evaluator.ELIGIBLE_CANDIDATES
    results = {
        first: {"primary": _primary(ratio=4.0, absolute_return=5.0)},
        second: {"primary": _primary(ratio=4.0, absolute_return=6.0)},
        third: {"primary": _primary(ratio=5.0, absolute_return=1.0)},
        fourth: {"primary": _primary(ratio=3.0, absolute_return=20.0)},
    }
    assert evaluator.rank_train(results) == [third, second, first, fourth]

    tied = {
        candidate: {"primary": _primary(ratio=4.0, absolute_return=5.0)}
        for candidate in reversed(evaluator.ELIGIBLE_CANDIDATES)
    }
    assert evaluator.rank_train(tied) == list(evaluator.ELIGIBLE_CANDIDATES)


def test_predecessor_is_enforced_before_any_source_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing-train.json"
    monkeypatch.setitem(evaluator.OUTPUTS, "train", missing)
    opened = False

    def forbidden_loader(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("prices/funding opened before predecessor validation")

    monkeypatch.setattr(evaluator, "load_sources", forbidden_loader)
    with pytest.raises(RuntimeError, match="missing HVSPR-8 predecessor"):
        evaluator.run("test", tmp_path / "test.json")
    assert opened is False

    _manifest(missing, passed=False)
    with pytest.raises(RuntimeError, match="predecessor did not pass"):
        evaluator.run("test", tmp_path / "test.json")
    assert opened is False


def test_source_loader_opens_only_requested_stage_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    start = pd.Timestamp("2023-07-01", tz="UTC")
    end = pd.Timestamp("2024-01-01", tz="UTC")
    market = pd.DataFrame({"date": [start]})
    funding = pd.DataFrame({"date": [start]})
    calls: list[str] = []
    monkeypatch.setattr(evaluator.engine, "load_csv_market", lambda *_: calls.append("csv_market") or market)
    monkeypatch.setattr(evaluator.engine, "load_train_funding", lambda *_: calls.append("train_funding") or funding)
    monkeypatch.setattr(evaluator, "load_postgres_funding", lambda *_: calls.append("postgres_funding") or funding)
    monkeypatch.setattr(evaluator, "load_postgres_sources", lambda *_: calls.append("postgres_market_and_funding") or (market, funding, {"mode": "postgres"}))

    evaluator.load_sources("train", start, end)
    assert calls == ["csv_market", "train_funding"]
    calls.clear()
    evaluator.load_sources("test", start, end)
    assert calls == ["csv_market", "postgres_funding"]
    calls.clear()
    evaluator.load_sources("eval", start, end)
    assert calls == ["csv_market", "postgres_funding"]
    calls.clear()
    evaluator.load_sources("final", start, end)
    assert calls == ["postgres_market_and_funding"]


def test_strict_mdd_uses_global_hwm_and_favorable_then_adverse() -> None:
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=5)
    market = pd.DataFrame({
        "date": [start, end], "open": [100.0, 100.0], "high": [120.0, 100.0],
        "low": [80.0, 100.0], "close": [100.0, 100.0],
    })
    funding = pd.DataFrame(columns=["date", "funding_rate", "mark_price"])
    clock = pd.DataFrame({"entry_time": [start], "exit_time": [end], "side": [1]})
    report = evaluator.engine.simulate(clock, market, funding, start, end, 0.0)
    assert report["strict_mdd_pct"] == pytest.approx((1.0 - 0.9 / 1.1) * 100)
    assert report["final_equity"] == pytest.approx(1.0)


def test_exact_funding_uses_fixed_quantity_and_settlement_mark() -> None:
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=5)
    market = pd.DataFrame({
        "date": [start, end], "open": [100.0, 100.0], "high": [100.0, 100.0],
        "low": [100.0, 100.0], "close": [100.0, 100.0],
    })
    funding = pd.DataFrame({"date": [start], "funding_rate": [0.01], "mark_price": [200.0]})
    clock = pd.DataFrame({"entry_time": [start], "exit_time": [end], "side": [1]})
    report = evaluator.engine.simulate(clock, market, funding, start, end, 0.0)
    assert report["trade_rows"][0]["funding_cash_over_pre_equity"] == pytest.approx(-0.01)
    assert report["final_equity"] == pytest.approx(0.99)


@pytest.mark.parametrize("winner_p,passed,frozen,decision", [
    (0.01, True, "winner", "pass"),
    (0.03, False, None, "terminal_reject_no_substitution"),
])
def test_train_evaluates_all_four_and_never_substitutes_raw_rank_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winner_p: float,
    passed: bool, frozen: str | None, decision: str,
) -> None:
    start = pd.Timestamp("2023-07-01", tz="UTC")
    market = pd.DataFrame({"date": [start]})
    funding = pd.DataFrame({"date": [start]})
    gross9 = {"manifest_hash": "gross9-manifest"}
    calls: list[str] = []
    metrics = {
        evaluator.ELIGIBLE_CANDIDATES[0]: _primary(ratio=3.1),
        evaluator.ELIGIBLE_CANDIDATES[1]: _primary(ratio=5.0, pvalue=winner_p),
        evaluator.ELIGIBLE_CANDIDATES[2]: _primary(ratio=4.0),
        evaluator.ELIGIBLE_CANDIDATES[3]: _primary(ratio=3.5),
    }
    monkeypatch.setattr(evaluator, "verify", lambda stage: (gross9, None))
    monkeypatch.setattr(evaluator, "load_sources", lambda *_: (market, funding, {"mode": "synthetic"}))
    monkeypatch.setattr(evaluator.engine, "validate_market", lambda *_: None)
    monkeypatch.setattr(evaluator.engine, "validate_funding", lambda *_: None)
    monkeypatch.setattr(evaluator, "_load_clock", lambda candidate, *_: calls.append(candidate) or pd.DataFrame({"candidate": [candidate]}))
    monkeypatch.setattr(evaluator, "evaluate_primary", lambda clock, *_: metrics[clock.iloc[0]["candidate"]])

    report = evaluator.run("train", tmp_path / "train.json")
    winner = evaluator.ELIGIBLE_CANDIDATES[1]
    assert calls == list(evaluator.ELIGIBLE_CANDIDATES)
    assert report["selection"]["raw_train_ranking"] == [winner, evaluator.ELIGIBLE_CANDIDATES[2], evaluator.ELIGIBLE_CANDIDATES[3], evaluator.ELIGIBLE_CANDIDATES[0]]
    assert report["selection"]["raw_train_rank_one"] == winner
    assert report["passed"] is passed
    assert report["frozen_train_winner"] == (winner if frozen else None)
    assert report["decision"] == decision
    assert report["substitution_authorized"] is False
    assert report["later_stage_outcomes_opened"] is False
    assert report["manifest_hash"] == evaluator.canonical_hash({key: value for key, value in report.items() if key != "manifest_hash"})


def test_later_stage_evaluates_only_frozen_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    winner = evaluator.ELIGIBLE_CANDIDATES[2]
    start = pd.Timestamp("2024-01-01", tz="UTC")
    market = pd.DataFrame({"date": [start]})
    funding = pd.DataFrame({"date": [start]})
    predecessor_path = tmp_path / "train.json"
    predecessor_path.write_text("predecessor", encoding="utf-8")
    monkeypatch.setitem(evaluator.OUTPUTS, "train", predecessor_path)
    predecessor = {"frozen_train_winner": winner, "manifest_hash": "train-manifest"}
    calls: list[str] = []
    monkeypatch.setattr(evaluator, "verify", lambda stage: ({"manifest_hash": "gross9"}, predecessor))
    monkeypatch.setattr(evaluator, "load_sources", lambda *_: (market, funding, {"mode": "synthetic"}))
    monkeypatch.setattr(evaluator.engine, "validate_market", lambda *_: None)
    monkeypatch.setattr(evaluator.engine, "validate_funding", lambda *_: None)
    monkeypatch.setattr(evaluator, "_load_clock", lambda candidate, *_: calls.append(candidate) or pd.DataFrame({"candidate": [candidate]}))
    monkeypatch.setattr(evaluator, "evaluate_primary", lambda *_: _primary())
    report = evaluator.run("test", tmp_path / "test.json")
    assert calls == [winner]
    assert report["candidate"] == winner
    assert report["frozen_train_winner"] == winner
    assert report["selection"]["raw_train_ranking"] is None
    assert report["passed"] is True
