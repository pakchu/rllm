from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training import (
    evaluate_high_volatility_multi_condition_pairwise_concordance_economics as evaluator,
)


def _passing_primary(pvalue: float = 0.01) -> dict:
    base = {
        "absolute_return_pct": 10.0,
        "cagr_to_strict_mdd": 3.1,
        "strict_mdd_pct": 5.0,
        "mean_gross_underlying_bp": 21.0,
    }
    stress = {"absolute_return_pct": 1.0, "cagr_to_strict_mdd": 2.6}
    return {
        "base": base,
        "stress": stress,
        "cluster_signflip": {"pvalue": pvalue},
        "calendar_halves": {
            "first": {"absolute_return_pct": 1.0},
            "second": {"absolute_return_pct": 1.0},
        },
    }


def _manifest(path: Path, **overrides) -> dict:
    core = {
        "policy_id": evaluator.POLICY_ID,
        "candidate": evaluator.PAIR,
        "stage": "train",
        "passed": True,
        "frozen_train_winner": evaluator.PAIR,
        "substitution_authorized": False,
        **overrides,
    }
    report = {**core, "manifest_hash": evaluator.canonical_hash(core)}
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def test_frozen_bindings_and_sole_pair_authorization() -> None:
    assert evaluator.POLICY_ID == "HVMCPAC-8"
    assert evaluator.PAIR == "CARSC-8__AND__HVTFR-8"
    assert evaluator.LEVERAGE == 0.5
    assert evaluator.BASE_COST == 0.0006
    assert evaluator.STRESS_COST == 0.0010
    assert evaluator.TRAIN_BONFERRONI_RAW_P_MAX == pytest.approx(0.1 / 6)
    assert evaluator.sha256(evaluator.PREREG) == evaluator.PREREG_SHA
    assert evaluator.sha256(evaluator.SUPPORT) == evaluator.SUPPORT_SHA
    assert evaluator.sha256(evaluator.GROSS9) == evaluator.GROSS9_SHA
    assert evaluator.sha256(evaluator.PAIR_CLOCK) == evaluator.PAIR_CLOCK_SHA

    gross9, predecessor = evaluator.verify("train")
    assert predecessor is None
    assert gross9["candidate"] == evaluator.PAIR
    assert gross9["advance_to_economic_outcomes"] is True
    assert gross9["evidence_boundary"]["outcomes_opened"] is False


def test_train_bonferroni_is_additional_to_all_normal_gates() -> None:
    normal_only = evaluator.economic_checks("test", _passing_primary(0.02))
    train = evaluator.economic_checks("train", _passing_primary(0.02))
    assert all(normal_only.values())
    assert train["cluster_signflip_p_max_0_1"] is True
    assert train["train_bonferroni_raw_weekly_p_max_0_1_over_6"] is False

    passing_train = evaluator.economic_checks("train", _passing_primary(0.1 / 6))
    assert all(passing_train.values())


def test_predecessor_is_required_and_validated_before_any_source_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing-train.json"
    monkeypatch.setitem(evaluator.OUTPUTS, "train", missing)
    opened = False

    def forbidden_loader(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("prices/funding opened before predecessor validation")

    monkeypatch.setattr(evaluator, "load_sources", forbidden_loader)
    with pytest.raises(RuntimeError, match="missing HVMCPAC-8 predecessor"):
        evaluator.run("test", tmp_path / "test.json")
    assert opened is False

    _manifest(missing, passed=False)
    with pytest.raises(RuntimeError, match="predecessor did not pass"):
        evaluator.run("test", tmp_path / "test.json")
    assert opened is False


def test_source_loader_opens_only_requested_stage_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = pd.Timestamp("2023-07-01", tz="UTC")
    end = pd.Timestamp("2024-01-01", tz="UTC")
    market = pd.DataFrame({"date": [start]})
    funding = pd.DataFrame({"date": [start]})
    calls: list[str] = []

    monkeypatch.setattr(
        evaluator.engine,
        "load_csv_market",
        lambda *_: calls.append("csv_market") or market,
    )
    monkeypatch.setattr(
        evaluator.engine,
        "load_train_funding",
        lambda *_: calls.append("train_funding") or funding,
    )
    monkeypatch.setattr(
        evaluator,
        "load_postgres_funding",
        lambda *_: calls.append("postgres_funding") or funding,
    )
    monkeypatch.setattr(
        evaluator,
        "load_postgres_sources",
        lambda *_: calls.append("postgres_market_and_funding")
        or (market, funding, {"mode": "postgres"}),
    )

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
    market = pd.DataFrame(
        {
            "date": [start, end],
            "open": [100.0, 100.0],
            "high": [120.0, 100.0],
            "low": [80.0, 100.0],
            "close": [100.0, 100.0],
        }
    )
    funding = pd.DataFrame(columns=["date", "funding_rate", "mark_price"])
    clock = pd.DataFrame(
        {"entry_time": [start], "exit_time": [end], "side": [1]}
    )

    report = evaluator.engine.simulate(clock, market, funding, start, end, 0.0)

    # Fixed 0.5 gross exposure makes favorable equity 1.10, then adverse 0.90.
    assert report["strict_mdd_pct"] == pytest.approx((1.0 - 0.9 / 1.1) * 100)
    assert report["final_equity"] == pytest.approx(1.0)


def test_exact_held_funding_uses_fixed_quantity_and_settlement_mark() -> None:
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=5)
    market = pd.DataFrame(
        {
            "date": [start, end],
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
        }
    )
    funding = pd.DataFrame(
        {"date": [start], "funding_rate": [0.01], "mark_price": [200.0]}
    )
    clock = pd.DataFrame(
        {"entry_time": [start], "exit_time": [end], "side": [1]}
    )

    report = evaluator.engine.simulate(clock, market, funding, start, end, 0.0)

    # Quantity is +0.5/100=.005 and cash funding is -.005*200*.01=-.01.
    assert report["trade_rows"][0]["funding_cash_over_pre_equity"] == pytest.approx(
        -0.01
    )
    assert report["final_equity"] == pytest.approx(0.99)


@pytest.mark.parametrize(
    ("pvalue", "passed", "winner", "decision"),
    [
        (0.01, True, evaluator.PAIR, "pass"),
        (0.02, False, None, "terminal_reject_no_substitution"),
    ],
)
def test_train_freezes_sole_pair_if_and_only_if_every_gate_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pvalue: float,
    passed: bool,
    winner: str | None,
    decision: str,
) -> None:
    start = pd.Timestamp("2023-07-01", tz="UTC")
    market = pd.DataFrame({"date": [start]})
    funding = pd.DataFrame({"date": [start]})
    clock = pd.DataFrame(
        {"entry_time": [start], "exit_time": [start], "side": [1]}
    )
    gross9 = {"manifest_hash": "gross9-manifest"}
    monkeypatch.setattr(evaluator, "verify", lambda stage: (gross9, None))
    monkeypatch.setattr(
        evaluator,
        "load_sources",
        lambda *_: (market, funding, {"mode": "synthetic"}),
    )
    monkeypatch.setattr(evaluator.engine, "validate_market", lambda *_: None)
    monkeypatch.setattr(evaluator.engine, "validate_funding", lambda *_: None)
    monkeypatch.setattr(evaluator.legacy, "load_clock", lambda *_: clock)
    monkeypatch.setattr(
        evaluator, "evaluate_primary", lambda *_: _passing_primary(pvalue)
    )

    report = evaluator.run("train", tmp_path / "train.json")

    assert report["passed"] is passed
    assert report["frozen_train_winner"] == winner
    assert report["decision"] == decision
    assert report["selection"]["raw_train_rank_one"] == evaluator.PAIR
    assert report["selection"]["substitution_authorized"] is False
    assert report["later_stage_outcomes_opened"] is False
    assert report["manifest_hash"] == evaluator.canonical_hash(
        {key: value for key, value in report.items() if key != "manifest_hash"}
    )
