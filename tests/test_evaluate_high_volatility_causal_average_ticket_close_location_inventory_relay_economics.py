from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_high_volatility_causal_average_ticket_close_location_inventory_relay_economics as evaluator


def _primary(
    ratio: float = 3.1,
    absolute_return: float = 10.0,
    pvalue: float = 0.10,
) -> dict:
    return {
        "base": {
            "absolute_return_pct": absolute_return,
            "cagr_to_strict_mdd": ratio,
            "strict_mdd_pct": 15.0,
            "mean_gross_underlying_bp": 20.0,
        },
        "stress": {"absolute_return_pct": 1.0, "cagr_to_strict_mdd": 2.5},
        "cluster_signflip": {"pvalue": pvalue},
        "calendar_halves": {
            "first": {"absolute_return_pct": 1.0},
            "second": {"absolute_return_pct": 1.0},
        },
    }


def _manifest(path: Path, **overrides) -> dict:
    core = {
        "policy_id": evaluator.POLICY_ID,
        "candidate": evaluator.POLICY_ID,
        "stage": "train",
        "passed": True,
        "frozen_train_winner": evaluator.POLICY_ID,
        "substitution_authorized": False,
        "predecessor": None,
        **overrides,
    }
    report = {**core, "manifest_hash": evaluator.canonical_hash(core)}
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _stub_outcomes(monkeypatch: pytest.MonkeyPatch, primary: dict) -> list[str]:
    start = pd.Timestamp("2023-07-01", tz="UTC")
    market = pd.DataFrame({"date": [start]})
    funding = pd.DataFrame({"date": [start]})
    calls: list[str] = []
    monkeypatch.setattr(evaluator, "verify", lambda stage: ({"manifest_hash": "gross9"}, None))
    monkeypatch.setattr(evaluator, "load_sources", lambda *_: (market, funding, {"mode": "synthetic"}))
    monkeypatch.setattr(evaluator.engine, "validate_market", lambda *_: None)
    monkeypatch.setattr(evaluator.engine, "validate_funding", lambda *_: None)
    monkeypatch.setattr(
        evaluator,
        "_load_clock",
        lambda *_: calls.append(evaluator.POLICY_ID) or pd.DataFrame({"candidate": [evaluator.POLICY_ID]}),
    )
    monkeypatch.setattr(evaluator, "evaluate_primary", lambda *_: primary)
    return calls


def test_frozen_singleton_authorization_and_bindings() -> None:
    assert evaluator.POLICY_ID == "HVCATCLIR-8"
    assert evaluator.LEVERAGE == 0.5
    assert evaluator.BASE_COST == 0.0006
    assert evaluator.STRESS_COST == 0.0010
    assert not hasattr(evaluator, "TRAIN_BONFERRONI_RAW_P_MAX")
    assert not hasattr(evaluator, "rank_train")
    assert evaluator.CLOCK["rows"] == 112
    assert evaluator.sha256(evaluator.PREREG) == evaluator.PREREG_SHA
    assert evaluator.sha256(evaluator.SUPPORT) == evaluator.SUPPORT_SHA
    assert evaluator.sha256(evaluator.GROSS9) == evaluator.GROSS9_SHA
    assert evaluator.sha256(evaluator.CLOCK["path"]) == evaluator.CLOCK["sha256"]

    gross9, predecessor = evaluator.verify("train")
    assert predecessor is None
    assert gross9["policy_id"] == evaluator.POLICY_ID
    assert gross9["advance_to_economic_outcomes"] is True
    assert gross9["evidence_boundary"]["outcomes_opened"] is False


def test_every_stage_uses_only_the_weekly_one_sided_p_gate() -> None:
    for stage in evaluator.STAGES:
        checks = evaluator.economic_checks(stage, _primary())
        assert checks["cluster_signflip_p_max_0_1"] is True
        assert len(checks) == 8
        assert not any("bonferroni" in name for name in checks)
        assert evaluator.economic_checks(stage, _primary(pvalue=0.1000001))["cluster_signflip_p_max_0_1"] is False


@pytest.mark.parametrize(
    ("mutator", "failed_check"),
    [
        (lambda value: value["base"].update(absolute_return_pct=0.0), "absolute_return_positive"),
        (lambda value: value["base"].update(cagr_to_strict_mdd=2.99), "cagr_to_strict_mdd_min_3"),
        (lambda value: value["base"].update(strict_mdd_pct=15.01), "strict_mdd_max_15"),
        (lambda value: value["base"].update(mean_gross_underlying_bp=19.99), "mean_gross_move_min_20bp"),
        (lambda value: value["stress"].update(absolute_return_pct=0.0), "stress_absolute_return_positive"),
        (lambda value: value["stress"].update(cagr_to_strict_mdd=2.49), "stress_cagr_to_strict_mdd_min_2_5"),
        (lambda value: value["calendar_halves"]["second"].update(absolute_return_pct=0.0), "each_calendar_half_positive"),
    ],
)
def test_economic_thresholds_are_strictly_frozen(mutator, failed_check: str) -> None:
    primary = _primary()
    mutator(primary)
    checks = evaluator.economic_checks("train", primary)
    assert checks[failed_check] is False
    assert sum(not passed for passed in checks.values()) == 1


def test_predecessor_is_enforced_before_any_price_or_funding_source_opens(
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
    with pytest.raises(RuntimeError, match="missing HVCATCLIR-8 predecessor"):
        evaluator.run("test", tmp_path / "test.json")
    assert opened is False

    _manifest(missing, passed=False)
    with pytest.raises(RuntimeError, match="predecessor did not pass"):
        evaluator.run("test", tmp_path / "test.json")
    assert opened is False


def test_frozen_authorization_is_verified_before_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def forbidden_loader(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("prices/funding opened before frozen authorization")

    monkeypatch.setattr(evaluator, "load_sources", forbidden_loader)
    original_sha = evaluator.sha256
    monkeypatch.setattr(
        evaluator,
        "sha256",
        lambda path: "drift" if Path(path) == evaluator.PREREG else original_sha(path),
    )
    with pytest.raises(RuntimeError, match="preregistration hash drift"):
        evaluator.run("train", tmp_path / "train.json")
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

    for stage, expected in (
        ("train", ["csv_market", "train_funding"]),
        ("test", ["csv_market", "postgres_funding"]),
        ("eval", ["csv_market", "postgres_funding"]),
        ("final", ["postgres_market_and_funding"]),
    ):
        evaluator.load_sources(stage, start, end)
        assert calls == expected
        calls.clear()


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


@pytest.mark.parametrize(
    ("pvalue", "passed", "frozen", "decision"),
    [
        (0.10, True, evaluator.POLICY_ID, "pass"),
        (0.100001, False, None, "terminal_reject_no_substitution"),
    ],
)
def test_train_evaluates_singleton_once_without_ranking_or_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pvalue: float,
    passed: bool,
    frozen: str | None,
    decision: str,
) -> None:
    calls = _stub_outcomes(monkeypatch, _primary(pvalue=pvalue))
    report = evaluator.run("train", tmp_path / "train.json")
    assert calls == [evaluator.POLICY_ID]
    assert report["candidate"] == evaluator.POLICY_ID
    assert report["selection"]["ranking_performed"] is False
    assert report["selection"]["bonferroni_applied"] is False
    assert report["passed"] is passed
    assert report["frozen_train_winner"] == frozen
    assert report["decision"] == decision
    assert report["substitution_authorized"] is False
    assert report["later_stage_outcomes_opened"] is False


def test_later_stage_uses_predecessor_chain_and_same_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor_path = tmp_path / "train.json"
    predecessor_path.write_text("predecessor", encoding="utf-8")
    monkeypatch.setitem(evaluator.OUTPUTS, "train", predecessor_path)
    predecessor = {"frozen_train_winner": evaluator.POLICY_ID, "manifest_hash": "train-manifest"}
    calls = _stub_outcomes(monkeypatch, _primary())
    monkeypatch.setattr(evaluator, "verify", lambda stage: ({"manifest_hash": "gross9"}, predecessor))

    report = evaluator.run("test", tmp_path / "test.json")
    assert calls == [evaluator.POLICY_ID]
    assert report["candidate"] == evaluator.POLICY_ID
    assert report["predecessor"] == {
        "stage": "train",
        "path": predecessor_path.as_posix(),
        "sha256": evaluator.sha256(predecessor_path),
        "manifest_hash": "train-manifest",
    }
    assert report["frozen_train_winner"] == evaluator.POLICY_ID
    assert report["passed"] is True
