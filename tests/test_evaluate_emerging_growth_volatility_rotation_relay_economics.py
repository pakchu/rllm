import json
import math
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_emerging_growth_volatility_rotation_relay_economics as economics


def test_exact_offset_funding_posts_to_containing_bar():
    start = pd.Timestamp("2023-07-01T00:00:00Z")
    end = start + pd.Timedelta(minutes=10)
    dates = pd.date_range(start, end, freq="5min", inclusive="both")
    market = pd.DataFrame(
        {"date": dates, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}
    )
    funding = pd.DataFrame(
        {
            "date": [start + pd.Timedelta(milliseconds=5), end],
            "funding_rate": [0.01, 0.5],
            "mark_price": [100.0, 100.0],
        }
    )
    clock = pd.DataFrame({"entry_time": [start], "exit_time": [end], "side": [1]})
    report = economics.engine.simulate(clock, market, funding, start, end, cost=0.0)
    assert math.isclose(report["final_equity"], 0.995, abs_tol=1e-12)


def test_frozen_accounting_and_stage_contract():
    assert economics.POLICY_ID == "EGVRR-12"
    assert economics.LEVERAGE == 0.5
    assert economics.BASE_COST == 0.0006
    assert economics.STRESS_COST == 0.001
    assert economics.CLOCK_ROWS == 100
    assert tuple(economics.STAGES) == ("train", "test", "eval", "final")
    assert economics.STAGES["final"][2] == "2026-08-01T00:00:00Z"
    assert economics.FREEZE == Path(
        "results/emerging_growth_volatility_rotation_relay_economic_evaluator_freeze_2026-08-10.json"
    )
    assert economics.CONTROLS == (
        "no_btc_variation_gate",
        "no_relative_change_tail",
        "vxeem_minus_vxn_raw",
        "one_session_stale_relative_change",
        "direction_flip",
        "forced_long",
    )
    source = Path(economics.__file__).read_text()
    assert "full calendar including idle time" in source
    assert "global peak, every held favorable then adverse" in source
    assert "rv20" not in source.lower()


def test_frozen_predecessor_artifact_hashes_are_exact():
    assert economics.PREREG_SHA == (
        "a436b758465bc80c8dae99c104d5f64f97f16d08680a047b37a5d5a6c960d897"
    )
    assert economics.SUPPORT_SHA == (
        "23c61c47ff90a9ffcbc648e0729ad596a460b1e31b5807c95c5ef8101fb34c3b"
    )
    assert economics.CLOCK_SHA == (
        "4f8548b294d8c5ef154f01ac06aa98a0f810d9e13103aae7b68cf47304b8af4e"
    )
    assert economics.NOVELTY_SHA == (
        "a5a9b0d4d19d43ae67c3a518f4879fdc54509dae983d526dd964494ec0effc76"
    )
    for path, expected in (
        (economics.PREREG, economics.PREREG_SHA),
        (economics.SUPPORT, economics.SUPPORT_SHA),
        (economics.CLOCK, economics.CLOCK_SHA),
        (economics.NOVELTY, economics.NOVELTY_SHA),
    ):
        assert economics.sha256(path) == expected


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, allow_nan=False) + "\n")


def _bind_verify_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    prereg = tmp_path / "prereg.json"
    support = tmp_path / "support.json"
    novelty = tmp_path / "novelty.json"
    clock = tmp_path / "clock.csv.gz"
    freeze = tmp_path / "freeze.json"
    _write_json(prereg, {"policy_id": "EGVRR-12"})
    clock.write_bytes(b"frozen-clock")
    _write_json(
        support,
        {
            "policy_id": "EGVRR-12",
            "clock": {"sha256": economics.sha256(clock), "rows": economics.CLOCK_ROWS},
        },
    )
    _write_json(
        novelty,
        {
            "advance_to_economic_outcomes": True,
            "evidence_boundary": {"outcomes_opened": False},
            "manifest_hash": "novelty-manifest",
        },
    )
    freeze_core = {
        "evaluator": {"sha256": economics.sha256(Path(economics.__file__))},
        "outcomes_opened": False,
    }
    frozen = {**freeze_core, "manifest_hash": economics.canonical_hash(freeze_core)}
    _write_json(freeze, frozen)
    for name, path in (
        ("PREREG", prereg),
        ("SUPPORT", support),
        ("NOVELTY", novelty),
        ("CLOCK", clock),
        ("FREEZE", freeze),
    ):
        monkeypatch.setattr(economics, name, path)
    monkeypatch.setattr(economics, "PREREG_SHA", economics.sha256(prereg))
    monkeypatch.setattr(economics, "SUPPORT_SHA", economics.sha256(support))
    monkeypatch.setattr(economics, "NOVELTY_SHA", economics.sha256(novelty))
    monkeypatch.setattr(economics, "CLOCK_SHA", economics.sha256(clock))
    return frozen


def test_verify_requires_outcome_blind_freeze_and_passed_novelty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    frozen = _bind_verify_inputs(monkeypatch, tmp_path)
    novelty, freeze = economics.verify("train")
    assert novelty["advance_to_economic_outcomes"] is True
    assert freeze == frozen

    novelty_payload = json.loads(economics.NOVELTY.read_text())
    novelty_payload["evidence_boundary"]["outcomes_opened"] = True
    _write_json(economics.NOVELTY, novelty_payload)
    monkeypatch.setattr(economics, "NOVELTY_SHA", economics.sha256(economics.NOVELTY))
    with pytest.raises(RuntimeError, match="novelty did not authorize economics"):
        economics.verify("train")


def test_verify_enforces_sequential_first_failure_and_replay_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _bind_verify_inputs(monkeypatch, tmp_path)
    outputs = {stage: tmp_path / f"{stage}.json" for stage in economics.STAGES}
    monkeypatch.setattr(economics, "OUTPUTS", outputs)

    with pytest.raises(RuntimeError, match="missing predecessor"):
        economics.verify("test")

    predecessor_core = {"stage": "train", "passed": True}
    predecessor = {
        **predecessor_core,
        "manifest_hash": economics.canonical_hash(predecessor_core),
    }
    _write_json(outputs["train"], predecessor)
    economics.verify("test")

    predecessor["passed"] = False
    _write_json(outputs["train"], predecessor)
    with pytest.raises(RuntimeError, match="predecessor did not pass"):
        economics.verify("test")

    predecessor["passed"] = True
    predecessor["manifest_hash"] = "drift"
    _write_json(outputs["train"], predecessor)
    with pytest.raises(RuntimeError, match="predecessor did not pass"):
        economics.verify("test")


def test_primary_uses_base_stress_weekly_signflip_and_strict_calendar_halves(
    monkeypatch: pytest.MonkeyPatch,
):
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    end = pd.Timestamp("2024-01-03T00:00:00Z")
    midpoint = start + (end - start) / 2
    clock = pd.DataFrame(
        {
            "entry_time": [start, midpoint - pd.Timedelta(hours=1), midpoint],
            "exit_time": [start + pd.Timedelta(hours=6), midpoint + pd.Timedelta(hours=1), midpoint + pd.Timedelta(hours=6)],
            "side": [1, -1, 1],
        }
    )
    calls = []

    def simulate(subset, market, funding, left, right, cost):
        calls.append((subset.copy(), left, right, cost))
        return {
            "absolute_return_pct": 1.0,
            "trade_rows": [{"week": "2024-01-01", "return": 0.01}],
        }

    monkeypatch.setattr(economics.engine, "simulate", simulate)
    monkeypatch.setattr(
        economics.engine,
        "cluster_p",
        lambda rows: {"pvalue": 0.05, "rows": len(rows)},
    )
    result = economics.evaluate_primary(clock, pd.DataFrame(), pd.DataFrame(), start, end)

    assert [call[3] for call in calls] == [0.0006, 0.001, 0.0006, 0.0006]
    assert calls[2][1:3] == (start, midpoint)
    assert calls[3][1:3] == (midpoint, end)
    assert calls[2][0].index.tolist() == [0]
    assert calls[3][0].index.tolist() == [2]
    assert result["cluster_signflip"] == {"pvalue": 0.05, "rows": 1}
    assert "trade_rows" not in result["base"]
    assert "trade_rows" not in result["stress"]
    assert all("trade_rows" not in half for half in result["calendar_halves"].values())


def _primary(*, passed: bool) -> dict:
    return {
        "base": {
            "absolute_return_pct": 1.0 if passed else 0.0,
            "cagr_to_strict_mdd": 3.0,
            "strict_mdd_pct": 15.0,
            "mean_gross_underlying_bp": 20.0,
        },
        "stress": {"absolute_return_pct": 1.0, "cagr_to_strict_mdd": 2.5},
        "cluster_signflip": {"pvalue": 0.1},
        "calendar_halves": {
            "first": {"absolute_return_pct": 0.1},
            "second": {"absolute_return_pct": 0.1},
        },
    }


@pytest.mark.parametrize(
    ("stage", "primary_passed", "expected_decision", "advance_next", "advance_audit"),
    [
        ("train", False, "terminal_reject_no_repair", False, False),
        ("train", True, "pass", True, False),
        ("final", True, "pass_to_post_stage_volatility_audit", False, True),
    ],
)
def test_run_is_manifest_bound_and_opens_post_stage_audit_only_after_final_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
    primary_passed: bool,
    expected_decision: str,
    advance_next: bool,
    advance_audit: bool,
):
    freeze = tmp_path / "freeze.json"
    freeze_payload = {"manifest_hash": "freeze-manifest"}
    _write_json(freeze, freeze_payload)
    novelty = tmp_path / "novelty.json"
    novelty_payload = {"manifest_hash": "novelty-manifest"}
    _write_json(novelty, novelty_payload)
    control_dir = tmp_path / "controls"
    control_dir.mkdir()
    controls = {}
    for name in economics.CONTROLS:
        path = control_dir / f"{name}.csv.gz"
        path.write_bytes(name.encode())
        controls[name] = {"sha256": economics.sha256(path)}
    support = tmp_path / "support.json"
    _write_json(support, {"controls": controls})
    outputs = {name: tmp_path / f"{name}.json" for name in economics.STAGES}
    predecessor = outputs["eval"]
    predecessor.write_text("passed predecessor\n")

    monkeypatch.setattr(economics, "FREEZE", freeze)
    monkeypatch.setattr(economics, "NOVELTY", novelty)
    monkeypatch.setattr(economics, "SUPPORT", support)
    monkeypatch.setattr(economics, "CONTROL_DIR", control_dir)
    monkeypatch.setattr(economics, "OUTPUTS", outputs)
    monkeypatch.setattr(economics, "verify", lambda requested: (novelty_payload, freeze_payload))
    market = pd.DataFrame({"date": [pd.Timestamp("2024-01-01T00:00:00Z")]})
    funding = pd.DataFrame({"date": [pd.Timestamp("2024-01-01T00:00:00Z")]})
    clock = pd.DataFrame(
        {
            "entry_time": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "exit_time": [pd.Timestamp("2024-01-01T06:00:00Z")],
            "side": [1],
        }
    )
    monkeypatch.setattr(
        economics,
        "load_sources",
        lambda requested, start, end: (market, funding, {"mode": "test"}),
    )
    monkeypatch.setattr(economics.engine, "validate_market", lambda *args: None)
    monkeypatch.setattr(economics.engine, "validate_funding", lambda *args: None)
    monkeypatch.setattr(economics.legacy, "load_clock", lambda *args: clock)
    monkeypatch.setattr(economics, "evaluate_primary", lambda *args: _primary(passed=primary_passed))
    monkeypatch.setattr(
        economics,
        "evaluate_control",
        lambda *args: {"base": {"absolute_return_pct": 0.0}, "stress": {"absolute_return_pct": 0.0}},
    )
    destination = tmp_path / f"written-{stage}-{primary_passed}.json"

    report = economics.run(stage, destination)
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert json.loads(destination.read_text()) == report
    assert report["manifest_hash"] == economics.canonical_hash(core)
    assert report["decision"] == expected_decision
    assert report["advance_to_next_stage"] is advance_next
    assert report["advance_to_post_stage_volatility_audit"] is advance_audit
    assert report["later_stage_outcomes_opened"] is False
    assert set(report["controls_diagnostic_only"]) == set(economics.CONTROLS)
    if stage == "final":
        assert report["predecessor"] == {
            "stage": "eval",
            "path": str(predecessor),
            "sha256": economics.sha256(predecessor),
        }
    else:
        assert report["predecessor"] is None
