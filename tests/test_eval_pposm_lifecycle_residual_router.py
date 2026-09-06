from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from training import eval_pposm_lifecycle_residual_router as ev
from training.search_inventory_purge_reclaim_alpha import Trade


def _score_row(signal: int, candidate: str, margin: float) -> dict[str, Any]:
    return {
        "identity": ev.lifecycle.lifecycle_identity(ev.TRAIN_WINDOW[0], signal, candidate),
        "base_identity": ev.counterfactual.signal_identity(ev.TRAIN_WINDOW[0], signal),
        "candidate_action": candidate,
        "split": "train",
        "window": "pre_2024",
        "date": "2021-01-01 00:00:00",
        "signal_time": "2021-01-01 00:00:00",
        "signal_position": signal,
        "scores": {"KEEP": 0.0, "SWITCH": margin},
        "switch_margin": margin,
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_sha256": "a" * 64,
        "score_normalization": "mean",
        "source_jsonl_sha256": "b" * 64,
        "source_identity_sha256": "c" * 64,
    }


def _pair_scores(margins: dict[int, tuple[float, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal, (skip, tp12) in margins.items():
        rows.append(_score_row(signal, "SKIP", skip))
        rows.append(_score_row(signal, "TP12", tp12))
    return rows


def _trade(signal: int, route: str) -> Trade:
    code = {"TP4": 4, "TP12": 12}.get(route, 0)
    return Trade(
        signal_position=int(signal),
        entry_position=int(signal) * 10 + code,
        exit_position=int(signal) * 10 + code + 1,
        side=1,
        gross_return=0.0,
        price_factor=1.0,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=1.0,
        adverse_price_factor=1.0,
        entry_date="2021-01-01",
    )


@dataclass(frozen=True)
class _Cfg:
    leverage: float = 1.0


class _Engine:
    market = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=2000, freq="5min")})


def _install_route_sensitive_economics(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_apply_routes(engine, signals, routes, *, start, end, spec):  # noqa: ANN001
        return tuple(_trade(signal, route) for signal, route in zip(signals, routes, strict=True) if route != "SKIP")

    def fake_economics(trades, *, start, end, cfg):  # noqa: ANN001
        tp12 = sum(1 for trade in trades if trade.entry_position % 100 in {12})
        total = len(trades)
        # Each TP12 replacement lifts ratios/log-equity, but excessive TP12 later
        # worsens MDD in a way the ranking/gates can observe.
        ret = 100.0 + tp12 * 1.0
        ratio = 3.0 + tp12 * 0.10
        stress_ratio = 2.8 + tp12 * 0.08
        mdd = 10.0 + max(0, tp12 - 30) * 0.001
        return {
            "base_6bp": {
                "absolute_return_pct": ret,
                "cagr_to_strict_mdd": ratio,
                "strict_mdd_pct": mdd,
                "trades": total,
            },
            "stress_10bp": {
                "absolute_return_pct": ret - 1.0 + tp12 * 0.5,
                "cagr_to_strict_mdd": stress_ratio,
                "strict_mdd_pct": mdd,
                "trades": total,
            },
        }

    monkeypatch.setattr(ev, "_apply_routes", fake_apply_routes)
    monkeypatch.setattr(ev, "_economics", fake_economics)


def test_validate_score_rows_requires_exact_204_train_anchor_identity_hash() -> None:
    rows = _pair_scores({signal: (0.1, 0.2) for signal in range(102)})
    expected = hashlib.sha256("\n".join(row["identity"] for row in rows).encode()).hexdigest()
    validation = ev.validate_score_rows(rows, expected_identity_sha256=expected)
    assert validation["rows"] == 204
    assert validation["anchors"] == 102
    assert validation["identity_sha256"] == expected
    assert validation["score_normalization"] == "mean"

    bad = list(rows)
    bad[0] = {**bad[0], "window": "test_2024"}
    with pytest.raises(ValueError, match="train/pre_2024"):
        ev.validate_score_rows(bad, expected_identity_sha256=expected)

    with pytest.raises(ValueError, match="identity hash mismatch"):
        ev.validate_score_rows(rows, expected_identity_sha256="0" * 64)


def test_evaluate_thresholds_returns_no_feasible_when_materiality_or_economics_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def flat_economics(trades, *, start, end, cfg):  # noqa: ANN001
        return {
            "base_6bp": {"absolute_return_pct": 100.0, "cagr_to_strict_mdd": 3.0, "strict_mdd_pct": 10.0, "trades": len(trades)},
            "stress_10bp": {"absolute_return_pct": 99.0, "cagr_to_strict_mdd": 2.8, "strict_mdd_pct": 10.0, "trades": len(trades)},
        }

    monkeypatch.setattr(
        ev,
        "_apply_routes",
        lambda engine, signals, routes, *, start, end, spec: tuple(_trade(signal, route) for signal, route in zip(signals, routes, strict=True) if route != "SKIP"),
    )
    monkeypatch.setattr(ev, "_economics", flat_economics)
    rows = _pair_scores({signal: (1.0, 1.0) for signal in range(12)})
    result = ev.evaluate_thresholds(
        score_rows=rows,
        full_signals=tuple(range(100)),
        engine=_Engine(),
        strategy_cfg=_Cfg(),
        manifest={"spec": {"side": 1, "hold_bars": 1, "capitulation_take_bps": 400, "normal_take_bps": 1200, "stop_bps": 100}},
    )
    assert result["status"] == "no_feasible_train_threshold"
    assert all(not item["feasibility"]["passes"] for item in result["evaluations"])


def test_evaluate_thresholds_ranks_feasible_by_min_ratio_then_log_equity_then_higher_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_route_sensitive_economics(monkeypatch)
    # threshold below 0.2 selects 50 TP12 routes; threshold 0.2 selects 20 TP12
    # routes.  Both pass materiality; the former wins by larger ratio/log-equity.
    rows = _pair_scores(
        {
            **{signal: (-1.0, 0.2) for signal in range(20)},
            **{signal: (-1.0, 0.1) for signal in range(20, 50)},
        }
    )
    result = ev.evaluate_thresholds(
        score_rows=rows,
        full_signals=tuple(range(100)),
        engine=_Engine(),
        strategy_cfg=_Cfg(),
        manifest={"spec": {"side": 1, "hold_bars": 1, "capitulation_take_bps": 400, "normal_take_bps": 1200, "stop_bps": 100}},
    )
    assert result["status"] == "feasible_train_threshold"
    selected = result["selected"]
    assert selected["threshold"] == pytest.approx(-1.0)
    assert selected["feasibility"]["materiality"]["non_default_counts"] == {"SKIP": 0, "TP12": 50}


def test_route_tie_prefers_skip_and_strict_threshold() -> None:
    rows = _pair_scores({1: (0.5, 0.5), 2: (0.5, 0.6)})
    assert ev.route_stream_for_threshold(rows, (0, 1, 2), 0.5) == ("TP4", "TP4", "TP12")
    assert ev.route_stream_for_threshold(rows, (0, 1, 2), 0.49) == ("TP4", "SKIP", "TP12")


def test_run_writes_structured_failure_and_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = _pair_scores({signal: (0.1, 0.1) for signal in range(102)})
    expected = hashlib.sha256("\n".join(row["identity"] for row in rows).encode()).hexdigest()
    score_path = tmp_path / "scores.jsonl"
    data_path = tmp_path / "train.jsonl"
    summary_path = tmp_path / "summary.json"
    score_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    data_rows = [
        {
            "split": "train",
            "metadata": {
                "identity": row["identity"],
                "base_identity": row["base_identity"],
                "candidate_action": row["candidate_action"],
                "signal_position": row["signal_position"],
                "signal_time": row["signal_time"],
                "window": row["window"],
            },
        }
        for row in rows
    ]
    data_path.write_text("\n".join(json.dumps(row) for row in data_rows) + "\n")
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    for row in rows:
        row["source_jsonl_sha256"] = data_sha
        row["source_identity_sha256"] = expected
    output_dir = tmp_path / "rlvr"
    output_dir.mkdir()
    adapter_path = output_dir / "adapter_model.safetensors"
    adapter_path.write_bytes(b"adapter")
    adapter_sha = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    for row in rows:
        row["adapter_sha256"] = adapter_sha
    score_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary_path.write_text(
        json.dumps(
            {
                "identity_sha256": expected,
                "output_sha256": {"train": data_sha},
                "rows": {"train": 204, "oos": 0},
                "reference_anchors": 102,
                "reference_anchor_pairs": 204,
                "manifest_freeze_hash": "freeze",
            }
        )
    )
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(
        json.dumps(
            {
                "protocol_version": "pposm_lifecycle_anchor_residual_sft_rlvr_v1",
                "status": "preregistered_before_lifecycle_sft_and_oos",
                "base_model": {"name": "Qwen/Qwen2.5-1.5B-Instruct"},
                "architecture": {"name": "test"},
                "source": {
                    "manifest_freeze_hash": "freeze",
                    "train_data": {
                        "path": str(data_path),
                        "sha256": data_sha,
                        "identity_sha256": expected,
                        "rows": 204,
                        "anchors": 102,
                    },
                    "train_summary": {
                        "path": str(summary_path),
                        "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    },
                },
            }
        )
    )
    rlvr_config_path = tmp_path / "rlvr_config.json"
    rlvr_config_path.write_text(
        json.dumps(
            {
                "dry_run": False,
                "config": {
                    "label_schema": "pposm_residual_utility",
                    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "train_jsonl": str(data_path),
                    "output_dir": str(output_dir),
                    "sft_adapter_dir": str(tmp_path / "sft"),
                },
            }
        )
    )

    monkeypatch.setattr(ev, "validate_train_data_identity", lambda train_rows, score_rows, expected_identity_sha256: {"train_data_rows": len(train_rows), "identity_sha256": expected_identity_sha256})
    monkeypatch.setattr(ev.lifecycle.frozen, "load_frozen_manifest", lambda manifest: ({"freeze_hash": "freeze", "spec": {"side": 1, "hold_bars": 1, "capitulation_take_bps": 400, "normal_take_bps": 1200, "stop_bps": 100}}, _Cfg()))
    monkeypatch.setattr(ev, "_execution_config", lambda cfg, leverage: cfg)
    monkeypatch.setattr(ev.lifecycle, "load_train_context", lambda manifest, cfg: (_Engine.market, pd.DataFrame(), pd.DataFrame(), [True] * len(_Engine.market), _Engine()))
    monkeypatch.setattr(ev, "_full_pre2024_signals", lambda market, active: tuple(range(462)))
    monkeypatch.setattr(ev, "evaluate_thresholds", lambda **kwargs: {"status": "no_feasible_train_threshold", "candidate_thresholds": 1, "control_economics": {}, "evaluations": []})

    failure_path = tmp_path / "failure.json"
    with pytest.raises(SystemExit) as exc:
        ev.run(
            ev.Config(
                manifest=tmp_path / "manifest.json",
                train_data=data_path,
                data_summary=summary_path,
                train_scores=score_path,
                threshold_output=tmp_path / "threshold.json",
                failure_output=failure_path,
                preregistration=prereg_path,
                rlvr_config=rlvr_config_path,
                expected_identity_sha256=expected,
            )
        )
    assert exc.value.code == 2
    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "no_feasible_train_threshold"
    assert failure["selection_boundary"] == "pre_2024_train_only_no_oos_inputs"
    assert not (tmp_path / "threshold.json").exists()


def test_validate_data_summary_fails_closed_on_missing_bindings() -> None:
    with pytest.raises(ValueError, match="does not bind"):
        ev.validate_data_summary(
            {},
            expected_identity_sha256="a" * 64,
            train_data_sha256="b" * 64,
        )


def test_validate_score_rows_rejects_non_hex_adapter_hash() -> None:
    rows = _pair_scores({signal: (0.1, 0.2) for signal in range(102)})
    rows[0]["adapter_sha256"] = "x" * 64
    expected = hashlib.sha256(
        "\n".join(row["identity"] for row in rows).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="adapter_sha256"):
        ev.validate_score_rows(rows, expected_identity_sha256=expected)
