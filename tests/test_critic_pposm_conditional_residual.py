from __future__ import annotations

import json
from pathlib import Path

from training.assemble_pposm_conditional_residual_result import Config, assemble
from training.critic_pposm_conditional_residual import evaluate


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _economics(
    ret: float, ratio: float, mdd: float, trades: int = 50, cost: float = 0.0006
) -> dict:
    return {
        "one_side_cost_rate": cost,
        "equity_stats": {
            "absolute_return_pct": ret,
            "cagr_to_strict_mdd": ratio,
            "strict_mdd_pct": mdd,
            "trades": trades,
        },
        "one_sided_utc_week_sign_flip": {"p_value_one_sided": 0.05},
    }


def _backtest() -> dict:
    windows = {}
    for name in ("test_2024", "eval_2025", "holdout_2026"):
        windows[name] = {
            "economics": {
                "baseline": {
                    "base_6bp": _economics(4.0, 3.0, 5.0),
                    "stress_10bp": _economics(3.0, 2.8, 5.0, cost=0.0010),
                },
                "predicted": {
                    "base_6bp": _economics(5.0, 3.2, 5.0),
                    "stress_10bp": _economics(4.0, 3.0, 5.005, cost=0.0010),
                },
            }
        }
    windows["combined_2024_2026_06_02"] = {
        "route_counts": {
            "baseline": {"TP4": 100},
            "predicted": {"TP4": 80, "SKIP": 20},
        },
        "agreement": {"decision_agreement_rate": 0.80},
        "economics": {
            "baseline": {
                "base_6bp": _economics(20.0, 3.00, 5.0, 48),
                "stress_10bp": _economics(18.0, 2.70, 5.0, 48, 0.0010),
            },
            "predicted": {
                "base_6bp": _economics(21.0, 3.06, 5.0, 50),
                "stress_10bp": _economics(19.0, 2.80, 5.005, 50, 0.0010),
            },
        },
    }
    return {
        "manifest_freeze_hash": "frozen",
        "invariants": {
            "baseline": "ALWAYS_TP4",
            "entry_rule": "exact_next_5m_open",
            "lifecycle": "TP_or_48h_cap",
            "funding_applied": True,
            "non_overlapping": True,
            "future_return_used_for_route": False,
        },
        "windows": windows,
    }


def _assembled(tmp_path: Path, *, constant: bool = False) -> Path:
    prereg = _write_json(
        tmp_path / "prereg.json",
        {
            "status": "preregistered_before_model_scoring",
            "source": {"manifest_freeze_hash": "frozen"},
            "architecture": {"name": "pairwise_residual_action_router"},
            "base_model": {"name": "qwen"},
        },
    )
    data = _write_json(
        tmp_path / "data.json",
        {
            "manifest_freeze_hash": "frozen",
            "identity_sha256_by_split": {"train": "train-identities"},
        },
    )
    sft = _write_json(tmp_path / "sft.json", {"rows": 100})
    rlvr = _write_json(
        tmp_path / "rlvr.json",
        {"config": {"label_schema": "pposm_residual_utility"}},
    )
    generic = _write_json(tmp_path / "generic.json", {"ok": True})
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    train_scores = tmp_path / "train_scores.jsonl"
    train_scores.write_text('{"score":1}\n', encoding="utf-8")
    threshold = _write_json(
        tmp_path / "threshold.json",
        {
            "protocol": "pposm_residual_train_only_threshold_v2",
            "train_base_signals": 100,
            "train_pair_identity_sha256": "train-identities",
            "selection_inputs": {
                "train_scores": {
                    "sha256": __import__("hashlib").sha256(train_scores.read_bytes()).hexdigest()
                }
            },
            "future_can_rank_repair_or_reselect": False,
        },
    )
    scores = tmp_path / "oos_scores.jsonl"
    scores.write_text('{"score":1}\n', encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"prediction":"SKIP"}\n', encoding="utf-8")
    route = _write_json(tmp_path / "route.json", {"route_counts": {"TP4": 80, "SKIP": 20}})
    backtest_value = _backtest()
    if constant:
        combined = backtest_value["windows"]["combined_2024_2026_06_02"]
        combined["route_counts"]["predicted"] = {"TP4": 100}
        combined["agreement"]["decision_agreement_rate"] = 1.0
    backtest = _write_json(tmp_path / "backtest.json", backtest_value)
    replay = tmp_path / "replay.json"
    replay.write_bytes(backtest.read_bytes())
    output = tmp_path / "result.json"
    assemble(
        Config(
            preregistration=prereg,
            data_summary=data,
            sft_summary=sft,
            sft_adapter=adapter,
            rlvr_config=rlvr,
            rlvr_reward_diagnostics=generic,
            rlvr_gradient_diagnostics=generic,
            rlvr_adapter=adapter,
            train_scores=train_scores,
            threshold=threshold,
            oos_scores=scores,
            route_predictions=predictions,
            route_report=route,
            backtest=backtest,
            replay_snapshot=replay,
            output=output,
        )
    )
    return output


def test_critic_passes_complete_nonreplaceable_result(tmp_path: Path) -> None:
    result = _assembled(tmp_path)
    report = evaluate(json.loads(result.read_text()), result_path=result)
    assert report["passed"] is True, report["failed_checks"]


def test_critic_rejects_constant_tp4(tmp_path: Path) -> None:
    result = _assembled(tmp_path, constant=True)
    report = evaluate(json.loads(result.read_text()), result_path=result)
    assert report["passed"] is False
    assert {"at_least_two_routes", "difference_rate_ge_10pct"} <= set(
        report["failed_checks"]
    )


def test_critic_rejects_artifact_tampering(tmp_path: Path) -> None:
    result = _assembled(tmp_path)
    (tmp_path / "route.json").write_text("{}\n", encoding="utf-8")
    report = evaluate(json.loads(result.read_text()), result_path=result)
    assert report["passed"] is False
    assert "artifact_hashes_exact" in report["failed_checks"]
