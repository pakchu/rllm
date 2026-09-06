from __future__ import annotations

import json
from pathlib import Path

from training.assemble_pposm_conditional_residual_result import Config, assemble, sha256_file


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_assemble_records_hashes_train_freeze_and_replay(tmp_path: Path) -> None:
    prereg = _write(
        tmp_path / "prereg.json",
        {
            "status": "preregistered_before_model_scoring",
            "source": {"manifest_freeze_hash": "frozen"},
            "architecture": {"name": "residual"},
            "base_model": {"name": "qwen"},
        },
    )
    data = _write(tmp_path / "data.json", {"manifest_freeze_hash": "frozen"})
    threshold = _write(
        tmp_path / "threshold.json",
        {
            "protocol": "pposm_residual_train_only_threshold_v2",
            "train_base_signals": 20,
            "threshold": 0.2,
            "selection_inputs": {"train_scores": {"sha256": "abc"}},
            "future_can_rank_repair_or_reselect": False,
        },
    )
    route = _write(tmp_path / "route.json", {"route_counts": {"TP4": 15, "SKIP": 5}})
    backtest = _write(
        tmp_path / "backtest.json",
        {"invariants": {"entry_rule": "exact_next_5m_open"}},
    )
    replay = tmp_path / "replay.json"
    replay.write_bytes(backtest.read_bytes())
    generic = _write(tmp_path / "generic.json", {"ok": True})
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    scores = tmp_path / "scores.jsonl"
    scores.write_text('{"score":1}\n', encoding="utf-8")
    threshold_payload = json.loads(threshold.read_text())
    threshold_payload["selection_inputs"]["train_scores"]["sha256"] = sha256_file(
        scores
    )
    _write(threshold, threshold_payload)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"prediction":"TP4"}\n', encoding="utf-8")
    output = tmp_path / "result.json"

    result = assemble(
        Config(
            preregistration=prereg,
            data_summary=data,
            sft_summary=generic,
            sft_adapter=adapter,
            rlvr_config=generic,
            rlvr_reward_diagnostics=generic,
            rlvr_gradient_diagnostics=generic,
            rlvr_adapter=adapter,
            train_scores=scores,
            threshold=threshold,
            oos_scores=scores,
            route_predictions=predictions,
            route_report=route,
            backtest=backtest,
            replay_snapshot=replay,
            output=output,
        )
    )

    assert result["checks"] == {
        "preregistration_frozen_before_scoring": True,
        "data_manifest_matches_preregistration": True,
        "threshold_train_only": True,
        "oos_rerank_or_repair": False,
        "artifact_hashes_recorded": True,
        "byte_replay_identical": True,
    }
    assert result["artifacts"]["backtest"]["sha256"] == sha256_file(backtest)
    assert len(result["result_hash"]) == 64
    assert json.loads(output.read_text())["result_hash"] == result["result_hash"]
