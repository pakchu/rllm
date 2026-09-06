from types import SimpleNamespace
import hashlib
import json

import pytest

from training import eval_pposm_lifecycle_specialist_router as router


def test_validate_preregistration_binds_split_and_specialist_sources(tmp_path):
    specialists = {}
    outputs = {}
    for candidate, marker in (("SKIP", b"skip"), ("TP12", b"tp12")):
        path = tmp_path / f"{candidate}.jsonl"
        path.write_bytes(marker)
        sha = hashlib.sha256(marker).hexdigest()
        specialists[candidate] = {
            "path": str(path),
            "sha256": sha,
            "identity_sha256": candidate.lower() * 8,
            "rows": 102,
            "targets": {"KEEP": 77, "SWITCH": 25},
        }
        outputs[candidate] = {
            "sha256": sha,
            "identity_sha256": candidate.lower() * 8,
            "rows": 102,
            "target_counts": {"KEEP": 77, "SWITCH": 25},
        }
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "identity_sha256": router.DEFAULT_EXPECTED_IDENTITY_SHA256,
                "outputs": outputs,
            }
        )
    )
    prereg = {
        "protocol_version": "pposm_lifecycle_two_specialist_sft_rlvr_v1",
        "status": "preregistered_before_specialist_training_and_oos",
        "base_model": {"name": "Qwen/Qwen2.5-1.5B-Instruct"},
        "source": {
            "manifest_freeze_hash": "freeze",
            "lifecycle_train_identity_sha256": router.DEFAULT_EXPECTED_IDENTITY_SHA256,
            "split_summary": {
                "path": str(split),
                "sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
            },
            "specialists": specialists,
        },
    }
    binding = router.validate_preregistration(prereg)
    assert binding["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"


def _score(signal, candidate, margin, *, signal_time="2023-01-01 00:00:00"):
    return {
        "identity": f"pposm-lifecycle-residual|{candidate}|pre_2024|{signal}",
        "base_identity": f"pposm-counterfactual-action|pre_2024|{signal}",
        "candidate_action": candidate,
        "split": "train",
        "window": "pre_2024",
        "target": "SWITCH" if margin > 0 else "KEEP",
        "signal_time": signal_time,
        "date": signal_time,
        "signal_position": signal,
        "signal_pos": signal,
        "scores": {"KEEP": 0.0, "SWITCH": margin},
        "switch_margin": margin,
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_sha256": ("a" if candidate == "SKIP" else "b") * 64,
        "score_normalization": "mean",
        "source_jsonl_sha256": ("c" if candidate == "SKIP" else "d") * 64,
        "source_identity_sha256": ("e" if candidate == "SKIP" else "f") * 64,
    }


def _train_row(signal, candidate, *, signal_time="2023-01-01 00:00:00"):
    return {
        "split": "train",
        "target": "KEEP",
        "metadata": {
            "identity": f"pposm-lifecycle-residual|{candidate}|pre_2024|{signal}",
            "base_identity": f"pposm-counterfactual-action|pre_2024|{signal}",
            "candidate_action": candidate,
            "signal_position": signal,
            "signal_time": signal_time,
            "window": "pre_2024",
        },
    }


def test_route_stream_uses_independent_thresholds_and_normalized_excess_tie_skip():
    rows = [
        _score(10, "SKIP", 0.30),
        _score(10, "TP12", 0.50),
        _score(20, "SKIP", 0.45),
        _score(20, "TP12", 0.65),
        _score(30, "SKIP", 0.10),
        _score(30, "TP12", 0.20),
    ]
    # Signal 10: SKIP excess .20 beats TP12 excess .10. Signal 20: excess ties
    # at .35, so SKIP wins. Signal 30: neither specialist strictly switches.
    routes = router.route_stream_for_thresholds(
        rows,
        [10, 11, 20, 30],
        {"SKIP": 0.10, "TP12": 0.40},
    )
    assert routes == ("SKIP", "TP4", "SKIP", "TP4")


def test_pair_threshold_candidates_are_independent_with_nextafter_sentinels():
    rows = [_score(1, "SKIP", -1.0), _score(1, "TP12", 2.0), _score(2, "SKIP", 3.0), _score(2, "TP12", 2.0)]
    skip, tp12 = router._pair_threshold_candidates(rows)
    assert len(skip) == 4
    assert skip[0] < -1.0 and skip[-1] > 3.0
    assert len(tp12) == 3
    assert tp12[0] < 2.0 and tp12[-1] > 2.0


def test_bind_specialist_scores_to_train_data_restores_frozen_interleaved_order():
    signals = range(102)
    train_rows = [row for signal in signals for row in (_train_row(signal, "SKIP"), _train_row(signal, "TP12"))]
    skip_rows = [_score(signal, "SKIP", signal / 100.0) for signal in reversed(range(102))]
    tp12_rows = [_score(signal, "TP12", signal / 200.0) for signal in range(102)]
    combined, validation = router.bind_specialist_scores_to_train_data(train_rows, skip_rows, tp12_rows, expected_identity_sha256="")
    assert [row["identity"] for row in combined] == [row["metadata"]["identity"] for row in train_rows]
    assert validation["combined_score_rows"] == 204


def test_bind_specialist_scores_rejects_missing_original_identity():
    train_rows = [row for signal in range(102) for row in (_train_row(signal, "SKIP"), _train_row(signal, "TP12"))]
    skip_rows = [_score(signal, "SKIP", 0.1) for signal in range(102)]
    tp12_rows = [_score(signal + 1, "TP12", 0.2) for signal in range(102)]
    with pytest.raises(ValueError, match="missing specialist score"):
        router.bind_specialist_scores_to_train_data(train_rows, skip_rows, tp12_rows, expected_identity_sha256="")


def test_validate_candidate_score_rows_rejects_wrong_candidate_and_post_2024():
    rows = [_score(i, "SKIP", 0.1) for i in range(102)]
    rows[3]["candidate_action"] = "TP12"
    with pytest.raises(ValueError, match="another candidate_action"):
        router._validate_candidate_score_rows(rows, candidate="SKIP")
    rows = [_score(i, "SKIP", 0.1) for i in range(102)]
    rows[4]["signal_time"] = "2024-01-01"
    with pytest.raises(ValueError, match="pre-2024"):
        router._validate_candidate_score_rows(rows, candidate="SKIP")

    rows = [_score(i, "SKIP", 0.1) for i in range(102)]
    rows[4]["score_normalization"] = "sum"
    with pytest.raises(ValueError, match="mean label-logprob"):
        router._validate_candidate_score_rows(rows, candidate="SKIP")


def test_evaluate_thresholds_selects_pair_threshold_and_writes_feasible_status(monkeypatch):
    rows = []
    full_signals = list(range(100))
    for i in range(100):
        rows.append(_score(i, "SKIP", 1.0 if i < 10 else -1.0))
        rows.append(_score(i, "TP12", 1.0 if 10 <= i < 20 else -1.0))

    def fake_apply_routes(engine, signals, routes, start, end, spec):
        return tuple(SimpleNamespace(signal_position=i, entry_position=i, exit_position=i, route=route) for i, route in enumerate(routes))

    monkeypatch.setattr(router, "_apply_routes", fake_apply_routes)

    def fake_economics(trades, start, end, cfg):
        skip = sum(trade.route == "SKIP" for trade in trades)
        tp12 = sum(trade.route == "TP12" for trade in trades)
        if skip == 0 and tp12 == 0:
            lift = 0.0
        elif skip == 10 and tp12 == 10:
            lift = 2.0
        else:
            lift = 1.0
        return {
            "base_6bp": {"absolute_return_pct": 10.0 + lift, "cagr_to_strict_mdd": 3.5 + lift, "strict_mdd_pct": 5.0, "trades": 50},
            "stress_10bp": {"absolute_return_pct": 9.0 + lift, "cagr_to_strict_mdd": 3.2 + lift, "strict_mdd_pct": 5.0, "trades": 50},
        }

    monkeypatch.setattr(router, "_economics", fake_economics)
    result = router.evaluate_thresholds(score_rows=rows, full_signals=full_signals, engine=object(), strategy_cfg=object(), manifest={"spec": {}})
    assert result["status"] == "feasible_train_pair_threshold"
    assert result["selected"]["feasibility"]["materiality"]["non_default_counts"] == {"SKIP": 10, "TP12": 10}
    assert result["candidate_thresholds_by_specialist"] == {"SKIP": 4, "TP12": 4}
