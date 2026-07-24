from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import training.run_sec_edgar_bitcoin_operational_capacity_synthetic_gate as gate

from training.preregister_sec_edgar_bitcoin_operational_capacity import (
    CLASSES,
    Config as PreregistrationConfig,
    parse_model_output,
)
from training.run_sec_edgar_bitcoin_operational_capacity_synthetic_gate import (
    Config,
    _validate_config,
    checkpoint_rank,
    final_gate,
    prediction_summary,
    schedule_multiplier,
    select_checkpoint,
    swap_invariance,
    validate_preregistration,
    validate_split_rows,
)


DATASETS = {
    "train": Path(
        "data/sec_edgar_bitcoin_operational_capacity_"
        "synthetic_train_2026-07-24.jsonl"
    ),
    "calibration": Path(
        "data/sec_edgar_bitcoin_operational_capacity_"
        "synthetic_calibration_2026-07-24.jsonl"
    ),
    "adversarial": Path(
        "data/sec_edgar_bitcoin_operational_capacity_"
        "synthetic_adversarial_2026-07-24.jsonl"
    ),
    "swaps": Path(
        "data/sec_edgar_bitcoin_operational_capacity_"
        "synthetic_swaps_2026-07-24.jsonl"
    ),
}


def _rows(name: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in DATASETS[name].read_text(encoding="utf-8").splitlines()
    ]


def _perfect_predictions(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    for row in rows:
        output = str(row["expected_output"])
        predictions.append(
            {
                "row_id": row["row_id"],
                "expected_output": output,
                "final_content": output,
                "parsed": parse_model_output(output, str(row["window"])),
                "exact": True,
                "model_called": not bool(row["guarded"]),
            }
        )
    return predictions


def _summary(
    *,
    exact: int,
    malformed: int,
    minimum_share: float,
) -> dict[str, object]:
    return {
        "exact_count": exact,
        "malformed_count": malformed,
        "per_class": {
            label: {"exact_share": minimum_share} for label in CLASSES
        },
    }


def test_preregistration_and_runner_paths_are_frozen() -> None:
    payload = validate_preregistration()
    assert payload["contract_hash"] == gate.PREREGISTRATION_CONTRACT_HASH
    _validate_config(Config())
    with pytest.raises(ValueError, match="paths are frozen"):
        _validate_config(Config(output="/tmp/not-frozen.json"))


def test_schedule_is_four_step_warmup_then_nonzero_cosine_decay() -> None:
    assert schedule_multiplier(1) == 0.25
    assert schedule_multiplier(2) == 0.5
    assert schedule_multiplier(4) == 1.0
    assert schedule_multiplier(5) == 1.0
    assert 0.0 < schedule_multiplier(64) < 0.001
    assert all(
        schedule_multiplier(step) >= schedule_multiplier(step + 1)
        for step in range(5, 64)
    )
    with pytest.raises(ValueError):
        schedule_multiplier(0)
    with pytest.raises(ValueError):
        schedule_multiplier(65)


def test_checkpoint_ranking_uses_only_frozen_ordered_keys() -> None:
    tied = {
        step: _summary(exact=120, malformed=1, minimum_share=0.90)
        for step in (16, 32, 48, 64)
    }
    selected, ranks = select_checkpoint(tied)
    assert selected == 16
    assert ranks["16"] == list(checkpoint_rank(tied[16], 16))

    better_exact = dict(tied)
    better_exact[64] = _summary(exact=121, malformed=10, minimum_share=0.50)
    assert select_checkpoint(better_exact)[0] == 64

    better_floor = dict(tied)
    better_floor[48] = _summary(exact=120, malformed=10, minimum_share=0.91)
    assert select_checkpoint(better_floor)[0] == 48

    fewer_malformed = dict(tied)
    fewer_malformed[32] = _summary(exact=120, malformed=0, minimum_share=0.90)
    assert select_checkpoint(fewer_malformed)[0] == 32


def test_all_frozen_split_rows_validate_before_model_use() -> None:
    specs = {
        "train": (512, 128),
        "calibration": (128, 32),
        "adversarial": (192, 48),
        "swaps": (128, 32),
    }
    for split, (count, per_class) in specs.items():
        validate_split_rows(
            _rows(split),
            split=split,
            expected_rows=count,
            expected_per_class=per_class,
        )


def test_prediction_summary_and_swap_gate_are_exact() -> None:
    rows = _rows("swaps")
    predictions = _perfect_predictions(rows)
    summary = prediction_summary(rows, predictions)
    assert summary["exact_share"] == 1.0
    assert summary["strict_parse_share"] == 1.0
    invariant = swap_invariance(rows, predictions)
    assert invariant == {
        "pairs": 64,
        "invariant_pairs": 64,
        "invariance_share": 1.0,
        "both_exact_pairs": 64,
        "both_exact_share": 1.0,
    }

    predictions[0] = {
        **predictions[0],
        "final_content": "UNSUPPORTED|NONE",
        "parsed": {"class": "UNSUPPORTED", "evidence_id": "NONE"},
        "exact": False,
    }
    assert swap_invariance(rows, predictions)["invariance_share"] < 1.0


def test_final_gate_passes_only_the_complete_frozen_battery() -> None:
    adversarial_rows = _rows("adversarial")
    swap_rows = _rows("swaps")
    adversarial_predictions = _perfect_predictions(adversarial_rows)
    swap_predictions = _perfect_predictions(swap_rows)
    memory = {
        "peak_allocated_bytes": 6 * 1024**3,
        "peak_reserved_bytes": 6 * 1024**3,
    }
    observed = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=adversarial_predictions,
        swap_rows=swap_rows,
        swap_predictions=swap_predictions,
        training_memory=memory,
        inference_memory=memory,
    )
    assert observed["passed"] is True
    assert all(observed["checks"].values())
    assert observed["guarded"]["model_calls"] == 0
    assert observed["ebct_negative"]["unsupported_share"] == 1.0
    assert observed["bpax_negative"]["unsupported_share"] == 1.0

    mixed_index = next(
        index
        for index, row in enumerate(adversarial_rows)
        if row["class"] == "MIXED"
    )
    failed_predictions = list(adversarial_predictions)
    failed_predictions[mixed_index] = {
        **failed_predictions[mixed_index],
        "final_content": "UNSUPPORTED|NONE",
        "parsed": {"class": "UNSUPPORTED", "evidence_id": "NONE"},
        "exact": False,
    }
    failed = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=failed_predictions,
        swap_rows=swap_rows,
        swap_predictions=swap_predictions,
        training_memory=memory,
        inference_memory=memory,
    )
    assert failed["passed"] is False
    assert failed["checks"]["adversarial_mixed_exact"] is False


def test_invariant_but_wrong_swap_pairs_fail_the_gate() -> None:
    adversarial_rows = _rows("adversarial")
    swap_rows = _rows("swaps")
    adversarial_predictions = _perfect_predictions(adversarial_rows)
    wrong_swaps = _perfect_predictions(swap_rows)
    for index, row in enumerate(swap_rows):
        wrong_output = (
            "MIXED|NONE"
            if row["expected_output"] == "UNSUPPORTED|NONE"
            else "UNSUPPORTED|NONE"
        )
        wrong_swaps[index] = {
            **wrong_swaps[index],
            "final_content": wrong_output,
            "parsed": parse_model_output(wrong_output, str(row["window"])),
            "exact": False,
        }
    assert swap_invariance(swap_rows, wrong_swaps)["invariance_share"] == 1.0
    memory = {
        "peak_allocated_bytes": 6 * 1024**3,
        "peak_reserved_bytes": 6 * 1024**3,
    }
    observed = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=adversarial_predictions,
        swap_rows=swap_rows,
        swap_predictions=wrong_swaps,
        training_memory=memory,
        inference_memory=memory,
    )
    assert observed["passed"] is False
    assert observed["checks"]["swap_invariance"] is True
    assert observed["checks"]["swap_exact"] is False


def test_memory_gates_use_allocated_and_reserved_limits() -> None:
    adversarial_rows = _rows("adversarial")
    swap_rows = _rows("swaps")
    perfect_adversarial = _perfect_predictions(adversarial_rows)
    perfect_swaps = _perfect_predictions(swap_rows)
    frozen = PreregistrationConfig()
    failed = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=perfect_adversarial,
        swap_rows=swap_rows,
        swap_predictions=perfect_swaps,
        training_memory={
            "peak_allocated_bytes": frozen.maximum_training_peak_bytes + 1,
            "peak_reserved_bytes": frozen.maximum_training_peak_bytes + 1,
        },
        inference_memory={
            "peak_allocated_bytes": (
                frozen.maximum_inference_peak_allocated_bytes + 1
            ),
            "peak_reserved_bytes": (
                frozen.maximum_inference_peak_reserved_bytes + 1
            ),
        },
    )
    assert failed["passed"] is False
    assert failed["checks"]["training_peak_allocated"] is False
    assert failed["checks"]["training_peak_reserved"] is False
    assert failed["checks"]["inference_peak_allocated"] is False
    assert failed["checks"]["inference_peak_reserved"] is False


def test_run_source_opens_final_splits_only_after_checkpoint_selection() -> None:
    source = inspect.getsource(gate.run)
    selection = source.index("selected_step, ranking = select_checkpoint")
    adversarial_read = source.index('datasets["adversarial"]["path"]')
    swaps_read = source.index('datasets["swaps"]["path"]')
    assert selection < adversarial_read
    assert selection < swaps_read
