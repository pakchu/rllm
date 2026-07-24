from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import training.run_venue_maintenance_extension_release_synthetic_gate as gate
from training.preregister_venue_maintenance_extension_release import (
    CLASSES,
    Config as PreregistrationConfig,
    parse_model_output,
)
from training.run_venue_maintenance_extension_release_synthetic_gate import (
    Config,
    _ordered_train_rows,
    _validate_retention_budget,
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
        "data/venue_maintenance_extension_release_synthetic_train_2026-07-24.jsonl"
    ),
    "calibration": Path(
        "data/venue_maintenance_extension_release_"
        "synthetic_calibration_2026-07-24.jsonl"
    ),
    "adversarial": Path(
        "data/venue_maintenance_extension_release_"
        "synthetic_adversarial_2026-07-24.jsonl"
    ),
    "swaps": Path(
        "data/venue_maintenance_extension_release_synthetic_swaps_2026-07-24.jsonl"
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


def _base_predictions(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    predictions = _perfect_predictions(rows)
    for index, row in enumerate(rows):
        if row["guarded"]:
            continue
        output = "UNSUPPORTED|NONE|NONE|NONE"
        predictions[index] = {
            **predictions[index],
            "final_content": output,
            "parsed": parse_model_output(output, str(row["window"])),
            "exact": output == row["expected_output"],
        }
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
        "per_class": {label: {"exact_share": minimum_share} for label in CLASSES},
    }


def _memory(gib: float = 6.0) -> dict[str, int]:
    value = int(gib * 1024**3)
    return {
        "peak_allocated_bytes": value,
        "peak_reserved_bytes": value,
    }


def test_preregistration_and_runner_paths_are_frozen() -> None:
    payload = validate_preregistration()
    assert payload["contract_sha256"] == gate.PREREGISTRATION_CONTRACT_HASH
    assert payload["manifest_sha256"] == gate.PREREGISTRATION_MANIFEST_HASH
    _validate_config(Config())
    with pytest.raises(ValueError, match="paths are frozen"):
        _validate_config(Config(output="/tmp/not-frozen.json"))


def test_schedule_is_four_step_warmup_then_nonzero_cosine_decay() -> None:
    assert schedule_multiplier(1) == 0.25
    assert schedule_multiplier(2) == 0.5
    assert schedule_multiplier(4) == 1.0
    assert schedule_multiplier(5) == 1.0
    assert 0.0 < schedule_multiplier(48) < 0.01
    assert all(
        schedule_multiplier(step) >= schedule_multiplier(step + 1)
        for step in range(5, 48)
    )
    with pytest.raises(ValueError):
        schedule_multiplier(0)
    with pytest.raises(ValueError):
        schedule_multiplier(49)


def test_checkpoint_ranking_uses_only_frozen_ordered_keys() -> None:
    tied = {
        step: _summary(exact=130, malformed=1, minimum_share=0.90)
        for step in (12, 24, 36, 48)
    }
    selected, ranks = select_checkpoint(tied)
    assert selected == 12
    assert ranks["12"] == list(checkpoint_rank(tied[12], 12))

    better_exact = dict(tied)
    better_exact[48] = _summary(
        exact=131,
        malformed=10,
        minimum_share=0.50,
    )
    assert select_checkpoint(better_exact)[0] == 48

    better_floor = dict(tied)
    better_floor[36] = _summary(
        exact=130,
        malformed=10,
        minimum_share=0.91,
    )
    assert select_checkpoint(better_floor)[0] == 36

    fewer_malformed = dict(tied)
    fewer_malformed[24] = _summary(
        exact=130,
        malformed=0,
        minimum_share=0.90,
    )
    assert select_checkpoint(fewer_malformed)[0] == 24


def test_all_frozen_split_rows_validate_before_model_use() -> None:
    specs = {
        "train": (384, 128),
        "calibration": (144, 48),
        "adversarial": (144, 48),
        "swaps": (96, 32),
    }
    for split, (count, per_class) in specs.items():
        validate_split_rows(
            _rows(split),
            split=split,
            expected_rows=count,
            expected_per_class=per_class,
        )


def test_train_order_reconstructs_and_verifies_frozen_hash() -> None:
    rows = _rows("train")
    preregistration = validate_preregistration()
    ordered = _ordered_train_rows(rows, preregistration)
    assert len(ordered) == 384
    assert {row["row_id"] for row in ordered} == {row["row_id"] for row in rows}


def test_prediction_summary_and_swap_gate_are_exact() -> None:
    rows = _rows("swaps")
    predictions = _perfect_predictions(rows)
    summary = prediction_summary(rows, predictions)
    assert summary["exact_share"] == 1.0
    assert summary["strict_parse_share"] == 1.0
    invariant = swap_invariance(rows, predictions)
    assert invariant == {
        "pairs": 48,
        "invariant_pairs": 48,
        "invariance_share": 1.0,
        "both_exact_pairs": 48,
        "both_exact_share": 1.0,
    }

    predictions[0] = {
        **predictions[0],
        "final_content": "UNSUPPORTED|NONE|NONE|NONE",
        "parsed": {
            "class": "UNSUPPORTED",
            "start_id": "NONE",
            "extension_id": "NONE",
            "completion_id": "NONE",
        },
        "exact": False,
    }
    assert swap_invariance(rows, predictions)["invariance_share"] < 1.0


def test_final_gate_passes_only_complete_battery_and_base_improvement() -> None:
    adversarial_rows = _rows("adversarial")
    swap_rows = _rows("swaps")
    adversarial_predictions = _perfect_predictions(adversarial_rows)
    swap_predictions = _perfect_predictions(swap_rows)
    observed = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=adversarial_predictions,
        base_adversarial_predictions=_base_predictions(adversarial_rows),
        swap_rows=swap_rows,
        swap_predictions=swap_predictions,
        base_swap_predictions=_base_predictions(swap_rows),
        training_memory=_memory(),
        inference_memory=_memory(),
    )
    assert observed["passed"] is True
    assert all(observed["checks"].values())
    assert observed["guarded"]["rows"] == 13
    assert observed["guarded"]["model_calls"] == 0
    assert observed["guarded"]["by_split"]["adversarial"]["rows"] == 7
    assert observed["guarded"]["by_split"]["swaps"]["rows"] == 6
    assert observed["base_comparison"]["strictly_outperformed"] is True

    material_index = next(
        index
        for index, row in enumerate(adversarial_rows)
        if row["class"] == "MATERIAL_EXTENSION_COMPLETED"
    )
    failed_predictions = list(adversarial_predictions)
    failed_predictions[material_index] = {
        **failed_predictions[material_index],
        "final_content": "UNSUPPORTED|NONE|NONE|NONE",
        "parsed": {
            "class": "UNSUPPORTED",
            "start_id": "NONE",
            "extension_id": "NONE",
            "completion_id": "NONE",
        },
        "exact": False,
    }
    failed = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=failed_predictions,
        base_adversarial_predictions=_base_predictions(adversarial_rows),
        swap_rows=swap_rows,
        swap_predictions=swap_predictions,
        base_swap_predictions=_base_predictions(swap_rows),
        training_memory=_memory(),
        inference_memory=_memory(),
    )
    assert failed["passed"] is False
    assert failed["checks"]["adversarial_material_extension_completed_exact"] is False

    no_improvement = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=adversarial_predictions,
        base_adversarial_predictions=adversarial_predictions,
        swap_rows=swap_rows,
        swap_predictions=swap_predictions,
        base_swap_predictions=swap_predictions,
        training_memory=_memory(),
        inference_memory=_memory(),
    )
    assert no_improvement["passed"] is False
    assert no_improvement["checks"]["base_strictly_outperformed"] is False


def test_swap_guarded_rows_must_be_exact_without_model_calls() -> None:
    adversarial_rows = _rows("adversarial")
    swap_rows = _rows("swaps")
    swap_predictions = _perfect_predictions(swap_rows)
    guarded_index = next(
        index for index, row in enumerate(swap_rows) if row["guarded"]
    )
    swap_predictions[guarded_index] = {
        **swap_predictions[guarded_index],
        "model_called": True,
    }
    observed = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=_perfect_predictions(adversarial_rows),
        base_adversarial_predictions=_base_predictions(adversarial_rows),
        swap_rows=swap_rows,
        swap_predictions=swap_predictions,
        base_swap_predictions=_base_predictions(swap_rows),
        training_memory=_memory(),
        inference_memory=_memory(),
    )
    assert observed["passed"] is False
    assert observed["checks"]["guarded_zero_model_calls"] is False
    assert observed["guarded"]["by_split"]["swaps"]["model_calls"] == 1


def test_invariant_but_wrong_swap_pairs_fail_the_gate() -> None:
    adversarial_rows = _rows("adversarial")
    swap_rows = _rows("swaps")
    wrong_swaps = _perfect_predictions(swap_rows)
    for index, row in enumerate(swap_rows):
        output = "CONTRADICTORY|NONE|NONE|NONE"
        wrong_swaps[index] = {
            **wrong_swaps[index],
            "final_content": output,
            "parsed": parse_model_output(output, str(row["window"])),
            "exact": output == row["expected_output"],
        }
    assert swap_invariance(swap_rows, wrong_swaps)["invariance_share"] == 1.0
    observed = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=_perfect_predictions(adversarial_rows),
        base_adversarial_predictions=_base_predictions(adversarial_rows),
        swap_rows=swap_rows,
        swap_predictions=wrong_swaps,
        base_swap_predictions=_base_predictions(swap_rows),
        training_memory=_memory(),
        inference_memory=_memory(),
    )
    assert observed["passed"] is False
    assert observed["checks"]["swap_invariance"] is True
    assert observed["checks"]["swap_exact"] is False


def test_memory_gates_use_allocated_and_reserved_limits() -> None:
    adversarial_rows = _rows("adversarial")
    swap_rows = _rows("swaps")
    selected_adversarial = _perfect_predictions(adversarial_rows)
    selected_swaps = _perfect_predictions(swap_rows)
    frozen = PreregistrationConfig()
    failed = final_gate(
        adversarial_rows=adversarial_rows,
        adversarial_predictions=selected_adversarial,
        base_adversarial_predictions=_base_predictions(adversarial_rows),
        swap_rows=swap_rows,
        swap_predictions=selected_swaps,
        base_swap_predictions=_base_predictions(swap_rows),
        training_memory={
            "peak_allocated_bytes": frozen.maximum_training_peak_bytes + 1,
            "peak_reserved_bytes": frozen.maximum_training_peak_bytes + 1,
        },
        inference_memory={
            "peak_allocated_bytes": (frozen.maximum_inference_peak_allocated_bytes + 1),
            "peak_reserved_bytes": (frozen.maximum_inference_peak_reserved_bytes + 1),
        },
    )
    assert failed["passed"] is False
    assert failed["checks"]["training_peak_allocated"] is False
    assert failed["checks"]["training_peak_reserved"] is False
    assert failed["checks"]["inference_peak_allocated"] is False
    assert failed["checks"]["inference_peak_reserved"] is False


def test_retention_budget_rejects_prospective_selected_copy() -> None:
    gib = 1024**3
    assert _validate_retention_budget(gib - 2, 1) == gib - 1
    with pytest.raises(RuntimeError, match="exceed 1 GiB"):
        _validate_retention_budget(gib - 2, 2)

    source = inspect.getsource(gate.run)
    budget_check = source.index(
        "_validate_retention_budget(\n"
        "            retained_bytes,\n"
        '            int(selected_checkpoint_manifest["bytes"]),'
    )
    copy = source.index("shutil.copytree")
    assert budget_check < copy


def test_run_source_opens_final_splits_only_after_checkpoint_selection() -> None:
    source = inspect.getsource(gate.run)
    selection = source.index("selected_step, ranking = select_checkpoint")
    adversarial_read = source.index(
        'adversarial_rows = _read_jsonl(dataset_paths["adversarial"])'
    )
    swaps_read = source.index('swap_rows = _read_jsonl(dataset_paths["swaps"])')
    assert selection < adversarial_read
    assert selection < swaps_read
