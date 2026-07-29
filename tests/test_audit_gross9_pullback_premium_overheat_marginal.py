from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import training.audit_gross9_pullback_premium_overheat_marginal as module


def test_preregistration_is_hash_bound_and_semantically_valid() -> None:
    payload = module.load_preregistration(module.PREREGISTRATION)
    assert payload["selection_contract"]["candidate_weight_grid"] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert payload["future_veto_contract"]["future_can_rerank"] is False
    assert payload["future_veto_contract"]["future_can_repair"] is False


def test_semantic_validator_rejects_missing_contamination_disclosure() -> None:
    payload = module.load_preregistration(module.PREREGISTRATION)
    mutated = copy.deepcopy(payload)
    mutated["candidate_disclosure"]["standalone_future_already_exposed"] = False
    with pytest.raises(RuntimeError, match="contamination disclosure"):
        module.validate_preregistration_semantics(mutated)


def test_same_gross_control_has_identical_configured_gross() -> None:
    combined, comparator = module.same_gross_weights(0.75)
    assert sum(combined.values()) == pytest.approx(9.75)
    assert sum(comparator.values()) == pytest.approx(9.75)
    assert combined[module.CANDIDATE] == pytest.approx(0.75)
    assert module.CANDIDATE not in comparator


def test_result_hash_detects_mutation() -> None:
    payload = module.finalize_payload({"phase": "selection", "value": 3})
    module.verify_result_hash(payload)
    mutated = dict(payload)
    mutated["value"] = 4
    with pytest.raises(RuntimeError, match="result hash drifted"):
        module.verify_result_hash(mutated)


def test_paired_statistics_are_deterministic_and_positive() -> None:
    effects = np.asarray([0.002] * 24, dtype=float)
    first = module.paired_statistics(effects)
    second = module.paired_statistics(effects)
    assert first == second
    assert first["active_weeks"] == 24
    assert first["sign_flip_pvalue"] < 0.01
    assert first["bootstrap_90pct_lower_mean"] > 0.0


def test_frozen_candidate_replays_exact_pre2024_schedule() -> None:
    frozen, known_oos = module.validate_candidate_freeze(module.Config())
    assert frozen["freeze_hash"] == known_oos["freeze_hash"]
    assert frozen["selection_schedule_hashes"]["pre_2024"]


def test_bounded_rex_scan_preserves_frozen_row_identity() -> None:
    observed = module.validate_frozen_rex_identity(
        module.legacy_all.Config(candidate_rex_top_n=50)
    )
    assert observed == module.EXPECTED_FROZEN_REX_GATES_HASH


def test_frozen_rex_context_skips_full_legacy_event_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="5min"),
            "open": np.ones(4),
            "high": np.ones(4),
            "low": np.ones(4),
            "close": np.ones(4),
        }
    )
    all_masks = {
        "train": np.asarray([True, True, False, False]),
        "test2024": np.asarray([False, False, True, True]),
        "eval2025": np.zeros(4, dtype=bool),
        "ytd2026": np.zeros(4, dtype=bool),
    }
    frozen_return = np.asarray([0.0, 0.01, 0.0, 0.0])

    def fail_full_builder(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("full legacy event universe must not be built")

    def fake_prep() -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        dict[str, np.ndarray],
        dict[str, float],
    ]:
        return market, pd.DataFrame(index=market.index), all_masks, {}

    def fake_add_rex(
        events: list[dict[str, object]],
        _market: pd.DataFrame,
        masks: dict[str, np.ndarray],
        cfg: object,
    ) -> dict[str, int]:
        assert tuple(masks) == module.SELECTION_SPLITS
        assert cfg.candidate_rex_top_n == module.FROZEN_REX_ROW_INDEX + 1
        module.legacy_base.SLEEVES.append("test_side_effect")
        module.legacy_all.EXTRA_SLEEVES.append("test_side_effect")
        events.extend(
            [
                {
                    "split": "train",
                    "sleeve": "cand_rex_veto_7",
                    "ret": frozen_return,
                },
                {
                    "split": "test2024",
                    "sleeve": "cand_rex_veto_7",
                    "ret": np.zeros(4),
                },
                {
                    "split": "train",
                    "sleeve": "cand_rex_veto_0",
                    "ret": np.zeros(4),
                },
                {
                    "split": "eval2025",
                    "sleeve": "cand_rex_veto_7",
                    "ret": np.zeros(4),
                },
            ]
        )
        return {}

    monkeypatch.setattr(
        module.legacy_base, "build_combined_events", fail_full_builder
    )
    monkeypatch.setattr(module.legacy_base.vw.ep, "_prep", fake_prep)
    monkeypatch.setattr(
        module.legacy_all, "add_rex_veto_candidates", fake_add_rex
    )

    legacy_sleeves = list(module.legacy_base.SLEEVES)
    extra_sleeves = list(module.legacy_all.EXTRA_SLEEVES)
    observed_market, masks, events = module.build_frozen_rex_context(
        module.legacy_all.Config(),
        module.SELECTION_SPLITS,
    )

    assert observed_market is market
    assert tuple(masks) == module.SELECTION_SPLITS
    assert len(events) == 2
    assert events[0]["split"] == "train"
    assert events[0]["sleeve"] == "cand_rex_veto_7"
    assert events[0]["ret"] is frozen_return
    assert module.legacy_base.SLEEVES == legacy_sleeves
    assert module.legacy_all.EXTRA_SLEEVES == extra_sleeves
