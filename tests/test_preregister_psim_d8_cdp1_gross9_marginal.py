from __future__ import annotations

from pathlib import Path

import pytest

from training import preregister_psim_d8_cdp1_gross9_marginal as prereg


def test_fixed_cdp1_top1_and_no_repair_contract() -> None:
    payload = prereg.build_preregistration()
    candidate = payload["candidate_contract"]
    future = payload["future_veto_contract"]

    assert candidate["fixed_top1"] == "CDP_S50_G05"
    assert candidate["signal_thresholds_or_state_repair_allowed"] is False
    assert future["future_can_rerank"] is False
    assert future["future_can_select_rank2"] is False
    assert future["future_can_repair_candidate_or_weight"] is False


def test_same_gross_grid_and_mdd_gate_are_frozen() -> None:
    payload = prereg.build_preregistration()
    selection = payload["selection_contract"]

    assert selection["candidate_weight_grid"] == [0.25, 0.5, 0.75, 1.0]
    assert selection["same_gross_comparator"] == (
        "Gross9 weights multiplied by (9 + candidate_weight) / 9"
    )
    assert selection["gates"]["strict_mdd_not_above_unscaled_gross9"] is True
    assert selection["gates"][
        "minimum_cagr_to_strict_mdd_improvement_vs_same_gross"
    ] == 0.05


def test_causal_execution_funding_drawdown_and_overlap_are_frozen() -> None:
    payload = prereg.build_preregistration()
    candidate = payload["candidate_contract"]
    accounting = payload["portfolio_accounting_contract"]

    assert candidate["decision_clock"] == "daily D8 card decision_at 12:05 UTC"
    assert candidate["entry_clock"] == (
        "first 5m open one complete bar after decision_at"
    )
    assert candidate["hold_bars_5m"] == 288
    assert candidate["base_cost_per_side"] == 0.0006
    assert candidate["stress_cost_per_side"] == 0.001
    assert candidate["funding"] == "exact_mark_funding_cashflows"
    assert accounting["bar_timing"] == "signal_t_entry_t_plus_1_open"
    assert accounting["drawdown"].startswith(
        "same-BTC OHLC upper-before-lower strict intraposition MDD"
    )
    assert accounting["overlap_controls"] == [
        "exact entry-position Jaccard against each Gross9 sleeve",
        "occupied-bar Jaccard against each Gross9 sleeve",
        "daily marked-return Pearson and Spearman correlations",
    ]


def test_missing_gross9_payloads_are_bound_without_being_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    real_hash = prereg.sha256_file

    def record(path: Path) -> str:
        opened.append(path)
        return real_hash(path)

    monkeypatch.setattr(prereg, "sha256_file", record)
    payload = prereg.build_preregistration()

    assert prereg.GROSS9_MARKET not in opened
    assert prereg.GROSS9_MARKET_OI not in opened
    assert payload["access_boundary"]["gross9_market_opened_or_hashed"] is False
    assert payload["gross9_authority"][
        "db_or_regenerated_cache_substitution_allowed"
    ] is False


def test_preregistration_is_deterministic_and_self_hashed(tmp_path: Path) -> None:
    first = prereg.build_preregistration()
    second = prereg.build_preregistration()
    core = {key: value for key, value in first.items() if key != "manifest_hash"}

    assert first == second
    assert first["manifest_hash"] == prereg.canonical_hash(core)
    path = tmp_path / "prereg.json"
    assert prereg.write_preregistration(path) == first
    assert prereg.write_preregistration(path) == first
    path.write_text("{}\n")
    with pytest.raises(RuntimeError, match="drift"):
        prereg.write_preregistration(path)
