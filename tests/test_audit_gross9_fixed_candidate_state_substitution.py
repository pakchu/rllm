from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import training.audit_gross9_fixed_candidate_state_substitution as module


def _preregistration() -> dict:
    return module.load_preregistration(module.PREREGISTRATION)


def _metric(
    *,
    absolute: float,
    cagr: float,
    mdd: float,
    ratio: float,
) -> dict:
    return {
        "absolute_return_pct": absolute,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": ratio,
        "trades": 100,
    }


def test_preregistration_freezes_candidate_universe_and_grids() -> None:
    payload = _preregistration()
    assert set(
        payload["candidate_universe"]["addition_candidates"]
    ) == set(module.ADDITION_CANDIDATES)
    assert set(
        payload["candidate_universe"]["state_substitution_candidates"]
    ) == set(module.STATE_CANDIDATES)
    assert payload["selection_contract"]["addition_weight_grid"] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert payload["future_veto_contract"]["future_can_rerank"] is False


def test_preregistration_hash_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(module.PREREGISTRATION.read_text())
    payload["selection_contract"]["gross_cap"] = 9.5
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="preregistration hash drifted"):
        module.load_preregistration(path)


def test_addition_weights_keep_gross9_and_use_same_gross_control() -> None:
    baseline = {
        "fresh_kimchi_fx": 2.0,
        "frozen_annual_rank7": 3.0,
        "rex_taker_low_range_position": 0.4,
        "cand_rex_veto_7": 1.6,
        "markov_transition_long": 2.0,
    }
    weights, control, gross = module._cell_weights(
        mode="addition",
        candidate="nonpb30_taker",
        changed_weight=0.5,
        baseline_weights=baseline,
    )
    assert weights["nonpb30_taker"] == 0.5
    assert sum(weights.values()) == 9.5
    assert gross == 9.5
    assert sum(control.values()) == pytest.approx(9.5)
    assert control["markov_transition_long"] == pytest.approx(2.0 * 9.5 / 9.0)


def test_state_substitution_preserves_gross_and_family_weight() -> None:
    baseline = {
        "fresh_kimchi_fx": 2.0,
        "frozen_annual_rank7": 3.0,
        "rex_taker_low_range_position": 0.4,
        "cand_rex_veto_7": 1.6,
        "markov_transition_long": 2.0,
    }
    candidate = "bocpd_top10_strict_majority_long"
    weights, control, gross = module._cell_weights(
        mode="state_substitution",
        candidate=candidate,
        changed_weight=0.75,
        baseline_weights=baseline,
    )
    assert gross == 9.0
    assert sum(weights.values()) == 9.0
    assert weights["markov_transition_long"] == 1.25
    assert weights[candidate] == 0.75
    assert (
        weights["markov_transition_long"] + weights[candidate]
    ) == 2.0
    assert control == baseline


def test_selection_requires_both_windows_to_beat_comparator() -> None:
    baseline = {
        "train": _metric(absolute=100, cagr=40, mdd=10, ratio=4.0),
        "test2024": _metric(absolute=100, cagr=40, mdd=10, ratio=4.0),
    }
    candidate = {
        "train": _metric(absolute=101, cagr=42, mdd=9.5, ratio=4.4),
        "test2024": _metric(
            absolute=102, cagr=43, mdd=9.5, ratio=4.5
        ),
    }
    comparator = {
        "train": _metric(
            absolute=100.5, cagr=40.5, mdd=10, ratio=4.1
        ),
        "test2024": _metric(
            absolute=100.5, cagr=40.5, mdd=10, ratio=4.1
        ),
    }
    standalone = {
        split: _metric(absolute=1, cagr=1, mdd=1, ratio=1)
        for split in module.SELECTION_SPLITS
    }
    row = module.selection_row(
        mode="addition",
        candidate_name="nonpb30_taker",
        changed_weight=0.25,
        gross=9.25,
        baseline=baseline,
        candidate=candidate,
        comparator=comparator,
        standalone=standalone,
        max_entry_jaccard=0.1,
        preregistration=_preregistration(),
    )
    assert row["passes"] is True
    bad = {
        **comparator,
        "test2024": _metric(
            absolute=100.5, cagr=40.5, mdd=10, ratio=4.48
        ),
    }
    failed = module.selection_row(
        mode="addition",
        candidate_name="nonpb30_taker",
        changed_weight=0.25,
        gross=9.25,
        baseline=baseline,
        candidate=candidate,
        comparator=bad,
        standalone=standalone,
        max_entry_jaccard=0.1,
        preregistration=_preregistration(),
    )
    assert failed["passes"] is False
    assert failed["checks"]["comparator_improvement"] is False


def test_state_diagnostics_excludes_markov_only_from_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = "kalman_top10_strict_majority_long"
    monkeypatch.setattr(
        module.portfolio,
        "SLEEVES",
        ("markov_transition_long", "fresh_kimchi_fx", candidate),
    )
    dates = pd.date_range("2024-01-01", periods=4, freq="12h")
    shape = (3, 4)
    entries = {
        "markov_transition_long": np.asarray([1]),
        "fresh_kimchi_fx": np.asarray([2]),
        candidate: np.asarray([1]),
    }
    data = {
        "R": np.zeros(shape),
        "A": np.zeros(shape),
        "U": np.zeros(shape),
        "L": np.zeros(shape),
        "H": np.zeros(shape),
        "counts": np.asarray([1, 1, 1]),
        "wins": np.zeros(3, dtype=int),
        "dates": dates,
        "entry_positions": entries,
    }
    data["L"][0, 1] = -0.01
    data["H"][1, 2] = 0.01
    data["L"][2, 1] = -0.01
    diagnostics = module._candidate_diagnostics(
        {split: data for split in module.SELECTION_SPLITS},
        {"markov_transition_long": 2.0, "fresh_kimchi_fx": 2.0},
        candidate,
        exclude_markov_from_acceptance=True,
    )
    markov = diagnostics["train"]["per_sleeve"][
        "markov_transition_long"
    ]
    assert markov["entry_jaccard"] == 1.0
    assert markov["acceptance_included"] is False
    assert diagnostics["max_acceptance_entry_jaccard"] == 0.0


def test_candidate_specs_match_committed_execution_contracts() -> None:
    specs = module._candidate_specs(module.Config())
    assert specs["nonpb30_taker"]["hold"] == 72
    assert specs["nonpb30_taker"]["stride"] == 12
    assert specs["oi_divergence_highfreq"]["hold"] == 30
    assert specs["oi_divergence_highfreq"]["stride"] == 6


def _fake_preregistered_hashes(
    preregistration: dict,
) -> tuple[dict[Path, str], list[Path]]:
    hashes = {
        Path(record["path"]).resolve(strict=True): str(record["sha256"])
        for record in preregistration["input_provenance"].values()
    }
    opened: list[Path] = []
    return hashes, opened


def test_configured_candidate_path_must_equal_preregistered_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration = _preregistration()
    source = Path(module.Config.nonpb30_config)
    replacement = tmp_path / source.name
    replacement.write_bytes(source.read_bytes())
    hashes, opened = _fake_preregistered_hashes(preregistration)

    def fake_sha256(path: str | Path) -> str:
        resolved = Path(path).resolve(strict=True)
        opened.append(resolved)
        return hashes[resolved]

    monkeypatch.setattr(module, "_sha256", fake_sha256)
    with pytest.raises(RuntimeError, match="configured path mismatch for nonpb30_config"):
        module.validate_inputs(
            module.Config(nonpb30_config=str(replacement)),
            preregistration,
        )


def test_state_scan_paths_and_builder_are_bound_to_preregistration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration = _preregistration()
    source = Path(module.state_top10.SCAN_PATHS["kalman"])
    replacement = tmp_path / source.name
    replacement.write_bytes(source.read_bytes())
    hashes, opened = _fake_preregistered_hashes(preregistration)

    def fake_sha256(path: str | Path) -> str:
        resolved = Path(path).resolve(strict=True)
        opened.append(resolved)
        return hashes[resolved]

    monkeypatch.setattr(module, "_sha256", fake_sha256)
    monkeypatch.setitem(module.state_top10.SCAN_PATHS, "kalman", replacement)
    with pytest.raises(RuntimeError, match="configured path mismatch for kalman_scan"):
        module.validate_inputs(module.Config(), preregistration)


def test_pre2025_input_validation_does_not_open_future_veto_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration = _preregistration()
    hashes, opened = _fake_preregistered_hashes(preregistration)

    def fake_sha256(path: str | Path) -> str:
        resolved = Path(path).resolve(strict=True)
        opened.append(resolved)
        return hashes[resolved]

    monkeypatch.setattr(module, "_sha256", fake_sha256)
    records = module.validate_inputs(module.Config(), preregistration)
    assert tuple(records) == module.SELECTION_PROVENANCE_KEYS
    assert Path(module.state_top10.__file__).resolve() in opened
    for name in module.FUTURE_ONLY_PROVENANCE_KEYS:
        assert Path(
            preregistration["input_provenance"][name]["path"]
        ).resolve() not in opened


def test_future_only_gross9_drift_cannot_change_pre2025_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration = copy.deepcopy(_preregistration())
    for name in module.FUTURE_ONLY_PROVENANCE_KEYS:
        preregistration["input_provenance"][name] = {
            "path": str(tmp_path / f"missing-{name}.json"),
            "sha256": "future-only-drift",
        }
    hashes, opened = _fake_preregistered_hashes(_preregistration())

    def fake_sha256(path: str | Path) -> str:
        resolved = Path(path).resolve(strict=True)
        opened.append(resolved)
        return hashes[resolved]

    monkeypatch.setattr(module, "_sha256", fake_sha256)
    records = module.validate_inputs(module.Config(), preregistration)
    authority = module.validate_pre2025_gross9_anchor(
        module.Config(), preregistration
    )
    assert tuple(records) == module.SELECTION_PROVENANCE_KEYS
    assert authority["future_bearing_source_opened"] is False
    assert authority["selection_windows"] == ["train", "test2024"]


def test_pre2025_gross9_anchor_contains_no_future_metrics() -> None:
    anchor = json.loads(
        Path(module.Config.gross9_pre2025_anchor).read_text()
    )
    assert set(anchor["selection_stats"]) == set(module.SELECTION_SPLITS)
    assert anchor["future_metrics_present"] is False
    serialized = json.dumps(anchor)
    assert "eval2025" not in serialized
    assert "ytd2026" not in serialized


def test_source_exposes_pre2025_phase_only() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'choices=("pre2025",)' in source
    assert module.SELECTION_SPLITS == ("train", "test2024")
