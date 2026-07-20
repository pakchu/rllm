from __future__ import annotations

import ast
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import training.build_regional_fiat_cross_rate_stress_support as support


REQUIRED_APIS = (
    "strict_robust_z",
    "derive_features",
    "build_candidates",
    "reserve_clock",
    "accepted_for_split",
    "support_metrics",
    "spearman_abs",
    "exact_jaccard",
    "signed_exposure_correlation",
)
REGIONS = ("EUR", "TRY", "BRL")


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _source_panel(days: int = 1156, *, start: str = "2020-11-01") -> pd.DataFrame:
    """Synthetic completed source closes only; no real regional or outcome values."""
    dates = pd.date_range(start, periods=days, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "source_available_not_before": dates + pd.Timedelta(days=1),
            "BTCUSDT_close": 100.0,
            "BTCEUR_close": 100.0,
            "BTCTRY_close": 100.0,
            "BTCBRL_close": 100.0,
            "source_complete": True,
        }
    )


def _panel_with_residual_shock() -> pd.DataFrame:
    panel = _source_panel()
    base = np.array([(-1.0) ** i * 0.001 for i in range(len(panel))])
    eur = base.copy()
    try_resid = base.copy()
    brl = base.copy()
    eur[182] = 0.020
    try_resid[182] = 0.020
    brl[182] = 0.0
    for name, residuals in [("BTCEUR_close", eur), ("BTCTRY_close", try_resid), ("BTCBRL_close", brl)]:
        panel[name] = 100.0 * np.exp(np.cumsum(residuals))
    return panel


def _events(entries: list[str], sides: list[int] | None = None) -> pd.DataFrame:
    sides = sides or [1] * len(entries)
    frame = pd.DataFrame(
        {
            "entry_time": [_utc(ts) for ts in entries],
            "side": sides,
        }
    )
    frame["exit_time"] = frame["entry_time"] + pd.Timedelta(minutes=5 * 576)
    frame["source_day"] = frame["entry_time"].dt.floor("1D") - pd.Timedelta(days=1)
    frame["state"] = -frame["side"]
    return frame


def _as_frame(result: object) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, tuple) and result and isinstance(result[0], pd.DataFrame):
        return result[0]
    raise AssertionError(f"expected DataFrame-like result, got {type(result)!r}")


def test_required_source_support_apis_exist() -> None:
    missing = [name for name in REQUIRED_APIS if not hasattr(support, name)]
    assert missing == []


def test_strict_robust_z_excludes_current_day_and_requires_180_prior_values() -> None:
    values = pd.Series([0.0, 2.0] * 90 + [1000.0], dtype=float)

    z = support.strict_robust_z(values, window=180)

    assert len(z) == len(values)
    assert z.iloc[:180].isna().all()
    expected = (1000.0 - 1.0) / (1.4826 * 1.0)
    assert z.iloc[180] == pytest.approx(expected)


def test_strict_robust_z_fails_closed_when_prior_mad_is_zero() -> None:
    values = pd.Series([7.0] * 180 + [9.0], dtype=float)

    z = support.strict_robust_z(values, window=180)

    assert pd.isna(z.iloc[180])


def test_derive_features_uses_source_only_closes_and_emits_two_region_event_onset() -> None:
    features = support.derive_features(_panel_with_residual_shock())
    candidates = _as_frame(support.build_candidates(features))

    assert {"z_EUR", "z_TRY", "z_BRL", "common_z", "state"}.issubset(features.columns)
    shock_day = _utc("2021-05-02")
    shock = features.loc[pd.to_datetime(features["date"], utc=True) == shock_day].iloc[0]
    assert shock["state"] == 1
    assert shock["z_EUR"] >= 1
    assert shock["z_TRY"] >= 1
    assert shock["z_BRL"] < 1
    assert len(candidates) == 1
    assert candidates.iloc[0]["entry_time"] == shock_day + pd.Timedelta(days=1, minutes=5)
    assert candidates.iloc[0]["side"] == -1


def test_derive_features_rejects_any_gap_or_partial_frozen_horizon() -> None:
    panel = _source_panel().drop(index=100).reset_index(drop=True)

    with pytest.raises(ValueError, match="exact frozen daily horizon"):
        support.derive_features(panel)


def test_first_valid_nonzero_state_has_no_preceding_valid_onset() -> None:
    rows = pd.DataFrame(
        {
            "date": [_utc("2021-05-01"), _utc("2021-05-02"), _utc("2021-05-03")],
            "state": [1, 0, 1],
            "z_EUR": [2, 0, 2],
            "z_TRY": [2, 0, 2],
            "z_BRL": [0, 0, 0],
        }
    )

    candidates = support.build_candidates(rows)

    assert list(candidates["source_day"]) == [_utc("2021-05-03")]


def test_build_candidates_skips_invalid_days_for_onset_previous_state() -> None:
    rows = pd.DataFrame(
        {
            "date": [_utc("2021-01-01"), _utc("2021-01-02"), _utc("2021-01-03"), _utc("2021-01-04")],
            "state": [0, 1, np.nan, 1],
            "z_EUR": [0, 2, np.nan, 2],
            "z_TRY": [0, 2, np.nan, 2],
            "z_BRL": [0, 0, np.nan, 0],
        }
    )

    candidates = _as_frame(support.build_candidates(rows))

    assert list(pd.to_datetime(candidates["source_day"], utc=True)) == [_utc("2021-01-02")]


def test_reserve_clock_is_global_and_suppresses_cross_split_overlap() -> None:
    candidates = _events(["2022-12-31 00:05", "2023-01-01 00:05", "2023-01-03 00:05"])

    reserved = _as_frame(support.reserve_clock(candidates))

    accepted_reservations = reserved.loc[reserved["reserved"]]
    assert list(pd.to_datetime(accepted_reservations["entry_time"], utc=True)) == [
        _utc("2022-12-31 00:05"),
        _utc("2023-01-03 00:05"),
    ]
    accepted_2023 = _as_frame(
        support.accepted_for_split(reserved, _utc("2023-01-01"), _utc("2024-01-01"))
    )
    assert list(pd.to_datetime(accepted_2023["entry_time"], utc=True)) == [_utc("2023-01-03 00:05")]


def test_accepted_for_split_requires_exact_half_open_576_bar_containment() -> None:
    split_start = _utc("2023-01-01")
    split_end = _utc("2023-01-04 00:05")
    exact = _events(["2023-01-02 00:05"])
    exits_at_end = exact.copy()
    exits_at_end.loc[0, "exit_time"] = split_end
    assert len(_as_frame(support.accepted_for_split(exits_at_end, split_start, split_end))) == 1
    assert _as_frame(
        support.accepted_for_split(exact, split_start, split_end - pd.Timedelta(minutes=5))
    ).empty

    malformed = exact.copy()
    malformed.loc[0, "exit_time"] += pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="exactly 576"):
        support.accepted_for_split(malformed, split_start, split_end)


def test_support_metrics_enforces_split_gates_and_contributor_predicate() -> None:
    entries = [f"2021-{month:02d}-{day:02d} 00:05" for month in range(5, 11) for day in (1, 4, 7)]
    events = _events(entries, sides=[1, -1] * 9)
    events["z_EUR"] = np.where(events["state"] > 0, 2.0, -2.0)
    events["z_TRY"] = np.where(events["state"] > 0, 2.0, -2.0)
    events["z_BRL"] = 0.0

    metrics = support.support_metrics(events, split="train")

    assert metrics["accepted_events"] == 18
    assert metrics["long_count"] == 9
    assert metrics["short_count"] == 9
    assert metrics["quarter_counts"]["2021Q2"] >= 4
    assert metrics["quarter_counts"]["2021Q3"] >= 4
    assert metrics["region_contribution_share"]["EUR"] == pytest.approx(1.0)
    assert metrics["region_contribution_share"]["TRY"] == pytest.approx(1.0)
    assert metrics["region_contribution_share"]["BRL"] == pytest.approx(0.0)
    assert metrics.get("passes_support", False) is False


@pytest.mark.parametrize(
    ("split", "months"),
    [
        ("train", pd.period_range("2021-05", "2022-12", freq="M")),
        ("selection", pd.period_range("2023-01", "2023-12", freq="M")),
    ],
)
def test_support_metrics_passes_only_when_every_frozen_density_gate_passes(
    split: str, months: pd.PeriodIndex
) -> None:
    entries = [
        f"{month.year}-{month.month:02d}-{day:02d} 00:05"
        for month in months
        for day in (2, 10, 18)
    ]
    sides = [1 if index % 2 == 0 else -1 for index in range(len(entries))]
    events = _events(entries, sides=sides)
    events["contributors"] = "EUR+TRY+BRL"

    metrics = support.support_metrics(events, split=split)

    assert metrics["passes_support"] is True
    assert all(metrics["gates"].values())


def test_spearman_abs_fails_closed_for_empty_too_short_constant_or_nonfinite_inputs() -> None:
    bad_inputs = [([], []), ([1.0, 2.0], [2.0, 1.0]), ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]), ([1.0, np.nan, 3.0], [1.0, 2.0, np.inf])]
    for left, right in bad_inputs:
        value = support.spearman_abs(left, right)
        assert value is None or (isinstance(value, float) and not np.isfinite(value))

    assert support.spearman_abs([1, 2, 3], [3, 2, 1]) == pytest.approx(1.0)


def test_exact_jaccard_normalizes_comparator_side_and_time_columns() -> None:
    primary = _events(["2023-01-01 00:05", "2023-01-04 00:05"])
    comparator = pd.DataFrame(
        {
            "clock_name": ["primary", "control"],
            "entry_ts": ["2023-01-01T00:05:00Z", "2023-01-02T00:05:00Z"],
            "direction": ["short", "long"],
        }
    )

    assert support.exact_jaccard(primary, comparator) == pytest.approx(1 / 2)
    empty_jaccard = support.exact_jaccard(
        pd.DataFrame(columns=["entry_time"]),
        pd.DataFrame(columns=["entry_time"]),
    )
    assert not np.isfinite(empty_jaccard)


def test_signed_exposure_correlation_uses_half_open_five_minute_occupancy_and_fails_closed() -> None:
    primary = _events(["2023-01-01 00:00"], sides=[1])
    same = pd.DataFrame(
        {
            "entry_ts": ["2023-01-01T00:00:00Z"],
            "exit_ts": ["2023-01-03T00:00:00Z"],
            "direction": ["long"],
        }
    )
    opposite = pd.DataFrame(
        {
            "entry_ts": ["2023-01-01T00:00:00Z"],
            "exit_ts": ["2023-01-03T00:00:00Z"],
            "direction": ["short"],
        }
    )
    start = _utc("2023-01-01")
    end = _utc("2023-01-04")

    assert support.signed_exposure_correlation(primary, same, start, end) == pytest.approx(1.0)
    assert support.signed_exposure_correlation(primary, opposite, start, end) == pytest.approx(-1.0)
    empty_correlation = support.signed_exposure_correlation(
        pd.DataFrame(), pd.DataFrame(), start, end
    )
    assert not np.isfinite(empty_correlation)


def test_deterministic_random_side_uses_rfxs2_seed_and_is_reproducible() -> None:
    entry = _utc("2023-01-01 00:05")
    expected = 1 if hashlib.sha256(f"RFXS2-576-random-side-20260720|{entry.isoformat()}".encode()).digest()[0] < 128 else -1
    func = getattr(support, "deterministic_random_side", None)
    if func is None:
        pytest.fail("deterministic_random_side API is required for the frozen control")

    assert func(entry) == expected
    assert func(entry) == func(entry)


def test_run_or_evaluate_accepts_injected_synthetic_panels_without_opening_real_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    def forbid_real_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(str(path))
        raise AssertionError(f"test must not open production data path: {path}")

    monkeypatch.setattr(pd, "read_csv", forbid_real_open)
    api = getattr(support, "evaluate", None) or getattr(support, "run", None)
    if api is None:
        pytest.fail("module must expose run or evaluate for injected source-only support evaluation")

    kwargs = {
        "source_panel": _panel_with_residual_shock(),
        "comparators": {},
        "output_dir": tmp_path,
        "write": False,
    }
    try:
        result = api(**kwargs)
    except TypeError as exc:
        pytest.fail(f"run/evaluate must support injected fake source panels; got TypeError: {exc}")

    assert opened == []
    assert result is not None


def test_static_production_input_paths_are_source_only_and_exclude_execution_funding_outcome_sources() -> None:
    module_path = Path(inspect.getsourcefile(support) or "")
    tree = ast.parse(module_path.read_text())
    forbidden = (
        "execution",
        "funding",
        "outcome",
        "pnl",
        "cagr",
        "mdd",
        "um_kline",
        "usd-m",
        "futures",
    )
    suspicious: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if not any(any(token in name.lower() for token in ("path", "file", "input")) for name in targets):
                continue
            value = ast.literal_eval(node.value) if isinstance(node.value, ast.Constant) else None
            if not isinstance(value, str):
                continue
            lowered = value.lower()
            is_production_input = lowered.startswith(("data/", "results/")) or "/data/" in lowered or "/results/" in lowered
            opens_later_data = (lowered.startswith("data/") or "/data/" in lowered) and any(
                token in lowered for token in ("2024-", "2025-", "2026-", "_2024", "_2025", "_2026")
            )
            if is_production_input and (any(word in lowered for word in forbidden) or opens_later_data):
                suspicious.append(("/".join(targets), value))
    assert suspicious == []

    frozen_inputs = set(support.OPENABLE_INPUTS)
    assert frozen_inputs == set(support.STATIC_INPUT_SHA256)
    assert frozen_inputs == {
        str(support.SOURCE_PANEL),
        str(support.SOURCE_MANIFEST),
        str(support.FQPR_CLOCKS),
        str(support.SDDR_CLOCKS),
        "docs/regional-fiat-cross-rate-stress-mechanism-decision-2026-07-20.md",
        "docs/regional-fiat-cross-rate-stress-rfxs576-source-rejection-2026-07-20.md",
        "docs/regional-fiat-cross-rate-stress-v2-mechanism-decision-2026-07-20.md",
    }
    for path in frozen_inputs:
        lowered = path.lower()
        assert not any(word in lowered for word in forbidden)
        if lowered.startswith("data/"):
            assert not any(
                token in lowered
                for token in ("2024-", "2025-", "2026-", "_2024", "_2025", "_2026")
            )

    allowed_path_literals = frozen_inputs | {
        str(support.DEFAULT_OUTPUT),
        str(support.DEFAULT_CLOCK_OUTPUT),
    }
    path_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(("data/", "results/"))
    }
    assert path_literals <= allowed_path_literals

    read_targets = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_csv"
            and node.args
        ):
            argument = node.args[0]
            read_targets.append(argument.id if isinstance(argument, ast.Name) else None)
    assert read_targets == ["SOURCE_PANEL", "FQPR_CLOCKS", "SDDR_CLOCKS"]
