from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from training import build_gross9_qtr_distill_clocks as builder


def _clock(component: str, rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    values = []
    for split, raw, side in rows:
        entry = pd.Timestamp(raw)
        values.append(
            {
                "candidate": component,
                "control": "primary",
                "split": split,
                "decision_time": entry - pd.Timedelta("10m"),
                "feature_available_time": entry - pd.Timedelta("5m"),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta("8h"),
                "side": side,
            }
        )
    return pd.DataFrame(values, columns=builder.COMMON_CLOCK_FIELDS)


def _write_clock(path: Path, component: str, rows: list[tuple[str, str, int]]) -> None:
    frame = _clock(component, rows)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=builder.COMMON_CLOCK_FIELDS)
        writer.writeheader()
        for row in frame.to_dict("records"):
            for column in ("decision_time", "feature_available_time", "entry_time", "exit_time"):
                row[column] = pd.Timestamp(row[column]).isoformat().replace("+00:00", "Z")
            writer.writerow(row)


def _binding(tmp_path: Path, component: str, rows: list[tuple[str, str, int]]) -> dict[str, dict[str, object]]:
    path = tmp_path / f"{component}.csv.gz"
    _write_clock(path, component, rows)
    return {"clock": {"path": str(path), "sha256": builder.sha256_file(path), "rows": len(rows)}}


def test_sleeve_veto_is_split_isolated_strict_lower_latest_and_no_reverse() -> None:
    base = _clock(
        "HVDEMWMV-24",
        [
            ("train", "2023-07-01T06:00:00Z", 1),
            ("train", "2023-07-01T14:00:00Z", 1),
            ("test", "2024-01-01T02:00:00Z", 1),
        ],
    )
    veto = _clock(
        "HVCQTR-24",
        [
            ("train", "2023-07-01T00:00:00Z", -1),  # exactly t-6h: excluded
            ("train", "2023-07-01T05:00:00Z", -1),  # older opposite loses to latest same-side
            ("train", "2023-07-01T06:00:00Z", 1),
            ("train", "2023-07-01T14:00:00Z", -1),  # latest opposite suppresses, no reverse short
            ("train", "2024-01-01T01:00:00Z", -1),  # wrong split; must not suppress test
        ],
    )
    result, diag = builder.build_sleeve_clock("HVDEMWMV-24", base, veto, weight=1 / 6)
    assert result[["split", "entry_time", "side"]].values.tolist() == [
        ["train", pd.Timestamp("2023-07-01T06:00:00Z"), 1],
        ["test", pd.Timestamp("2024-01-01T02:00:00Z"), 1],
    ]
    assert result.iloc[0]["veto_relation"] == "same_side_latest_keep"
    assert result.iloc[1]["veto_relation"] == "none"
    assert diag["opposite_latest_veto_suppressions"] == 1
    assert set(result["side"]) == {1}


def test_sleeve_local_half_open_reservation_is_per_split() -> None:
    base = _clock(
        "HVCPF17-8",
        [
            ("train", "2023-07-01T00:00:00Z", 1),
            ("train", "2023-07-01T01:00:00Z", 1),
            ("train", "2023-07-01T08:00:00Z", -1),
            ("test", "2023-07-01T01:00:00Z", -1),
        ],
    )
    veto = _clock("HVCQTR-24", [])
    result, diag = builder.build_sleeve_clock("HVCPF17-8", base, veto, weight=1 / 6)
    assert result[["split", "entry_time", "side"]].values.tolist() == [
        ["train", pd.Timestamp("2023-07-01T00:00:00Z"), 1],
        ["train", pd.Timestamp("2023-07-01T08:00:00Z"), -1],
        ["test", pd.Timestamp("2023-07-01T01:00:00Z"), -1],
    ]
    assert diag["reservation_dropped_rows"] == 1


def test_portfolio_transitions_exit_before_entry_and_net_without_priority() -> None:
    a = pd.DataFrame(
        [
            {"candidate": builder.sleeve_id("HVDEMWMV-24"), "control": "primary", "split": "train", "base_component_id": "HVDEMWMV-24", "veto_component_id": builder.VETO_COMPONENT, "weight": 1 / 6, "base_entry_time": pd.Timestamp("2023-07-01T00:00:00Z"), "veto_entry_time": pd.NaT, "veto_side": pd.NA, "veto_relation": "none", "decision_time": pd.Timestamp("2023-06-30T23:55:00Z"), "feature_available_time": pd.Timestamp("2023-06-30T23:55:00Z"), "entry_time": pd.Timestamp("2023-07-01T00:00:00Z"), "exit_time": pd.Timestamp("2023-07-01T08:00:00Z"), "side": 1, "signed_weight": 1 / 6},
            {"candidate": builder.sleeve_id("HVDEMWMV-24"), "control": "primary", "split": "train", "base_component_id": "HVDEMWMV-24", "veto_component_id": builder.VETO_COMPONENT, "weight": 1 / 6, "base_entry_time": pd.Timestamp("2023-07-01T08:00:00Z"), "veto_entry_time": pd.NaT, "veto_side": pd.NA, "veto_relation": "none", "decision_time": pd.Timestamp("2023-07-01T07:55:00Z"), "feature_available_time": pd.Timestamp("2023-07-01T07:55:00Z"), "entry_time": pd.Timestamp("2023-07-01T08:00:00Z"), "exit_time": pd.Timestamp("2023-07-01T16:00:00Z"), "side": -1, "signed_weight": -1 / 6},
        ],
        columns=builder.SLEEVE_COLUMNS,
    )
    b = a.iloc[[0]].copy()
    b["candidate"] = builder.sleeve_id("HVCPF17-8")
    b["base_component_id"] = "HVCPF17-8"
    b["signed_weight"] = -1 / 6
    b["side"] = -1
    b["exit_time"] = pd.Timestamp("2023-07-01T16:00:00Z")
    trans, seg, eps = builder.build_portfolio_schedules({"HVDEMWMV-24": a, "HVCPF17-8": b})
    at0 = trans.loc[trans["timestamp"].eq(pd.Timestamp("2023-07-01T00:00:00Z"))].iloc[0]
    assert at0["target_exposure"] == pytest.approx(0.0)
    assert at0["gross_exposure"] == pytest.approx(1 / 3)
    at8 = trans.loc[trans["timestamp"].eq(pd.Timestamp("2023-07-01T08:00:00Z"))].iloc[0]
    assert at8["exit_delta"] == pytest.approx(-1 / 6)
    assert at8["entry_delta"] == pytest.approx(-1 / 6)
    assert at8["target_exposure"] == pytest.approx(-1 / 3)
    assert eps["side"].tolist() == [-1]
    assert seg["gross_exposure"].max() <= builder.MAX_GROSS


def test_verify_and_load_authenticate_full_hash_and_row_count(tmp_path: Path) -> None:
    artifacts = {
        "HVDEMWMV-24": _binding(tmp_path, "HVDEMWMV-24", [("train", "2023-07-01T00:00:00Z", 1), ("test", "2024-01-01T00:00:00Z", -1)]),
        "HVCQTR-24": _binding(tmp_path, "HVCQTR-24", [("train", "2023-07-01T00:00:00Z", 1)]),
    }
    verified = builder.verify_bound_component_artifacts(artifacts, ["HVDEMWMV-24", "HVCQTR-24"])
    assert verified["HVDEMWMV-24"]["clock"]["rows"] == 2
    loaded = builder.load_full_component_clock("HVDEMWMV-24", artifacts)
    assert set(loaded["split"]) == {"train", "test"}
    artifacts["HVDEMWMV-24"]["clock"]["rows"] = 1
    with pytest.raises(RuntimeError, match="row-count drift"):
        builder.verify_bound_component_artifacts(artifacts, ["HVDEMWMV-24"])


def test_run_writes_shadow_package_without_outcome_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    components = builder.COMPONENT_ORDER
    clocks = {
        c: _clock(c, [("train", "2023-07-01T00:00:00Z", 1), ("test", "2024-01-01T00:00:00Z", -1)])
        for c in components
    }
    monkeypatch.setattr(builder, "load_validated_preregistration", lambda: {"status": "validated_against_committed_preregistration"})
    monkeypatch.setattr(builder, "verify_bound_component_artifacts", lambda: {c: {"verified": True} for c in components})
    monkeypatch.setattr(builder, "load_full_component_clock", lambda component: clocks[component])
    report = builder.run(tmp_path / "sleeves", tmp_path / "portfolio", tmp_path / "result.json")
    assert report["status"] == "shadow_only_train_distillation_not_formal_alpha"
    assert len(report["sleeves"]) == 4
    assert report["portfolio_source_stats"]["max_gross_exposure"] <= 0.5
    assert report["evidence_boundary"]["market_rows_opened"] is False
    assert report["evidence_boundary"]["funding_opened"] is False
    assert report["evidence_boundary"]["economic_outcomes_opened"] is False
    assert (tmp_path / "result.json").is_file()
    assert len(list((tmp_path / "sleeves").glob("*.csv.gz"))) == 4
    assert len(list((tmp_path / "portfolio").glob("*.csv.gz"))) == 3


def test_preregistration_validation_rejects_missing_or_unbound(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(builder.importlib, "import_module", lambda name: SimpleNamespace(DEFAULT_OUTPUT=tmp_path / "missing.json"))
    with pytest.raises(RuntimeError, match="missing committed preregistration artifact"):
        builder.load_validated_preregistration()

    payload = {
        "policy_id": builder.POLICY_ID,
        "selection_rule": {"selected_bases": list(builder.BASE_ORDER), "winner_veto": builder.VETO_COMPONENT},
        "portfolio_construction": {
            "sleeve_weights": {f"{base}__ASYNC_ACTIVE_OPPOSITE_VETO_6H__{builder.VETO_COMPONENT}": builder.BASE_WEIGHTS[base] for base in builder.BASE_ORDER}
        },
        "implementation": {"portfolio_builder": {"sha256": "PENDING"}},
    }
    payload["manifest_hash"] = builder.canonical_hash(payload)
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    module = SimpleNamespace(DEFAULT_OUTPUT=path, validate=lambda p: None, build=lambda: payload)
    monkeypatch.setattr(builder.importlib, "import_module", lambda name: module)
    with pytest.raises(RuntimeError, match="builder hash is missing or placeholder"):
        builder.load_validated_preregistration()
