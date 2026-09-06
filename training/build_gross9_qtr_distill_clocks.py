"""Build split-specific G9QTR-DISTILL-8 shadow clocks and exposure schedules.

This module intentionally materializes schedules only.  It authenticates complete
component clock files by frozen path/sha256/row-count bindings, then builds a
compact train-derived shadow candidate:

    bases: HVDEMWMV-24, HVCPF17-8, HVDIMIO-8, HVLVR-8
    veto:  HVCQTR-24
    weights: 1/6, 1/6, 1/12, 1/12  (max gross <= 0.5)

For every split independently, each base event is kept unless the latest same-
split HVCQTR event in the strict-lower/inclusive-upper 6h window
(t-6h < veto.entry <= t) is opposite-side.  Opposite veto suppresses to cash and
never reverses.  Each sleeve then applies an 8h sleeve-local half-open
reservation.  Portfolio schedules are deterministic target-exposure schedules;
simultaneous exits are applied before entries, same-direction sleeves sum, and
opposite-direction sleeves net without priority.

No market, funding, price, return, or PnL fields are read here.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_gross9_async_pair_train_clocks as gross9_components
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

POLICY_ID = "G9QTR-DISTILL-8"
PROTOCOL_VERSION = "gross9_qtr_distill_split_clocks_v1"
AS_OF_DATE = "2026-09-02"
PREREG_MODULE = "training.preregister_gross9_qtr_distill"
DEFAULT_RESULT = Path("results/gross9_qtr_distill_split_clock_source_support_2026-09-02.json")
SLEEVE_CLOCK_DIR = Path("data/gross9_qtr_distill_sleeve_clocks_2026-09-02")
PORTFOLIO_DIR = Path("data/gross9_qtr_distill_portfolio_schedules_2026-09-02")

BASE_WEIGHTS: dict[str, float] = {
    "HVDEMWMV-24": 1.0 / 6.0,
    "HVCPF17-8": 1.0 / 6.0,
    "HVDIMIO-8": 1.0 / 12.0,
    "HVLVR-8": 1.0 / 12.0,
}
BASE_ORDER = tuple(BASE_WEIGHTS)
VETO_COMPONENT = "HVCQTR-24"
COMPONENT_ORDER = BASE_ORDER + (VETO_COMPONENT,)
COMPONENT_ARTIFACTS = gross9_components.COMPONENT_ARTIFACTS
COMMON_CLOCK_FIELDS = gross9_components.COMMON_CLOCK_FIELDS
CONFIRMATION_WINDOW = pd.Timedelta(hours=6)
HOLD = pd.Timedelta(hours=8)
GRID = pd.Timedelta(minutes=5)
MAX_GROSS = sum(BASE_WEIGHTS.values())

SLEEVE_COLUMNS = (
    "candidate",
    "control",
    "split",
    "base_component_id",
    "veto_component_id",
    "weight",
    "base_entry_time",
    "veto_entry_time",
    "veto_side",
    "veto_relation",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "signed_weight",
)
TRANSITION_COLUMNS = (
    "candidate",
    "split",
    "timestamp",
    "exit_delta",
    "entry_delta",
    "target_exposure",
    "gross_exposure",
    "active_sleeves",
)
SEGMENT_COLUMNS = (
    "candidate",
    "split",
    "start_time",
    "end_time",
    "target_exposure",
    "gross_exposure",
    "side",
    "active_sleeves",
)
EPISODE_COLUMNS = (
    "candidate",
    "split",
    "episode_id",
    "start_time",
    "end_time",
    "side",
    "max_abs_target_exposure",
    "segments",
)


def sha256_file(path: str | Path) -> str:
    return gross9_components.sha256_file(path)


def canonical_hash(value: Any) -> str:
    return gross9_components.canonical_hash(value)


def _parse_timestamp(value: str, column: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise RuntimeError(f"{POLICY_ID} {column} must be timezone-aware")
    return ts.tz_convert("UTC")


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} expected JSON object: {path}")
    return value


def _validate_manifest_hash(value: Mapping[str, Any], *, label: str) -> None:
    core = dict(value)
    manifest_hash = core.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} {label} manifest hash drift")


def candidate_id() -> str:
    return POLICY_ID


def sleeve_id(base: str) -> str:
    return f"{POLICY_ID}__{base}__QTR_OPPOSITE_VETO_6H"


def split_sort_key(split: str) -> tuple[int, str]:
    order = {"train": 0, "test": 1, "eval": 2, "final": 3}
    return (order.get(split, 99), split)


def load_validated_preregistration(module_name: str = PREREG_MODULE) -> Mapping[str, Any]:
    try:
        prereg = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - exact branch tested via monkeypatch
        raise RuntimeError(f"{POLICY_ID} missing preregistration module: {module_name}") from exc
    output = Path(getattr(prereg, "DEFAULT_OUTPUT", ""))
    if not output.is_file():
        raise RuntimeError(f"{POLICY_ID} missing committed preregistration artifact")
    payload = _read_json_object(output)
    validate = getattr(prereg, "validate", None)
    build = getattr(prereg, "build", None)
    if not callable(validate) or not callable(build):
        raise RuntimeError(f"{POLICY_ID} preregistration module lacks build/validate")
    validate(payload)
    if payload != build():
        raise RuntimeError(f"{POLICY_ID} preregistration artifact differs from code")
    _validate_manifest_hash(payload, label="preregistration")
    if payload.get("policy_id") != POLICY_ID:
        raise RuntimeError(f"{POLICY_ID} preregistration policy drift")
    selection = payload.get("selection_rule", {})
    if tuple(selection.get("selected_bases", ())) != BASE_ORDER or selection.get("winner_veto") != VETO_COMPONENT:
        raise RuntimeError(f"{POLICY_ID} preregistration component drift")
    expected_sleeve_weights = {f"{base}__ASYNC_ACTIVE_OPPOSITE_VETO_6H__{VETO_COMPONENT}": BASE_WEIGHTS[base] for base in BASE_ORDER}
    if payload.get("portfolio_construction", {}).get("sleeve_weights") != expected_sleeve_weights:
        raise RuntimeError(f"{POLICY_ID} preregistration weight drift")
    impl = payload.get("implementation", {}).get("portfolio_builder", {})
    bound_sha = impl.get("sha256")
    if not isinstance(bound_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", bound_sha):
        raise RuntimeError(f"{POLICY_ID} preregistration builder hash is missing or placeholder")
    if bound_sha != sha256_file(__file__):
        raise RuntimeError(f"{POLICY_ID} preregistration builder hash mismatch")
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "manifest_hash": payload["manifest_hash"],
        "status": "validated_against_committed_preregistration",
    }


def verify_bound_component_artifacts(
    component_artifacts: Mapping[str, Mapping[str, Mapping[str, Any]]] = COMPONENT_ARTIFACTS,
    component_order: Sequence[str] = COMPONENT_ORDER,
) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for component in component_order:
        binding = component_artifacts[component]["clock"]
        path = Path(str(binding["path"]))
        expected_sha = str(binding["sha256"])
        expected_rows = int(binding["rows"])
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise RuntimeError(f"{POLICY_ID} {component} clock SHA drift")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or any(field not in reader.fieldnames for field in COMMON_CLOCK_FIELDS):
                raise RuntimeError(f"{POLICY_ID} {component} clock schema drift")
            rows = sum(1 for _ in reader)
        if rows != expected_rows:
            raise RuntimeError(f"{POLICY_ID} {component} full clock row-count drift")
        verified[component] = {"clock": {"path": str(path), "sha256": observed_sha, "rows": rows}, "full_file_authenticated": True}
    return verified


def load_full_component_clock(
    component: str,
    component_artifacts: Mapping[str, Mapping[str, Mapping[str, Any]]] = COMPONENT_ARTIFACTS,
) -> pd.DataFrame:
    if component not in component_artifacts:
        raise ValueError(f"unknown {POLICY_ID} component: {component}")
    binding = component_artifacts[component]["clock"]
    path = Path(str(binding["path"]))
    if sha256_file(path) != str(binding["sha256"]):
        raise RuntimeError(f"{POLICY_ID} {component} clock SHA drift")
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in COMMON_CLOCK_FIELDS):
            raise RuntimeError(f"{POLICY_ID} {component} clock schema drift")
        for raw in reader:
            parsed = {field: raw[field] for field in COMMON_CLOCK_FIELDS}
            for column in ("decision_time", "feature_available_time", "entry_time", "exit_time"):
                parsed[column] = _parse_timestamp(str(parsed[column]), column)
            parsed["side"] = int(parsed["side"])
            rows.append(parsed)
    if len(rows) != int(binding["rows"]):
        raise RuntimeError(f"{POLICY_ID} {component} full clock row-count drift")
    frame = pd.DataFrame(rows, columns=COMMON_CLOCK_FIELDS)
    if frame.empty:
        raise RuntimeError(f"{POLICY_ID} {component} has no rows")
    if not frame["candidate"].eq(component).all() or not frame["control"].eq("primary").all():
        raise RuntimeError(f"{POLICY_ID} {component} identity/control drift")
    if not frame["side"].isin((-1, 1)).all():
        raise RuntimeError(f"{POLICY_ID} {component} non-strict side")
    if frame.duplicated(["split", "entry_time"]).any():
        raise RuntimeError(f"{POLICY_ID} {component} duplicate split/entry_time")
    if not frame["decision_time"].le(frame["entry_time"]).all() or not frame["feature_available_time"].le(frame["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} {component} unavailable feature/decision")
    _validate_5m_grid(frame, ["decision_time", "feature_available_time", "entry_time", "exit_time"])
    return frame.sort_values(["split", "entry_time"], kind="stable").reset_index(drop=True)


def _validate_5m_grid(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if frame.empty:
            continue
        ns = pd.to_datetime(frame[column], utc=True).astype("int64")
        if not (ns % GRID.value == 0).all():
            raise RuntimeError(f"{POLICY_ID} {column} not aligned to 5m grid")


def reserve_half_open(clock: pd.DataFrame) -> pd.DataFrame:
    ordered = clock.copy()
    ordered["_split_sort"] = ordered["split"].map(lambda value: split_sort_key(str(value)))
    ordered = ordered.sort_values(["_split_sort", "entry_time", "candidate", "side"], kind="stable")
    keep: list[int] = []
    next_available_by_split: dict[str, pd.Timestamp] = {}
    for index, row in ordered.iterrows():
        split = str(row["split"])
        next_available = next_available_by_split.get(split)
        if next_available is not None and row["entry_time"] < next_available:
            continue
        keep.append(index)
        next_available_by_split[split] = pd.Timestamp(row["exit_time"])
    return ordered.loc[keep].drop(columns=["_split_sort"]).reset_index(drop=True)


def _latest_veto_same_split(veto_clock: pd.DataFrame, split: str, entry_time: pd.Timestamp) -> pd.Series | None:
    window = veto_clock.loc[
        veto_clock["split"].eq(split)
        & veto_clock["entry_time"].gt(entry_time - CONFIRMATION_WINDOW)
        & veto_clock["entry_time"].le(entry_time)
    ].sort_values("entry_time", kind="stable")
    if window.empty:
        return None
    return window.iloc[-1]


def build_sleeve_clock(base: str, base_clock: pd.DataFrame, veto_clock: pd.DataFrame, *, weight: float | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if weight is None:
        weight = BASE_WEIGHTS[base]
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "base_events_seen": int(len(base_clock)),
        "no_veto_window_keeps": 0,
        "same_side_latest_veto_keeps": 0,
        "opposite_latest_veto_suppressions": 0,
        "pre_reservation_rows": 0,
        "post_reservation_rows": 0,
        "reservation_dropped_rows": 0,
        "by_split": defaultdict(lambda: {"base_events_seen": 0, "kept": 0, "suppressed": 0}),
    }
    for event in base_clock.sort_values(["split", "entry_time"], kind="stable").itertuples(index=False):
        split = str(event.split)
        entry = pd.Timestamp(event.entry_time)
        side = int(event.side)
        diagnostics["by_split"][split]["base_events_seen"] += 1
        veto_row = _latest_veto_same_split(veto_clock, split, entry)
        veto_entry: pd.Timestamp | pd.NaT = pd.NaT
        veto_side: int | pd.NA = pd.NA
        relation = "none"
        decision_time = pd.Timestamp(event.decision_time)
        feature_available_time = pd.Timestamp(event.feature_available_time)
        if veto_row is None:
            diagnostics["no_veto_window_keeps"] += 1
        else:
            veto_entry = pd.Timestamp(veto_row["entry_time"])
            veto_side = int(veto_row["side"])
            decision_time = max(decision_time, pd.Timestamp(veto_row["decision_time"]))
            feature_available_time = max(feature_available_time, pd.Timestamp(veto_row["feature_available_time"]))
            if veto_side == side:
                diagnostics["same_side_latest_veto_keeps"] += 1
                relation = "same_side_latest_keep"
            else:
                diagnostics["opposite_latest_veto_suppressions"] += 1
                diagnostics["by_split"][split]["suppressed"] += 1
                continue
        diagnostics["by_split"][split]["kept"] += 1
        rows.append(
            {
                "candidate": sleeve_id(base),
                "control": "primary",
                "split": split,
                "base_component_id": base,
                "veto_component_id": VETO_COMPONENT,
                "weight": weight,
                "base_entry_time": entry,
                "veto_entry_time": veto_entry,
                "veto_side": veto_side,
                "veto_relation": relation,
                "decision_time": decision_time,
                "feature_available_time": feature_available_time,
                "entry_time": entry,
                "exit_time": entry + HOLD,
                "side": side,
                "signed_weight": side * weight,
            }
        )
    pre = pd.DataFrame(rows, columns=SLEEVE_COLUMNS)
    diagnostics["pre_reservation_rows"] = len(pre)
    clock = reserve_half_open(pre) if not pre.empty else pre
    diagnostics["post_reservation_rows"] = len(clock)
    diagnostics["reservation_dropped_rows"] = len(pre) - len(clock)
    diagnostics["by_split"] = dict(diagnostics["by_split"])
    validate_sleeve_clock(clock)
    return clock, diagnostics


def validate_sleeve_clock(clock: pd.DataFrame) -> None:
    if clock.empty:
        return
    if not clock["side"].isin((-1, 1)).all():
        raise RuntimeError(f"{POLICY_ID} sleeve non-strict side")
    if not clock["decision_time"].le(clock["entry_time"]).all() or not clock["feature_available_time"].le(clock["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} sleeve unavailable decision/feature")
    used = clock.loc[clock["veto_entry_time"].notna()]
    if not used.empty:
        lag = used["entry_time"] - used["veto_entry_time"]
        if not (lag.ge(pd.Timedelta(0)).all() and lag.lt(CONFIRMATION_WINDOW).all()):
            raise RuntimeError(f"{POLICY_ID} sleeve veto lag drift")
    for _, group in clock.sort_values(["split", "entry_time"], kind="stable").groupby("split", sort=False):
        if len(group) > 1 and not group["entry_time"].iloc[1:].reset_index(drop=True).ge(group["exit_time"].iloc[:-1].reset_index(drop=True)).all():
            raise RuntimeError(f"{POLICY_ID} sleeve reservation overlap")
    _validate_5m_grid(clock, ["decision_time", "feature_available_time", "entry_time", "exit_time"])


def build_portfolio_schedules(sleeves: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events_by_split_ts: dict[tuple[str, pd.Timestamp], dict[str, Any]] = defaultdict(lambda: {"exit_delta": 0.0, "entry_delta": 0.0, "exit_sleeves": [], "entry_sleeves": []})
    all_splits: set[str] = set()
    for base, clock in sleeves.items():
        if clock.empty:
            continue
        for row in clock.itertuples(index=False):
            split = str(row.split)
            all_splits.add(split)
            signed = float(row.signed_weight)
            events_by_split_ts[(split, pd.Timestamp(row.exit_time))]["exit_delta"] -= signed
            events_by_split_ts[(split, pd.Timestamp(row.exit_time))]["exit_sleeves"].append(base)
            events_by_split_ts[(split, pd.Timestamp(row.entry_time))]["entry_delta"] += signed
            events_by_split_ts[(split, pd.Timestamp(row.entry_time))]["entry_sleeves"].append(base)

    transition_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for split in sorted(all_splits, key=split_sort_key):
        timestamps = sorted(ts for (s, ts) in events_by_split_ts if s == split)
        exposure = 0.0
        gross = 0.0
        active: dict[str, float] = {}
        previous_ts: pd.Timestamp | None = None
        previous_exposure = 0.0
        previous_gross = 0.0
        previous_active: list[str] = []
        episode_id = 0
        current_episode: dict[str, Any] | None = None
        for ts in timestamps:
            if previous_ts is not None and ts > previous_ts and abs(previous_exposure) > 1e-12:
                side = 1 if previous_exposure > 0 else -1
                segment_rows.append({"candidate": candidate_id(), "split": split, "start_time": previous_ts, "end_time": ts, "target_exposure": previous_exposure, "gross_exposure": previous_gross, "side": side, "active_sleeves": ",".join(previous_active)})
                if current_episode is None or current_episode["side"] != side:
                    if current_episode is not None:
                        current_episode["end_time"] = previous_ts
                        episode_rows.append(current_episode)
                    episode_id += 1
                    current_episode = {"candidate": candidate_id(), "split": split, "episode_id": episode_id, "start_time": previous_ts, "end_time": ts, "side": side, "max_abs_target_exposure": abs(previous_exposure), "segments": 1}
                else:
                    current_episode["end_time"] = ts
                    current_episode["max_abs_target_exposure"] = max(float(current_episode["max_abs_target_exposure"]), abs(previous_exposure))
                    current_episode["segments"] = int(current_episode["segments"]) + 1
            elif previous_ts is not None and current_episode is not None:
                current_episode["end_time"] = previous_ts
                episode_rows.append(current_episode)
                current_episode = None

            bucket = events_by_split_ts[(split, ts)]
            # exits before entries at identical timestamp
            for base in sorted(bucket["exit_sleeves"]):
                active.pop(base, None)
            for base in sorted(bucket["entry_sleeves"]):
                signed = BASE_WEIGHTS[base]
                # find actual signed side from entry delta contribution by scanning rows at ts
                clock = sleeves[base]
                match = clock[(clock["split"].eq(split)) & (clock["entry_time"].eq(ts))]
                if not match.empty:
                    active[base] = float(match.iloc[-1]["signed_weight"])
            exposure = sum(active.values())
            gross = sum(abs(v) for v in active.values())
            if gross > MAX_GROSS + 1e-12 or abs(exposure) > MAX_GROSS + 1e-12:
                raise RuntimeError(f"{POLICY_ID} portfolio max gross drift")
            transition_rows.append({"candidate": candidate_id(), "split": split, "timestamp": ts, "exit_delta": float(bucket["exit_delta"]), "entry_delta": float(bucket["entry_delta"]), "target_exposure": exposure, "gross_exposure": gross, "active_sleeves": ",".join(sorted(active))})
            previous_ts = ts
            previous_exposure = exposure
            previous_gross = gross
            previous_active = sorted(active)
        if current_episode is not None:
            current_episode["end_time"] = previous_ts
            episode_rows.append(current_episode)

    transitions = pd.DataFrame(transition_rows, columns=TRANSITION_COLUMNS)
    segments = pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS)
    episodes = pd.DataFrame(episode_rows, columns=EPISODE_COLUMNS)
    for frame, columns in ((transitions, ["timestamp"]), (segments, ["start_time", "end_time"]), (episodes, ["start_time", "end_time"])):
        _validate_5m_grid(frame, columns)
    return transitions, segments, episodes


def source_stats(clock: pd.DataFrame) -> dict[str, Any]:
    if clock.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "signed_episodes": 0, "months": 0, "iso_weeks": 0, "halves": 0, "splits": {}}
    split_stats: dict[str, Any] = {}
    for split, group in clock.groupby("split", sort=False):
        longs = int(group["side"].eq(1).sum())
        shorts = int(group["side"].eq(-1).sum())
        iso = group["entry_time"].dt.isocalendar()
        months = set(group["entry_time"].dt.strftime("%Y-%m"))
        halves = set(group["entry_time"].dt.strftime("%Y") + "H" + (((group["entry_time"].dt.month - 1) // 6) + 1).astype(str))
        # reservation means events are signed episodes inside a sleeve.
        split_stats[str(split)] = {"events": int(len(group)), "longs": longs, "shorts": shorts, "signed_episodes": int(len(group)), "months": len(months), "iso_weeks": len(set(zip(iso["year"].astype(int), iso["week"].astype(int), strict=True))), "halves": len(halves)}
    return {"events": int(len(clock)), "longs": int(clock["side"].eq(1).sum()), "shorts": int(clock["side"].eq(-1).sum()), "signed_episodes": int(len(clock)), "months": len(set(clock["entry_time"].dt.strftime("%Y-%m"))), "iso_weeks": sum(s["iso_weeks"] for s in split_stats.values()), "halves": len(set(clock["entry_time"].dt.strftime("%Y") + "H" + (((clock["entry_time"].dt.month - 1) // 6) + 1).astype(str))), "splits": split_stats}


def portfolio_stats(episodes: pd.DataFrame, segments: pd.DataFrame) -> dict[str, Any]:
    return {
        "signed_episodes": int(len(episodes)),
        "segments": int(len(segments)),
        "max_abs_target_exposure": 0.0 if segments.empty else float(segments["target_exposure"].abs().max()),
        "max_gross_exposure": 0.0 if segments.empty else float(segments["gross_exposure"].max()),
        "splits": {str(split): {"signed_episodes": int(len(group))} for split, group in episodes.groupby("split", sort=False)},
    }


def run(
    sleeve_clock_dir: Path = SLEEVE_CLOCK_DIR,
    portfolio_dir: Path = PORTFOLIO_DIR,
    result_path: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    preregistration = load_validated_preregistration()
    verified = verify_bound_component_artifacts()
    clocks = {component: load_full_component_clock(component) for component in COMPONENT_ORDER}
    sleeve_clock_dir.mkdir(parents=True, exist_ok=True)
    portfolio_dir.mkdir(parents=True, exist_ok=True)

    sleeves: dict[str, pd.DataFrame] = {}
    sleeve_records: dict[str, Any] = {}
    for base in BASE_ORDER:
        clock, diagnostics = build_sleeve_clock(base, clocks[base], clocks[VETO_COMPONENT])
        path = sleeve_clock_dir / f"{sleeve_id(base)}.csv.gz"
        _write_gzip_csv(clock, path)
        sleeves[base] = clock
        sleeve_records[base] = {"sleeve_id": sleeve_id(base), "base_component_id": base, "veto_component_id": VETO_COMPONENT, "weight": BASE_WEIGHTS[base], "clock": {"path": str(path), "sha256": sha256_file(path), "rows": len(clock)}, "construction_diagnostics": diagnostics, "source_stats": source_stats(clock)}

    transitions, segments, episodes = build_portfolio_schedules(sleeves)
    transition_path = portfolio_dir / f"{POLICY_ID}__transitions.csv.gz"
    segment_path = portfolio_dir / f"{POLICY_ID}__segments.csv.gz"
    episode_path = portfolio_dir / f"{POLICY_ID}__signed_episodes.csv.gz"
    _write_gzip_csv(transitions, transition_path)
    _write_gzip_csv(segments, segment_path)
    _write_gzip_csv(episodes, episode_path)

    stats = portfolio_stats(episodes, segments)
    if stats["max_gross_exposure"] > MAX_GROSS + 1e-12:
        raise RuntimeError(f"{POLICY_ID} max gross exceeds cap")
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "status": "shadow_only_train_distillation_not_formal_alpha",
        "preregistration": preregistration,
        "implementation": {"builder": {"path": "training/build_gross9_qtr_distill_clocks.py", "sha256": sha256_file(__file__)}},
        "components": {"base_order": list(BASE_ORDER), "veto_component": VETO_COMPONENT, "base_weights": {k: BASE_WEIGHTS[k] for k in BASE_ORDER}, "max_gross": MAX_GROSS},
        "verified_component_artifacts": verified,
        "construction": {
            "split_isolation": "each split is processed independently; veto lookback never crosses split labels",
            "operator": "for each base event t,s, latest same-split HVCQTR event with t-6h < veto.entry <= t; opposite suppresses, same/none keeps, no reverse",
            "reservation": "8h sleeve-local half-open reservation after veto materialization",
            "portfolio_netting": "target exposure transitions apply exits before entries at same timestamp; same direction sums; opposite direction nets; no priority",
            "grid": "5m timestamp grid",
        },
        "sleeves": sleeve_records,
        "portfolio_schedules": {
            "transitions": {"path": str(transition_path), "sha256": sha256_file(transition_path), "rows": len(transitions)},
            "segments": {"path": str(segment_path), "sha256": sha256_file(segment_path), "rows": len(segments)},
            "signed_episodes": {"path": str(episode_path), "sha256": sha256_file(episode_path), "rows": len(episodes)},
        },
        "portfolio_source_stats": stats,
        "evidence_boundary": {
            "component_clock_fields_opened": list(COMMON_CLOCK_FIELDS),
            "component_clock_files_authenticated_full_sha_and_row_count": True,
            "split_schedules_opened": True,
            "market_rows_opened": False,
            "entry_exit_prices_opened": False,
            "funding_opened": False,
            "returns_or_pnl_opened": False,
            "economic_outcomes_opened": False,
            "oos_outcomes_opened": False,
        },
        "decision": "materialized_shadow_distilled_clock_package",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleeve-clock-dir", type=Path, default=SLEEVE_CLOCK_DIR)
    parser.add_argument("--portfolio-dir", type=Path, default=PORTFOLIO_DIR)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    report = run(args.sleeve_clock_dir, args.portfolio_dir, args.result)
    print(json.dumps({"policy_id": report["policy_id"], "decision": report["decision"], "portfolio_source_stats": report["portfolio_source_stats"]}, indent=2))
