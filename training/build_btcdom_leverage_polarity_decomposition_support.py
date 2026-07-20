"""Build the frozen outcome-blind DLPD-12 source-support verdict."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import preregister_btcdom_leverage_polarity_decomposition as dlpd


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path(
    "results/btcdom_leverage_polarity_decomposition_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "6d5ba05072d7e1677239e2a6dba9ec8dab79bfb7a7e25fe89b3396e269adc9ff"
)
BUILDER = Path("training/build_btcdom_leverage_polarity_decomposition_support.py")
DEFAULT_CLOCKS = Path("data/btcdom_leverage_polarity_decomposition_clocks_2022_2023.csv.gz")
DEFAULT_RESULT = Path(
    "results/btcdom_leverage_polarity_decomposition_support_2026-07-20.json"
)
FORBIDDEN_EVENT_TOKENS = (
    "contract_open",
    "contract_high",
    "contract_low",
    "contract_close",
    "return",
    "label",
    "pnl",
    "funding",
    "cagr",
    "drawdown",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_preregistration(path: str | Path = PREREGISTRATION) -> dict[str, Any]:
    if sha256_file(path) != PREREGISTRATION_SHA256:
        raise ValueError("DLPD preregistration file hash mismatch")
    payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != dlpd.canonical_hash(core):
        raise ValueError("DLPD preregistration manifest hash mismatch")
    if payload.get("candidate") != dlpd.POLICY_ID:
        raise ValueError("DLPD policy identifier changed")
    for key in (
        "outcomes_opened",
        "outcome_sources_opened",
        "post_2023_source_rows_opened",
        "real_event_incidence_opened",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"DLPD unopened boundary changed: {key}")
    expected_cfg = {
        **dlpd.asdict(dlpd.FROZEN_CONFIG),
        "support_years": list(dlpd.FROZEN_CONFIG.support_years),
    }
    if payload.get("policy", {}).get("config") != expected_cfg:
        raise ValueError("DLPD support configuration changed")
    if tuple(payload.get("source_only_controls", ())) != dlpd.SOURCE_ONLY_CONTROLS:
        raise ValueError("DLPD control set changed")
    return payload


def build_clocks(source: pd.DataFrame) -> pd.DataFrame:
    states = dlpd.signal_states(source)
    parts = [
        dlpd.schedule(source, states, control=control, year=year)
        for control in dlpd.CONTROLS
        for year in dlpd.FROZEN_CONFIG.support_years
    ]
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame(columns=dlpd.EVENT_COLUMNS)
    clocks = pd.concat(nonempty, ignore_index=True).sort_values(
        ["control", "entry_time", "side"], kind="mergesort"
    ).reset_index(drop=True)
    if tuple(clocks.columns) != dlpd.EVENT_COLUMNS:
        raise ValueError("DLPD clock schema changed")
    lowered = tuple(column.lower() for column in clocks.columns)
    if any(token in column for token in FORBIDDEN_EVENT_TOKENS for column in lowered):
        raise ValueError("DLPD source clock retained an outcome field")
    return clocks


def support_summary(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {
            "events": 0,
            "long": 0,
            "short": 0,
            "long_share": 0.0,
            "short_share": 0.0,
            "month_counts": {},
            "quarter_counts": {},
            "max_month_share": 0.0,
            "first_entry": None,
            "last_exit": None,
        }
    side = cast(pd.Series, events["side"])
    entries = pd.to_datetime(events["entry_time"], utc=True)
    exits = pd.to_datetime(events["exit_time"], utc=True)
    if not side.isin((-1, 1)).all():
        raise ValueError("DLPD support contains a non-directional event")
    count = int(len(events))
    long_count = int(side.eq(1).sum())
    short_count = int(side.eq(-1).sum())
    months = entries.dt.strftime("%Y-%m").value_counts().sort_index()
    quarter_labels = (
        entries.dt.year.astype(str)
        + "Q"
        + (((entries.dt.month - 1) // 3) + 1).astype(str)
    )
    quarters = quarter_labels.value_counts().sort_index()
    return {
        "events": count,
        "long": long_count,
        "short": short_count,
        "long_share": float(long_count / count),
        "short_share": float(short_count / count),
        "month_counts": {str(key): int(value) for key, value in months.items()},
        "quarter_counts": {str(key): int(value) for key, value in quarters.items()},
        "max_month_share": float(months.max() / count),
        "first_entry": cast(pd.Timestamp, entries.min()).isoformat(),
        "last_exit": cast(pd.Timestamp, exits.max()).isoformat(),
    }


def nearest_share(
    left: pd.DatetimeIndex,
    right: pd.DatetimeIndex,
    *,
    hours: int,
) -> float:
    if len(left) == 0 or len(right) == 0:
        return 0.0
    left_ns = np.asarray(left.asi8, dtype=np.int64)
    right_ns = np.sort(np.asarray(right.asi8, dtype=np.int64))
    insertion = np.searchsorted(right_ns, left_ns)
    threshold = int(pd.Timedelta(hours=hours).value)
    matched = np.zeros(len(left_ns), dtype=bool)
    for positions in (insertion - 1, insertion):
        valid = (positions >= 0) & (positions < len(right_ns))
        distance = np.full(len(left_ns), np.iinfo(np.int64).max, dtype=np.int64)
        distance[valid] = np.abs(left_ns[valid] - right_ns[positions[valid]])
        matched |= distance <= threshold
    return float(matched.mean())


def novelty_metrics(
    primary: pd.DatetimeIndex,
    comparator: pd.DatetimeIndex,
    *,
    year: int,
    hours: int,
) -> dict[str, Any]:
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    left = primary[(primary >= start) & (primary < end)].unique()
    right = comparator[(comparator >= start) & (comparator < end)].unique()
    intersection = left.intersection(right)
    union = left.union(right)
    primary_near = nearest_share(left, right, hours=hours)
    comparator_near = nearest_share(right, left, hours=hours)
    return {
        "coverage": [start.isoformat(), end.isoformat()],
        "primary_events": int(len(left)),
        "comparator_events": int(len(right)),
        "exact_intersection": int(len(intersection)),
        "exact_jaccard": float(len(intersection) / len(union)) if len(union) else 0.0,
        "near_hours": int(hours),
        "primary_near_share": primary_near,
        "comparator_near_share": comparator_near,
        "max_bidirectional_near_share": max(primary_near, comparator_near),
    }


def _load_csv_clock(contract: dict[str, Any]) -> pd.DatetimeIndex:
    frame = pd.read_csv(_resolve(contract["clock"]))
    if "candidate" in frame.columns:
        frame = cast(pd.DataFrame, frame[frame["candidate"].eq(contract["candidate"])].copy())
    if contract.get("control") is not None:
        if "control" not in frame.columns:
            raise ValueError(f"comparator lacks control column: {contract['candidate']}")
        frame = cast(pd.DataFrame, frame[frame["control"].eq(contract["control"])].copy())
    field = str(contract["entry_field"])
    if field not in frame.columns:
        raise ValueError(f"comparator lacks entry field: {contract['candidate']}")
    values = pd.to_datetime(frame[field], utc=True)
    if values.duplicated().any():
        raise ValueError(f"comparator has duplicate entries: {contract['candidate']}")
    return pd.DatetimeIndex(values).sort_values()


def _load_json_clock(contract: dict[str, Any]) -> pd.DatetimeIndex:
    payload = json.loads(_resolve(contract["clock"]).read_text(encoding="utf-8"))
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError(f"comparator JSON lacks events: {contract['candidate']}")
    field = str(contract["entry_field"])
    values = pd.to_datetime([event[field] for event in events], utc=True)
    if values.duplicated().any():
        raise ValueError(f"comparator has duplicate entries: {contract['candidate']}")
    return pd.DatetimeIndex(values).sort_values()


def load_comparator_clocks(
    preregistration: dict[str, Any],
) -> tuple[dict[str, pd.DatetimeIndex], int]:
    output: dict[str, pd.DatetimeIndex] = {}
    rows = 0
    for contract in preregistration["support_comparators"]:
        if sha256_file(contract["clock"]) != contract["clock_sha256"]:
            raise ValueError(f"DLPD comparator clock hash mismatch: {contract['candidate']}")
        if sha256_file(contract["support"]) != contract["support_sha256"]:
            raise ValueError(f"DLPD comparator support hash mismatch: {contract['candidate']}")
        if contract["format"] == "csv":
            entries = _load_csv_clock(contract)
        elif contract["format"] == "json_events":
            entries = _load_json_clock(contract)
        else:
            raise ValueError(f"unknown comparator format: {contract['format']}")
        output[str(contract["candidate"])] = entries
        rows += len(entries)
    if set(output) != {item["candidate"] for item in dlpd.COMPARATORS}:
        raise ValueError("DLPD comparator set changed")
    return output, rows


def support_checks(
    summaries: dict[str, dict[str, dict[str, Any]]],
    novelty: dict[str, dict[str, Any]],
    preregistration: dict[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    gate = preregistration["support_gate"]
    checks: dict[str, bool] = {}
    for year in gate["years"]:
        name = str(year)
        summary = summaries["primary"][name]
        checks[f"{name}_minimum_events"] = summary["events"] >= int(
            gate["minimum_events_per_year"]
        )
        checks[f"{name}_long_share"] = summary["long_share"] >= float(
            gate["minimum_side_share"]
        )
        checks[f"{name}_short_share"] = summary["short_share"] >= float(
            gate["minimum_side_share"]
        )
        checks[f"{name}_month_concentration"] = summary["max_month_share"] <= float(
            gate["maximum_month_share"]
        )
        for quarter in range(1, 5):
            key = f"{year}Q{quarter}"
            checks[f"{key}_minimum_events"] = int(
                summary["quarter_counts"].get(key, 0)
            ) >= int(gate["minimum_events_per_quarter"])
    for candidate, metrics in novelty.items():
        checks[f"{candidate}_exact_jaccard"] = metrics["exact_jaccard"] <= float(
            gate["maximum_exact_entry_jaccard"]
        )
        checks[f"{candidate}_near_containment"] = metrics[
            "max_bidirectional_near_share"
        ] <= float(gate["maximum_bidirectional_near_share"])
    failures = [name for name, passed in checks.items() if not passed]
    return checks, failures


def _clock_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n")
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output,
        mode="wb",
        filename="",
        compresslevel=9,
        mtime=0,
    ) as stream:
        stream.write(text.encode("utf-8"))
    return output.getvalue()


def write_frozen_bytes(path: str | Path, payload: bytes) -> None:
    target = _resolve(path)
    if target.exists() and target.read_bytes() != payload:
        raise FileExistsError(f"existing frozen DLPD artifact differs: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def write_frozen_json(path: str | Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    write_frozen_bytes(path, encoded)


def build(
    *,
    preregistration_path: str | Path = PREREGISTRATION,
    clocks_path: str | Path = DEFAULT_CLOCKS,
    result_path: str | Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    prereg = load_preregistration(preregistration_path)
    source, source_audit = dlpd.load_source()
    clocks = build_clocks(source)
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for control in dlpd.CONTROLS:
        summaries[control] = {}
        for year in dlpd.FROZEN_CONFIG.support_years:
            subset = cast(
                pd.DataFrame,
                clocks[
                    clocks["control"].eq(control) & clocks["split"].eq(str(year))
                ].copy(),
            )
            summaries[control][str(year)] = support_summary(subset)

    primary = cast(pd.DataFrame, clocks[clocks["control"].eq("primary")].copy())
    primary_entries = pd.DatetimeIndex(pd.to_datetime(primary["entry_time"], utc=True))
    comparators, comparator_rows = load_comparator_clocks(prereg)
    gate = prereg["support_gate"]
    novelty = {
        candidate: novelty_metrics(
            primary_entries,
            entries,
            year=int(gate["novelty_year"]),
            hours=int(gate["novelty_hours"]),
        )
        for candidate, entries in comparators.items()
    }
    checks, failures = support_checks(summaries, novelty, prereg)

    clock_payload = _clock_bytes(clocks)
    write_frozen_bytes(clocks_path, clock_payload)
    core: dict[str, Any] = {
        "protocol_version": "btcdom_leverage_polarity_decomposition_support_v1",
        "as_of_date": "2026-07-20",
        "candidate": dlpd.POLICY_ID,
        "preregistration": str(preregistration_path),
        "preregistration_sha256": sha256_file(preregistration_path),
        "preregistration_manifest_hash": prereg["manifest_hash"],
        "builder": str(BUILDER),
        "builder_sha256": sha256_file(BUILDER),
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "btc_execution_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "post_2023_source_rows_loaded": 0,
        "real_event_incidence_opened": True,
        "source_rows_loaded": int(len(source)),
        "source_audit": source_audit,
        "comparator_clock_rows_loaded": int(comparator_rows),
        "clock_path": str(clocks_path),
        "clock_sha256": hashlib.sha256(clock_payload).hexdigest(),
        "clock_rows_all_controls": int(len(clocks)),
        "primary_clock_rows": int(len(primary)),
        "support": summaries["primary"],
        "control_support": summaries,
        "novelty": novelty,
        "support_checks": checks,
        "support_failures": failures,
        "support_passed": not failures,
        "next_action": (
            "freeze strict 2022 evaluator before opening outcomes"
            if not failures
            else "reject DLPD-12 before all market outcomes"
        ),
    }
    core["manifest_hash"] = dlpd.canonical_hash(core)
    write_frozen_json(result_path, core)
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--clocks", default=str(DEFAULT_CLOCKS))
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    args = parser.parse_args()
    result = build(
        preregistration_path=args.preregistration,
        clocks_path=args.clocks,
        result_path=args.result,
    )
    print(
        json.dumps(
            {
                "support_passed": result["support_passed"],
                "support_failures": result["support_failures"],
                "support": result["support"],
                "novelty": result["novelty"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
