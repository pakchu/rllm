"""Build the outcome-blind SDDR-12 source-support clock and verdict."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import preregister_stablecoin_denominator_dislocation as sddr


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path(
    "results/stablecoin_denominator_dislocation_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "0db69de6f278d37e8bff09b5843127dc326b75eec2722d9b46d3882434c19280"
)
BUILDER = Path("training/build_stablecoin_denominator_dislocation_support.py")
DEFAULT_CLOCKS = Path("data/stablecoin_denominator_dislocation_clocks_2023.csv.gz")
DEFAULT_RESULT = Path(
    "results/stablecoin_denominator_dislocation_support_2026-07-20.json"
)
SUPPORT_START = pd.Timestamp("2023-08-04T08:00:00Z")
SUPPORT_END = pd.Timestamp("2024-01-01T00:00:00Z")
CONTROLS = (
    "primary",
    "no_disagreement",
    "usdc_only",
    "fdusd_only",
    "stale_1h",
)
FORBIDDEN_CLOCK_TOKENS = (
    "price",
    "open",
    "high",
    "low",
    "close",
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


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_payload() -> dict[str, Any]:
    payload = asdict(sddr.FROZEN_CONFIG)
    payload["full_signal_months"] = list(sddr.FROZEN_CONFIG.full_signal_months)
    return payload


def load_preregistration(path: str | Path = PREREGISTRATION) -> dict[str, Any]:
    if sha256_file(path) != PREREGISTRATION_SHA256:
        raise ValueError("SDDR preregistration file hash mismatch")
    payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise ValueError("SDDR preregistration manifest hash mismatch")
    if payload.get("candidate") != sddr.POLICY_ID:
        raise ValueError("SDDR policy identifier changed")
    for key in (
        "outcomes_opened",
        "outcome_sources_opened",
        "post_2023_source_rows_opened",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"SDDR outcome-blind boundary changed: {key}")
    if payload.get("policy", {}).get("config") != _config_payload():
        raise ValueError("SDDR support configuration changed after freeze")
    if tuple(payload.get("source_only_controls", ())) != CONTROLS[1:]:
        raise ValueError("SDDR source-only controls changed after freeze")
    return payload


def build_clocks(source: pd.DataFrame) -> pd.DataFrame:
    states = sddr.signal_states(source)
    parts = [
        sddr.schedule(source, states, control=control)
        for control in CONTROLS
    ]
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame(columns=sddr.EVENT_COLUMNS)
    clocks = pd.concat(nonempty, ignore_index=True)
    clocks = clocks.sort_values(
        ["control", "entry_time", "side"], kind="mergesort"
    ).reset_index(drop=True)
    if tuple(clocks.columns) != sddr.EVENT_COLUMNS:
        raise ValueError("SDDR clock schema changed")
    lowered = tuple(column.lower() for column in clocks.columns)
    if any(token in column for token in FORBIDDEN_CLOCK_TOKENS for column in lowered):
        raise ValueError("SDDR source clock retained an outcome field")
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
            "max_month_share": 0.0,
            "first_entry": None,
            "last_exit": None,
        }
    side = cast(pd.Series, events["side"])
    entries = pd.to_datetime(events["entry_time"], utc=True)
    exits = pd.to_datetime(events["exit_time"], utc=True)
    if not side.isin((-1, 1)).all():
        raise ValueError("SDDR support contains a non-directional event")
    count = int(len(events))
    long_count = int(side.eq(1).sum())
    short_count = int(side.eq(-1).sum())
    month_counts = entries.dt.strftime("%Y-%m").value_counts().sort_index()
    return {
        "events": count,
        "long": long_count,
        "short": short_count,
        "long_share": float(long_count / count),
        "short_share": float(short_count / count),
        "month_counts": {
            str(month): int(value) for month, value in month_counts.items()
        },
        "max_month_share": float(month_counts.max() / count),
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
    for offsets in (insertion - 1, insertion):
        valid = (offsets >= 0) & (offsets < len(right_ns))
        distance = np.full(len(left_ns), np.iinfo(np.int64).max, dtype=np.int64)
        distance[valid] = np.abs(left_ns[valid] - right_ns[offsets[valid]])
        matched |= distance <= threshold
    return float(matched.mean())


def novelty_metrics(
    primary: pd.DatetimeIndex,
    comparator: pd.DatetimeIndex,
    *,
    hours: int,
) -> dict[str, Any]:
    left = primary[(primary >= SUPPORT_START) & (primary < SUPPORT_END)].unique()
    right = comparator[
        (comparator >= SUPPORT_START) & (comparator < SUPPORT_END)
    ].unique()
    intersection = left.intersection(right)
    union = left.union(right)
    primary_near = nearest_share(left, right, hours=hours)
    comparator_near = nearest_share(right, left, hours=hours)
    return {
        "coverage": [SUPPORT_START.isoformat(), SUPPORT_END.isoformat()],
        "primary_events": int(len(left)),
        "comparator_events": int(len(right)),
        "exact_intersection": int(len(intersection)),
        "exact_jaccard": float(len(intersection) / len(union)) if len(union) else 0.0,
        "near_hours": int(hours),
        "primary_near_share": primary_near,
        "comparator_near_share": comparator_near,
        "max_bidirectional_near_share": max(primary_near, comparator_near),
    }


def load_comparator_clocks(prereg: dict[str, Any]) -> tuple[dict[str, pd.DatetimeIndex], int]:
    contract = prereg["support_comparators"]
    clock_path = Path(contract["clock"])
    support_path = Path(contract["support"])
    if sha256_file(clock_path) != contract["clock_sha256"]:
        raise ValueError("SQFD comparator clock hash mismatch")
    if sha256_file(support_path) != contract["support_sha256"]:
        raise ValueError("SQFD comparator support hash mismatch")
    support = json.loads(_resolve(support_path).read_text(encoding="utf-8"))
    if support.get("outcomes_opened") is not False:
        raise ValueError("SQFD comparator support opened outcomes")
    frame = cast(
        pd.DataFrame,
        pd.read_csv(
            _resolve(clock_path),
            usecols=["candidate", "control", "entry_time"],
        ),
    )
    frame = cast(
        pd.DataFrame,
        frame[
            frame["candidate"].eq("SQFD-6")
            & frame["control"].isin(contract["controls"])
        ].copy(),
    )
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    if frame[["control", "entry_time"]].duplicated().any():
        raise ValueError("SQFD comparator contains a duplicate control timestamp")
    clocks = {
        control: pd.DatetimeIndex(
            frame.loc[frame["control"].eq(control), "entry_time"]
        ).sort_values()
        for control in contract["controls"]
    }
    if set(clocks) != set(sddr.COMPARATOR_CONTROLS):
        raise ValueError("SQFD comparator control set changed")
    return clocks, int(len(frame))


def support_checks(
    summary: dict[str, Any],
    novelty: dict[str, dict[str, Any]],
    prereg: dict[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    gate = prereg["support_gate"]
    checks: dict[str, bool] = {
        "minimum_events": summary["events"] >= int(gate["minimum_events"]),
        "long_share": summary["long_share"] >= float(gate["minimum_side_share"]),
        "short_share": summary["short_share"] >= float(gate["minimum_side_share"]),
        "month_concentration": summary["max_month_share"]
        <= float(gate["maximum_month_share"]),
    }
    for month in gate["full_signal_months"]:
        checks[f"{month}_minimum_events"] = int(
            summary["month_counts"].get(month, 0)
        ) >= int(gate["minimum_events_per_full_signal_month"])
    for control, metrics in novelty.items():
        checks[f"{control}_exact_jaccard"] = metrics["exact_jaccard"] <= float(
            gate["maximum_exact_entry_jaccard"]
        )
        checks[f"{control}_near_containment"] = metrics[
            "max_bidirectional_near_share"
        ] <= float(gate["maximum_novelty_containment"])
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
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"existing frozen artifact differs: {path}")
        return
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
    preregistration: str | Path = PREREGISTRATION,
    clocks_path: str | Path = DEFAULT_CLOCKS,
    result_path: str | Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    prereg = load_preregistration(preregistration)
    source, source_audit = sddr.load_source()
    clocks = build_clocks(source)
    primary = cast(
        pd.DataFrame,
        clocks[clocks["control"].eq("primary")].copy(),
    )
    summaries = {
        control: support_summary(
            cast(pd.DataFrame, clocks[clocks["control"].eq(control)].copy())
        )
        for control in CONTROLS
    }
    comparators, comparator_rows = load_comparator_clocks(prereg)
    primary_entries = pd.DatetimeIndex(
        pd.to_datetime(primary["entry_time"], utc=True)
    )
    novelty = {
        control: novelty_metrics(
            primary_entries,
            entries,
            hours=int(prereg["support_gate"]["novelty_containment_hours"]),
        )
        for control, entries in comparators.items()
    }
    checks, failures = support_checks(summaries["primary"], novelty, prereg)

    clock_payload = _clock_bytes(clocks)
    write_frozen_bytes(clocks_path, clock_payload)
    core: dict[str, Any] = {
        "protocol_version": "stablecoin_denominator_dislocation_support_v1",
        "as_of_date": "2026-07-20",
        "candidate": sddr.POLICY_ID,
        "preregistration": str(preregistration),
        "preregistration_sha256": sha256_file(preregistration),
        "preregistration_manifest_hash": prereg["manifest_hash"],
        "builder": str(BUILDER),
        "builder_sha256": sha256_file(BUILDER),
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "btc_execution_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "post_2023_sddr_source_rows_loaded": 0,
        "source_rows_loaded": int(len(source)),
        "source_audit": source_audit,
        "comparator_clock_rows_loaded": comparator_rows,
        "clock_path": str(clocks_path),
        "clock_sha256": hashlib.sha256(clock_payload).hexdigest(),
        "clock_rows_all_controls": int(len(clocks)),
        "primary_clock_rows": int(len(primary)),
        "support": summaries["primary"],
        "control_support": summaries,
        "novelty": novelty,
        "checks": checks,
        "failed_checks": failures,
        "support_passed": not failures,
        "advance_to_frozen_outcome_evaluator": not failures,
        "sealed_outcome_stages": [
            "train_2023",
            "test_2024",
            "eval_2025",
            "final_2026h1",
        ],
        "rejection_action": (
            "freeze strict evaluator before opening train_2023 outcomes"
            if not failures
            else "reject SDDR-12 without opening execution OHLC or funding"
        ),
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    write_frozen_json(result_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--clocks", default=str(DEFAULT_CLOCKS))
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    args = parser.parse_args()
    report = build(
        preregistration=args.preregistration,
        clocks_path=args.clocks,
        result_path=args.result,
    )
    print(
        json.dumps(
            {
                "support_passed": report["support_passed"],
                "failed_checks": report["failed_checks"],
                "support": report["support"],
                "novelty": report["novelty"],
                "clock_sha256": report["clock_sha256"],
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
