"""Build the outcome-blind SMCC-144 support and novelty decision."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from preprocessing.same_millisecond_cascade import BAR_COLUMNS
from training import preregister_same_millisecond_cascade as prereg


PREREGISTRATION_PATH = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_FILE_SHA256 = (
    "49fb04bc666f56f00efebf1f03c08e4a386f69b12fb12feeac22f9fde4ff9111"
)
PREREGISTRATION_MANIFEST_HASH = (
    "5628bdf5e9f6079ebb14585886738c9176ae869e249bcad89927775c7dafa302"
)
SOURCE_ACCESS_SEAL_PATH = Path(
    "results/same_millisecond_cascade_source_access_seal_2026-07-20.json"
)
# These three constants intentionally keep support execution disabled until the
# hash-only source seal is produced and committed.  No evaluator logic may
# change when they are populated.
SOURCE_ACCESS_SEAL_FILE_SHA256: str | None = None
EXPECTED_SOURCE_SHA256: str | None = None
EXPECTED_SOURCE_MANIFEST_SHA256: str | None = None
DEFAULT_SOURCE = Path(
    "data/binance_um_same_millisecond_cascade_btc_2020_2023/"
    "BTCUSDT_same_millisecond_5m_2020-01-01_2023-12-31.csv.gz"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "data/binance_um_same_millisecond_cascade_btc_2020_2023/build_manifest.json"
)
DEFAULT_OUTPUT = Path("results/same_millisecond_cascade_support_2026-07-20.json")
DEFAULT_CLOCK = Path("data/same_millisecond_cascade_clock_2020_2023.csv.gz")
SOURCE_START = pd.Timestamp("2020-01-01", tz="UTC")
SOURCE_END = pd.Timestamp("2024-01-01", tz="UTC")
FIVE_MINUTES: pd.Timedelta = cast(pd.Timedelta, pd.Timedelta("5min"))
CLOCK_COLUMNS = (
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "score",
    "threshold",
)


@dataclass(frozen=True)
class SupportConfig:
    source: str = str(DEFAULT_SOURCE)
    source_manifest: str = str(DEFAULT_SOURCE_MANIFEST)
    output: str = str(DEFAULT_OUTPUT)
    clock: str = str(DEFAULT_CLOCK)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _bool_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    text = series.astype("string").str.strip().str.lower()
    if not text.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"{label} contains non-boolean values")
    return text.isin(["true", "1"])


def load_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION_PATH) != PREREGISTRATION_FILE_SHA256:
        raise ValueError("SMCC preregistration file hash mismatch")
    payload = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise ValueError("SMCC preregistration manifest hash mismatch")
    return payload


def load_source_access_seal(
    cfg: SupportConfig,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if (
        SOURCE_ACCESS_SEAL_FILE_SHA256 is None
        or EXPECTED_SOURCE_SHA256 is None
        or EXPECTED_SOURCE_MANIFEST_SHA256 is None
    ):
        raise RuntimeError("SMCC source access is not hash-sealed yet")
    if sha256_file(SOURCE_ACCESS_SEAL_PATH) != SOURCE_ACCESS_SEAL_FILE_SHA256:
        raise ValueError("SMCC source access seal hash mismatch")
    seal = json.loads(SOURCE_ACCESS_SEAL_PATH.read_text(encoding="utf-8"))
    expected = {
        "preregistration_hash": payload["manifest_hash"],
        "source_path": cfg.source,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_manifest_path": cfg.source_manifest,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "outcomes_opened": False,
        "source_rows_parsed": 0,
    }
    if any(seal.get(key) != value for key, value in expected.items()):
        raise ValueError("SMCC source access seal contract mismatch")
    if sha256_file(cfg.source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("SMCC source bytes changed after access seal")
    if sha256_file(cfg.source_manifest) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("SMCC source manifest changed after access seal")
    return seal


def validate_source_frame(frame: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    if list(frame.columns) != list(BAR_COLUMNS):
        raise ValueError("SMCC source schema differs from the frozen builder schema")
    checked = frame.copy()
    dates = pd.to_datetime(_series(checked, "date"), utc=True, errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("SMCC source timestamps are invalid")
    expected = pd.date_range(SOURCE_START, SOURCE_END, freq="5min", inclusive="left")
    if len(dates) != len(expected) or not dates.reset_index(drop=True).equals(pd.Series(expected)):
        raise ValueError("SMCC source is not the frozen complete five-minute grid")
    checked["date"] = dates

    flag_columns = (
        "source_observed",
        "source_complete",
        "source_gap_day",
        "verified_zero_volume_empty",
        "post_gap_quarantine",
    )
    for column in flag_columns:
        checked[column] = _bool_series(_series(checked, column), column)
    numeric_columns = [column for column in BAR_COLUMNS if column not in {"date", *flag_columns}]
    numeric = checked.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("SMCC source contains non-finite numeric values")
    checked.loc[:, numeric_columns] = numeric

    observed = _series(checked, "source_observed").astype(bool)
    gap_day = _series(checked, "source_gap_day").astype(bool)
    verified_empty = _series(checked, "verified_zero_volume_empty").astype(bool)
    base_complete = (observed | verified_empty) & ~gap_day
    expected_post = (
        (~base_complete)
        .shift(1, fill_value=False)
        .rolling(window=int(payload["policy"]["post_gap_quarantine_bars"]), min_periods=1)
        .max()
        .astype(bool)
    )
    if not _series(checked, "post_gap_quarantine").astype(bool).equals(expected_post):
        raise ValueError("SMCC post-gap quarantine flags do not replay")
    expected_complete = base_complete & ~expected_post
    if not _series(checked, "source_complete").astype(bool).equals(expected_complete):
        raise ValueError("SMCC source-complete flags do not replay")

    expected_gap_days = set(payload["source_contract"]["required_source_gap_days"])
    actual_gap_days = set(
        _series(checked.loc[gap_day], "date").dt.strftime("%Y-%m-%d").unique().tolist()
    )
    if actual_gap_days != expected_gap_days:
        raise ValueError("SMCC source-gap day set differs from preregistration")

    quote = _series(checked, "quote_notional").astype(float)
    group_quote = _series(checked, "max_ms_quote_notional").astype(float)
    group_signed = _series(checked, "max_ms_signed_quote_notional").astype(float)
    expected_share = np.divide(
        group_quote,
        quote,
        out=np.zeros(len(checked), dtype=float),
        where=quote.to_numpy(float) != 0.0,
    )
    expected_coherence = np.divide(
        np.abs(group_signed.to_numpy(float)),
        group_quote.to_numpy(float),
        out=np.zeros(len(checked), dtype=float),
        where=group_quote.to_numpy(float) != 0.0,
    )
    expected_side = np.sign(group_signed.to_numpy(float)).astype(np.int8)
    expected_score = (
        expected_share
        * expected_coherence
        * np.clip(_series(checked, "max_ms_sweep_bp").to_numpy(float), 0.0, None)
    )
    identities = (
        np.allclose(_series(checked, "max_ms_notional_share"), expected_share, atol=1e-10, rtol=1e-10)
        and np.allclose(_series(checked, "max_ms_coherence"), expected_coherence, atol=1e-10, rtol=1e-10)
        and np.array_equal(_series(checked, "max_ms_side").to_numpy(np.int8), expected_side)
        and np.allclose(_series(checked, "max_ms_score"), expected_score, atol=1e-9, rtol=1e-9)
    )
    if not identities:
        raise ValueError("SMCC source feature identities do not replay")
    return checked


def load_source(
    source_path: str | Path,
    manifest_path: str | Path,
    payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], str, str]:
    source = Path(source_path)
    manifest_file = Path(manifest_path)
    manifest_hash = sha256_file(manifest_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("SMCC source manifest opened outcomes")
    if protocol.get("source_archive_manifest_sha256") != payload["source_contract"]["archive_manifest_sha256"]:
        raise ValueError("SMCC source archive contract mismatch")
    if protocol.get("source_audit_sha256") != payload["source_contract"]["source_audit_sha256"]:
        raise ValueError("SMCC source audit contract mismatch")
    source_hash = sha256_file(source)
    if manifest.get("combined_sha256") != source_hash:
        raise ValueError("SMCC combined source hash mismatch")
    if Path(str(manifest.get("combined_output"))) != source:
        raise ValueError("SMCC source path differs from its manifest")
    frame = pd.read_csv(source, compression="gzip")
    return validate_source_frame(frame, payload), manifest, source_hash, manifest_hash


def lagged_threshold(
    score: pd.Series,
    clean: pd.Series,
    *,
    window: int,
    minimum: int,
    quantile: float,
) -> pd.Series:
    if not 0.0 < quantile < 1.0:
        raise ValueError("SMCC quantile must be strictly between zero and one")
    return cast(
        pd.Series,
        score.astype(float)
        .where(clean.astype(bool))
        .shift(1)
        .rolling(window=window, min_periods=minimum)
        .quantile(quantile),
    )


def build_clock(frame: pd.DataFrame, policy: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    score = _series(frame, "max_ms_score").astype(float)
    clean = _series(frame, "source_complete").astype(bool)
    threshold = lagged_threshold(
        score,
        clean,
        window=int(policy["baseline_bars"]),
        minimum=int(policy["baseline_min_periods"]),
        quantile=float(policy["score_quantile"]),
    )
    eligible = (
        clean
        & _series(frame, "agg_trade_count").astype(float).ge(
            int(policy["minimum_bar_agg_trade_count"])
        )
        & _series(frame, "max_ms_event_count").astype(float).ge(
            int(policy["minimum_group_agg_trade_count"])
        )
        & _series(frame, "max_ms_coherence").astype(float).ge(
            float(policy["minimum_group_coherence"])
        )
        & _series(frame, "max_ms_side").astype(int).ne(0)
        & _series(frame, "max_ms_sweep_bp").astype(float).gt(0.0)
        & score.gt(0.0)
        & threshold.notna()
        & score.ge(threshold)
    )
    dates = pd.to_datetime(_series(frame, "date"), utc=True, errors="raise")
    delay = int(policy["execution_delay_bars"])
    hold = int(policy["hold_bars"])
    next_free = 0
    rows: list[dict[str, Any]] = []
    for position in np.flatnonzero(eligible.to_numpy(bool)):
        entry_position = int(position) + delay
        exit_position = entry_position + hold
        if entry_position < next_free or exit_position >= len(frame):
            continue
        entry_time = dates.iloc[entry_position]
        exit_time = dates.iloc[exit_position]
        if entry_time >= SOURCE_END or exit_time >= SOURCE_END:
            continue
        rows.append(
            {
                "decision_time": dates.iloc[int(position)] + FIVE_MINUTES,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": int(_series(frame, "max_ms_side").iloc[int(position)]),
                "score": float(score.iloc[int(position)]),
                "threshold": float(threshold.iloc[int(position)]),
            }
        )
        next_free = exit_position
    clock = pd.DataFrame(rows, columns=pd.Index(CLOCK_COLUMNS))
    if not clock.empty:
        entries = pd.to_datetime(_series(clock, "entry_time"), utc=True, errors="raise")
        exits = pd.to_datetime(_series(clock, "exit_time"), utc=True, errors="raise")
        if entries.duplicated().any() or not entries.is_monotonic_increasing:
            raise ValueError("SMCC scheduled entries are duplicate or unordered")
        if not (exits > entries).all():
            raise ValueError("SMCC scheduled exits do not follow entries")
        if not set(_series(clock, "side").astype(int).unique()).issubset({-1, 1}):
            raise ValueError("SMCC clock contains an invalid side")
    return clock, eligible


def support_summary(clock: pd.DataFrame, gates: dict[str, Any]) -> dict[str, Any]:
    entries = pd.to_datetime(_series(clock, "entry_time"), utc=True, errors="raise")
    sides = _series(clock, "side").astype(int)
    total = int(len(clock))
    by_year = {str(year): int(entries.dt.year.eq(year).sum()) for year in range(2020, 2024)}
    h1 = int(((entries >= pd.Timestamp("2023-01-01", tz="UTC")) & (entries < pd.Timestamp("2023-07-01", tz="UTC"))).sum())
    h2 = int(((entries >= pd.Timestamp("2023-07-01", tz="UTC")) & (entries < SOURCE_END)).sum())
    long_count = int(sides.eq(1).sum())
    short_count = int(sides.eq(-1).sum())
    long_share = float(long_count / total) if total else 0.0
    short_share = float(short_count / total) if total else 0.0
    month_counts = entries.dt.strftime("%Y-%m").value_counts().sort_index()
    max_month_share = float(month_counts.max() / total) if total else 1.0
    checks = {
        "total_min": total >= int(gates["total_2020_2023_min"]),
        "total_max": total <= int(gates["total_2020_2023_max"]),
        "each_year": min(by_year.values(), default=0) >= int(gates["each_calendar_year_min"]),
        "each_2023_half": min(h1, h2) >= int(gates["each_2023_half_min"]),
        "long_share": float(gates["each_side_share_min"]) <= long_share <= float(gates["each_side_share_max"]),
        "short_share": float(gates["each_side_share_min"]) <= short_share <= float(gates["each_side_share_max"]),
        "month_concentration": max_month_share <= float(gates["maximum_single_month_share"]),
    }
    return {
        "events": total,
        "by_year": by_year,
        "2023_halves": {"h1": h1, "h2": h2},
        "sides": {
            "long": long_count,
            "short": short_count,
            "long_share": long_share,
            "short_share": short_share,
        },
        "by_month": {str(key): int(value) for key, value in month_counts.items()},
        "maximum_month_share": max_month_share,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _utc_entries(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="raise")
    if parsed.isna().any() or parsed.duplicated().any():
        raise ValueError(f"{label} contains missing or duplicate entries")
    return parsed.sort_values().reset_index(drop=True)


def one_to_one_matches(
    primary: pd.Series,
    comparator: pd.Series,
    *,
    tolerance: pd.Timedelta,
) -> int:
    if tolerance < pd.Timedelta(0):
        raise ValueError("SMCC novelty tolerance cannot be negative")
    left = _utc_entries(primary, "primary novelty clock").astype("int64").to_numpy(np.int64)
    right = _utc_entries(comparator, "comparator novelty clock").astype("int64").to_numpy(np.int64)
    tolerance_ns = int(tolerance.value)
    unused = np.ones(len(right), dtype=bool)
    matches = 0
    for value in left:
        candidates = np.flatnonzero(unused & (np.abs(right - value) <= tolerance_ns))
        if not len(candidates):
            continue
        distances = np.abs(right[candidates] - value)
        order = np.lexsort((right[candidates], distances))
        chosen = int(candidates[int(order[0])])
        unused[chosen] = False
        matches += 1
    return matches


def overlap_metrics(primary: pd.Series, comparator: pd.Series) -> dict[str, Any]:
    primary_checked = _utc_entries(primary, "primary overlap clock")
    comparator_checked = _utc_entries(comparator, "comparator overlap clock")

    def calculate(tolerance: pd.Timedelta) -> dict[str, Any]:
        matches = one_to_one_matches(
            primary_checked,
            comparator_checked,
            tolerance=tolerance,
        )
        union = len(primary_checked) + len(comparator_checked) - matches
        return {
            "matches": matches,
            "jaccard": float(matches / union) if union else 1.0,
            "primary_containment": float(matches / len(primary_checked)) if len(primary_checked) else 1.0,
            "comparator_containment": float(matches / len(comparator_checked)) if len(comparator_checked) else 1.0,
        }

    return {
        "primary_events": int(len(primary_checked)),
        "comparator_events": int(len(comparator_checked)),
        "exact": calculate(cast(pd.Timedelta, pd.Timedelta(0))),
        "tolerant_12_bars": calculate(12 * FIVE_MINUTES),
    }


def _member_entries(frame: pd.DataFrame, spec: dict[str, Any]) -> dict[str, pd.Series]:
    expected_members = [str(value) for value in spec["members"]]
    member_column = spec.get("member_column")
    if member_column is None:
        if len(expected_members) != 1:
            raise ValueError("ungrouped SMCC comparator must have one member")
        member_frames = {expected_members[0]: frame}
    else:
        if member_column not in frame.columns:
            raise ValueError(f"comparator is missing member column {member_column}")
        actual_members = set(_series(frame, member_column).astype(str).unique())
        if actual_members != set(expected_members):
            raise ValueError("comparator member registry changed")
        member_frames = {
            member: frame.loc[_series(frame, member_column).astype(str).eq(member)]
            for member in expected_members
        }
    output: dict[str, pd.Series] = {}
    for member, member_frame in member_frames.items():
        entry_column = spec.get("entry_column")
        if entry_column is None:
            if spec.get("derived_entry") != "signal_date + 2 completed five-minute bars":
                raise ValueError("unknown derived comparator entry contract")
            if "signal_date" not in member_frame.columns:
                raise ValueError("derived comparator entry is missing signal_date")
            entries = pd.to_datetime(_series(member_frame, "signal_date"), utc=True, errors="raise") + 2 * FIVE_MINUTES
        else:
            if entry_column not in member_frame.columns:
                raise ValueError(f"comparator is missing entry column {entry_column}")
            entries = pd.to_datetime(_series(member_frame, entry_column), utc=True, errors="raise")
        output[member] = _utc_entries(entries, f"{spec['family']}:{member}")
    return output


def novelty_report(clock: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    primary_entries = _utc_entries(_series(clock, "entry_time"), "SMCC primary entries")
    contract = payload["novelty_gates"]
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for spec in contract["comparator_registry"]:
        try:
            path = Path(spec["path"])
            if sha256_file(path) != spec["sha256"]:
                raise ValueError("artifact hash mismatch")
            frame = pd.read_csv(path)
            members = _member_entries(frame, spec)
            coverage_start = pd.Timestamp(spec["coverage"][0], tz="UTC")
            coverage_end = pd.Timestamp(spec["coverage"][1], tz="UTC")
            primary_coverage = primary_entries.loc[
                (primary_entries >= coverage_start) & (primary_entries < coverage_end)
            ]
            for member, entries in members.items():
                comparator_coverage = entries.loc[
                    (entries >= coverage_start) & (entries < coverage_end)
                ]
                metrics = overlap_metrics(primary_coverage, comparator_coverage)
                checks = {
                    "exact_jaccard": metrics["exact"]["jaccard"]
                    <= float(contract["exact_entry_jaccard_max"]),
                    "tolerant_jaccard": metrics["tolerant_12_bars"]["jaccard"]
                    <= float(contract["tolerant_one_to_one_jaccard_max"]),
                    "primary_containment": metrics["tolerant_12_bars"]["primary_containment"]
                    <= float(contract["primary_containment_max"]),
                }
                reports.append(
                    {
                        "family": spec["family"],
                        "member": member,
                        "coverage": spec["coverage"],
                        "metrics": metrics,
                        "checks": checks,
                        "passed": bool(all(checks.values())),
                    }
                )
        except Exception as exc:  # fail closed into a deterministic rejection artifact
            errors.append(f"{spec['family']}: {type(exc).__name__}: {exc}")

    dense = contract["dense_bafr"]
    dense_report: dict[str, Any]
    try:
        dense_path = Path(dense["path"])
        if sha256_file(dense_path) != dense["sha256"]:
            raise ValueError("artifact hash mismatch")
        dense_frame = pd.read_csv(dense_path)
        dense_entries = _utc_entries(
            _series(dense_frame, dense["entry_column"]), "BAFR dense clock"
        )
        start = pd.Timestamp(dense["coverage"][0], tz="UTC")
        end = pd.Timestamp(dense["coverage"][1], tz="UTC")
        dense_metrics = overlap_metrics(
            primary_entries.loc[(primary_entries >= start) & (primary_entries < end)],
            dense_entries.loc[(dense_entries >= start) & (dense_entries < end)],
        )
        dense_report = {
            "metrics": {
                "primary_events": dense_metrics["primary_events"],
                "comparator_events": dense_metrics["comparator_events"],
                "exact": dense_metrics["exact"],
            },
            "gated": False,
        }
    except Exception as exc:
        errors.append(f"BAFR: {type(exc).__name__}: {exc}")
        dense_report = {"error": errors[-1], "gated": False}
    return {
        "status": "EVALUATED",
        "sparse_members": reports,
        "dense_bafr_report": dense_report,
        "errors": errors,
        "passed": bool(not errors and reports and all(item["passed"] for item in reports)),
    }


def _clock_bytes(clock: pd.DataFrame) -> bytes:
    serialized = clock.copy()
    for column in ("decision_time", "entry_time", "exit_time"):
        serialized[column] = pd.to_datetime(
            _series(serialized, column), utc=True, errors="raise"
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    csv_bytes = serialized.loc[:, list(CLOCK_COLUMNS)].to_csv(
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0, compresslevel=6) as handle:
        handle.write(csv_bytes)
    return buffer.getvalue()


def _write_once(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite frozen SMCC artifact: {path}")
        return "verified_existing"
    with path.open("xb") as handle:
        handle.write(content)
    return "created"


def run_support(cfg: SupportConfig) -> dict[str, Any]:
    payload = load_preregistration()
    load_source_access_seal(cfg, payload)
    frame, source_manifest, source_hash, source_manifest_hash = load_source(
        cfg.source,
        cfg.source_manifest,
        payload,
    )
    clock, eligible = build_clock(frame, payload["policy"])
    support = support_summary(clock, payload["support_gates"])
    if support["passed"]:
        novelty = novelty_report(clock, payload)
    else:
        novelty = {
            "status": "SKIPPED_SUPPORT_FAILURE",
            "sparse_members": [],
            "dense_bafr_report": None,
            "errors": [],
            "passed": False,
        }
    decision = "PASS_SUPPORT" if support["passed"] and novelty["passed"] else "REJECT_NO_REPAIR"
    clock_content = _clock_bytes(clock)
    clock_hash = hashlib.sha256(clock_content).hexdigest()
    result = {
        "protocol_version": "same_millisecond_cascade_support_v1",
        "preregistration_hash": payload["manifest_hash"],
        "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256,
        "source_sha256": source_hash,
        "source_manifest_sha256": source_manifest_hash,
        "outcomes_opened": False,
        "post_entry_market_rows_read": 0,
        "funding_rows_read": 0,
        "source_counts": {
            "rows": int(len(frame)),
            "source_observed": int(_series(frame, "source_observed").sum()),
            "source_complete": int(_series(frame, "source_complete").sum()),
            "source_gap_day_rows": int(_series(frame, "source_gap_day").sum()),
            "verified_zero_volume_empty": int(
                _series(frame, "verified_zero_volume_empty").sum()
            ),
            "post_gap_quarantine": int(_series(frame, "post_gap_quarantine").sum()),
            "raw_eligible_bars": int(eligible.sum()),
            "scheduled_events": int(len(clock)),
        },
        "support_gates": support,
        "novelty": novelty,
        "decision": decision,
        "clock_path": cfg.clock,
        "clock_sha256": clock_hash,
        "source_manifest_protocol": source_manifest["protocol"],
    }
    json_content = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    clock_status = _write_once(Path(cfg.clock), clock_content)
    result_status = _write_once(Path(cfg.output), json_content)
    return {
        **result,
        "write_status": {"clock": clock_status, "result": result_status},
    }


def parse_args() -> SupportConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--clock", default=str(DEFAULT_CLOCK))
    return SupportConfig(**vars(parser.parse_args()))


def main() -> None:
    result = run_support(parse_args())
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "events": result["support_gates"]["events"],
                "clock_sha256": result["clock_sha256"],
                "outcomes_opened": False,
                "write_status": result["write_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
