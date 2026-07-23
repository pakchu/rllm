"""Build outcome-blind DCLB-864 source-support and novelty evidence."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from training import (
    preregister_dollar_collateral_liquidity_bank_relay_v2 as prereg,
)


PROTOCOL_VERSION = "dollar_collateral_liquidity_bank_relay_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/build_dollar_collateral_liquidity_bank_relay_support.py"
)
TEST_PATH = Path(
    "tests/test_build_dollar_collateral_liquidity_bank_relay_support.py"
)
IMPLEMENTATION_CONTRACT = Path(
    "docs/dclb-source-support-implementation-contract-2026-07-24.md"
)
IMPLEMENTATION_CONTRACT_SHA256 = (
    "e95743a9673cff2acda08e48663b00b47085a6db7a650fac4ab15bd4f99d4365"
)
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "423ae3c234e71be4e168a06d5270c7d437ad61118087b2fb548c37b0072269e6"
)
PREREGISTRATION_MANIFEST_HASH = (
    "fbee6397da5b891a9e586ca05388dbec64b5f02245324272987670ff25865eea"
)
PREREGISTRATION_BUILDER = Path(
    "training/preregister_dollar_collateral_liquidity_bank_relay_v2.py"
)
PREREGISTRATION_BUILDER_SHA256 = (
    "8fe202d116d5a2c813f2243334530dc3bb3e2851808aa728ccd2bf54838ccf20"
)
BASE_PREREGISTRATION_BUILDER = Path(prereg.BASE_BUILDER)
BASE_PREREGISTRATION_BUILDER_SHA256 = (
    "852b4225adfcb2ca1c00f4c923e39674aec36245f1d3eda4c8bd6c07683dab99"
)
DEFAULT_CLOCK_OUTPUT = Path(
    "data/dollar_collateral_liquidity_bank_relay_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT_OUTPUT = Path(
    "results/dollar_collateral_liquidity_bank_relay_support_2026-07-24.json"
)

NEW_YORK = ZoneInfo("America/New_York")
BAR = pd.Timedelta(minutes=5)
HOLD = pd.Timedelta(minutes=4_320)
COMMON_START = pd.Timestamp("2020-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
COMMON_END = pd.Timestamp("2024-01-01T00:00:00Z")
EXCLUDED_H8_RELEASES = frozenset(
    {"2020-10-02", "2023-03-31", "2023-06-30", "2023-12-15"}
)
RFC3339_ZONE = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$")

WINDOWS = {
    "train": (COMMON_START, TRAIN_END),
    "selection": (TRAIN_END, COMMON_END),
    "2020": (COMMON_START, pd.Timestamp("2021-01-01T00:00:00Z")),
    "2021": (
        pd.Timestamp("2021-01-01T00:00:00Z"),
        pd.Timestamp("2022-01-01T00:00:00Z"),
    ),
    "2022": (pd.Timestamp("2022-01-01T00:00:00Z"), TRAIN_END),
    "2023_h1": (TRAIN_END, pd.Timestamp("2023-07-01T00:00:00Z")),
    "2023_h2": (pd.Timestamp("2023-07-01T00:00:00Z"), COMMON_END),
    "2023_q1": (TRAIN_END, pd.Timestamp("2023-04-01T00:00:00Z")),
    "2023_q2": (
        pd.Timestamp("2023-04-01T00:00:00Z"),
        pd.Timestamp("2023-07-01T00:00:00Z"),
    ),
    "2023_q3": (
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2023-10-01T00:00:00Z"),
    ),
    "2023_q4": (pd.Timestamp("2023-10-01T00:00:00Z"), COMMON_END),
}

CLOCK_COLUMNS = (
    "control",
    "signal_id",
    "signal_available_time",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "h41_direction",
    "h41_transition",
    "rrp_direction",
    "rrp_transition",
    "macro_relation",
    "macro_strength",
    "h8_relief",
    "h8_agreement",
    "bank_relation",
    "h41_age_bucket",
    "rrp_count_bucket",
    "prior_side_transition",
)
FORBIDDEN_CLOCK_TOKENS = (
    "release_date",
    "observation_date",
    "operation_date",
    "raw",
    "rank",
    "numerator",
    "zscore",
    "price",
    "open",
    "high",
    "low",
    "close",
    "return",
    "future",
    "label",
    "funding",
    "pnl",
    "reward",
    "cagr",
    "mdd",
)
INDEPENDENT_CONTROLS = (
    "primary",
    "h41_only",
    "rrp_interval_only",
    "h8_only",
    "macro_concordant_only",
    "macro_discordant_only",
    "bank_supports_only",
    "bank_opposes_only",
    "stale_h41_one_release",
    "stale_rrp_one_interval",
    "one_h8_release_execution_delay",
    "nsa_h8",
)


class ComparatorContractFailure(RuntimeError):
    """Preserve fail-closed comparator evidence for canonical reporting."""

    def __init__(
        self,
        code: str,
        rows_decoded: int,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.rows_decoded = int(rows_decoded)


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def csv_header_bytes(path: str | Path) -> bytes:
    source = _path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as handle:
        header = handle.readline()
    if not header.endswith(b"\n") or b"\n" in header[:-1]:
        raise RuntimeError(f"DCLB-864 CSV header is not one LF line: {path}")
    return header


def csv_header(path: str | Path) -> list[str]:
    header = csv_header_bytes(path).decode("utf-8")
    columns = next(csv.reader([header.rstrip("\n")]))
    if len(columns) != len(set(columns)):
        raise RuntimeError(f"DCLB-864 duplicate CSV header columns: {path}")
    return columns


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def _format_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise RuntimeError("DCLB-864 timestamp must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.microsecond or timestamp.nanosecond:
        raise RuntimeError("DCLB-864 timestamp must be whole-second")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp_strings(
    values: pd.Series,
    *,
    label: str,
) -> pd.Series:
    strings = values.astype("string")
    if strings.eq("").any() or not strings.str.contains(RFC3339_ZONE).all():
        raise RuntimeError(f"DCLB-864 {label} timestamp lacks RFC3339 zone")
    parsed = pd.to_datetime(strings, utc=True, errors="raise")
    if parsed.isna().any():
        raise RuntimeError(f"DCLB-864 {label} timestamp missing")
    return parsed


def _parse_iso_dates(values: pd.Series, *, label: str) -> pd.Series:
    strings = values.astype("string")
    if not strings.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise RuntimeError(f"DCLB-864 {label} is not an exact ISO date")
    parsed = pd.to_datetime(strings, format="%Y-%m-%d", errors="raise")
    if not parsed.dt.strftime("%Y-%m-%d").eq(strings).all():
        raise RuntimeError(f"DCLB-864 {label} date round-trip failed")
    return parsed


def _git_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_protocol_committed() -> None:
    paths = (
        str(SCRIPT_PATH),
        str(TEST_PATH),
        str(IMPLEMENTATION_CONTRACT),
        str(PREREGISTRATION_BUILDER),
        str(BASE_PREREGISTRATION_BUILDER),
    )
    tracked = _git_check("ls-files", "--error-unmatch", "--", *paths)
    if tracked.returncode:
        raise RuntimeError("DCLB-864 source-support protocol is not committed")
    clean = _git_check("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("DCLB-864 source-support protocol differs from HEAD")


def validate_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("DCLB-864 preregistration artifact hash drift")
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    prereg.validate_manifest(payload)
    if payload != prereg.build_manifest():
        raise RuntimeError("DCLB-864 preregistration differs from frozen builder")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("DCLB-864 preregistration manifest hash drift")
    for field in (
        "outcomes_opened",
        "source_incidence_opened",
        "source_rows_decoded",
        "comparator_rows_decoded",
    ):
        if payload.get(field) is not False:
            raise RuntimeError(f"DCLB-864 preregistration boundary opened: {field}")
    if any(payload["evidence_boundary"].values()):
        raise RuntimeError("DCLB-864 preregistration evidence boundary opened")
    if tuple(payload["source_only_controls"]["ordered"]) != prereg.CONTROL_ORDER:
        raise RuntimeError("DCLB-864 control order drift")
    return payload


def verify_pre_source_bindings(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    prereg.validate_frozen_dependencies()
    bindings: list[tuple[str | Path, str, str]] = [
        (
            IMPLEMENTATION_CONTRACT,
            IMPLEMENTATION_CONTRACT_SHA256,
            "implementation_contract",
        ),
        (PREREGISTRATION, PREREGISTRATION_SHA256, "preregistration"),
        (
            PREREGISTRATION_BUILDER,
            PREREGISTRATION_BUILDER_SHA256,
            "preregistration_builder_v2",
        ),
        (
            BASE_PREREGISTRATION_BUILDER,
            BASE_PREREGISTRATION_BUILDER_SHA256,
            "preregistration_builder_v1",
        ),
    ]
    for path, expected in prereg.frozen_dependencies().items():
        bindings.append((path, expected, str(path)))
    audit: dict[str, dict[str, str]] = {}
    for path, expected, label in bindings:
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"DCLB-864 frozen binding changed: {label}")
        audit[label] = {"path": str(path), "sha256": actual}
    expected_allowlists = {
        "h41": list(prereg.H41_ALLOWLIST),
        "rrp": list(prereg.RRP_ALLOWLIST),
        "h8": list(prereg.H8_ALLOWLIST),
    }
    for name, allowlist in expected_allowlists.items():
        if payload["source_contracts"][name]["allowlist"] != allowlist:
            raise RuntimeError(f"DCLB-864 {name} allowlist drift")
    return audit


def _strictly_increasing(values: pd.Series) -> bool:
    return bool(values.is_monotonic_increasing and not values.duplicated().any())


def validate_h41_source(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != list(prereg.H41_ALLOWLIST):
        raise RuntimeError("DCLB-864 H.4.1 loader did not preserve allowlist")
    result = frame.copy()
    releases = _parse_iso_dates(result["release_date"], label="H.4.1 release")
    observations = _parse_iso_dates(
        result["observation_date"], label="H.4.1 observation"
    )
    available = _parse_timestamp_strings(
        result["available_at_utc"], label="H.4.1 availability"
    )
    if not _strictly_increasing(releases):
        raise RuntimeError("DCLB-864 H.4.1 release dates not unique/increasing")
    if not _strictly_increasing(observations):
        raise RuntimeError("DCLB-864 H.4.1 observation dates not unique/increasing")
    if not _strictly_increasing(available):
        raise RuntimeError("DCLB-864 H.4.1 availability not unique/increasing")
    if not observations.lt(releases).all():
        raise RuntimeError("DCLB-864 H.4.1 observation is not strictly prior")
    local_dates = available.dt.tz_convert(NEW_YORK).dt.strftime("%Y-%m-%d")
    if not local_dates.eq(result["release_date"]).all():
        raise RuntimeError("DCLB-864 H.4.1 availability date mismatch")
    if releases.ge(pd.Timestamp("2024-01-01")).any() or available.ge(
        COMMON_END
    ).any():
        raise RuntimeError("DCLB-864 H.4.1 contains 2024-or-later source")
    level = pd.to_numeric(
        result["net_liquidity_usd_millions"], errors="raise"
    ).astype(float)
    if not np.isfinite(level).all() or not level.gt(0.0).all():
        raise RuntimeError("DCLB-864 H.4.1 net-liquidity primitive invalid")
    result["available_at_utc"] = available
    result["net_liquidity_usd_millions"] = level
    return result


def validate_rrp_source(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != list(prereg.RRP_ALLOWLIST):
        raise RuntimeError("DCLB-864 ON RRP loader did not preserve allowlist")
    result = frame.copy()
    operations = _parse_iso_dates(
        result["operation_date"], label="ON RRP operation"
    )
    available = _parse_timestamp_strings(
        result["result_available_at_utc"], label="ON RRP availability"
    )
    if not _strictly_increasing(operations):
        raise RuntimeError("DCLB-864 ON RRP operation dates not unique/increasing")
    if not _strictly_increasing(available):
        raise RuntimeError("DCLB-864 ON RRP availability not unique/increasing")
    if operations.ge(pd.Timestamp("2024-01-01")).any() or available.ge(
        COMMON_END
    ).any():
        raise RuntimeError("DCLB-864 ON RRP contains 2024-or-later source")
    complete_text = result["source_complete"].astype("string")
    if not complete_text.isin(("true", "false")).all():
        raise RuntimeError("DCLB-864 ON RRP source_complete is not exact boolean")
    complete = complete_text.eq("true")
    quarantine = result["quarantine_reason"].astype("string")
    amount_text = result["total_amount_accepted_usd"].astype("string")
    if not quarantine.loc[complete].eq("").all():
        raise RuntimeError("DCLB-864 complete ON RRP row has quarantine reason")
    if not amount_text.loc[~complete].eq("").all():
        raise RuntimeError("DCLB-864 quarantined ON RRP amount is not blank")
    if not quarantine.loc[~complete].ne("").all():
        raise RuntimeError("DCLB-864 incomplete ON RRP row lacks quarantine reason")
    amount = pd.Series(np.nan, index=result.index, dtype=float)
    if complete.any():
        parsed = pd.to_numeric(amount_text.loc[complete], errors="raise").astype(float)
        if not np.isfinite(parsed).all() or not parsed.ge(0.0).all():
            raise RuntimeError("DCLB-864 complete ON RRP amount invalid")
        amount.loc[complete] = parsed
    result["result_available_at_utc"] = available
    result["total_amount_accepted_usd"] = amount
    result["source_complete"] = complete
    return result


def validate_h8_source(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != list(prereg.H8_ALLOWLIST):
        raise RuntimeError("DCLB-864 H.8 loader did not preserve allowlist")
    result = frame.copy()
    releases = _parse_iso_dates(result["release_date"], label="H.8 release")
    release_time = _parse_timestamp_strings(
        result["release_time_utc"], label="H.8 release"
    )
    if not _strictly_increasing(releases):
        raise RuntimeError("DCLB-864 H.8 release dates not unique/increasing")
    if not _strictly_increasing(release_time):
        raise RuntimeError("DCLB-864 H.8 release times not unique/increasing")
    local = release_time.dt.tz_convert(NEW_YORK)
    if not local.dt.strftime("%Y-%m-%d").eq(result["release_date"]).all():
        raise RuntimeError("DCLB-864 H.8 release date/time mismatch")
    if not local.dt.day_name().eq(result["release_weekday"]).all():
        raise RuntimeError("DCLB-864 H.8 release weekday mismatch")
    if releases.ge(pd.Timestamp("2024-01-01")).any() or release_time.ge(
        COMMON_END
    ).any():
        raise RuntimeError("DCLB-864 H.8 contains 2024-or-later source")
    for column in prereg.H8_ALLOWLIST:
        if column in {"release_date", "release_time_utc", "release_weekday"}:
            continue
        values = pd.to_numeric(result[column], errors="raise").astype(float)
        if not np.isfinite(values).all() or not values.gt(0.0).all():
            raise RuntimeError(f"DCLB-864 H.8 primitive invalid: {column}")
        result[column] = values
    result["release_time_utc"] = release_time
    return result


def _load_source(
    path: str | Path,
    *,
    expected_path: str,
    allowlist: Sequence[str],
    validator: Any,
    source_name: str,
) -> pd.DataFrame:
    if str(path) != expected_path:
        raise RuntimeError(f"DCLB-864 {source_name} path differs from freeze")
    frame = pd.read_csv(
        _path(path),
        usecols=list(allowlist),
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )
    frame = frame.loc[:, list(allowlist)]
    return validator(frame)


def load_h41_source(path: str | Path = prereg.H41_SOURCE) -> pd.DataFrame:
    return _load_source(
        path,
        expected_path=prereg.H41_SOURCE,
        allowlist=prereg.H41_ALLOWLIST,
        validator=validate_h41_source,
        source_name="H.4.1",
    )


def load_rrp_source(path: str | Path = prereg.RRP_SOURCE) -> pd.DataFrame:
    return _load_source(
        path,
        expected_path=prereg.RRP_SOURCE,
        allowlist=prereg.RRP_ALLOWLIST,
        validator=validate_rrp_source,
        source_name="ON RRP",
    )


def load_h8_source(path: str | Path = prereg.H8_SOURCE) -> pd.DataFrame:
    return _load_source(
        path,
        expected_path=prereg.H8_SOURCE,
        allowlist=prereg.H8_ALLOWLIST,
        validator=validate_h8_source,
        source_name="H.8",
    )


def midrank_numerator(current: float, prior: Iterable[float]) -> int:
    history = np.asarray(list(prior), dtype=np.float64)
    if not len(history) or not np.isfinite(history).all() or not np.isfinite(current):
        raise RuntimeError("DCLB-864 midrank received invalid history")
    return int(2 * np.sum(history < current) + np.sum(history == current))


def _direction(sign: int) -> str:
    return "RELIEF" if sign > 0 else "STRESS" if sign < 0 else "NEUTRAL"


def _transition(previous: int | None, current: int) -> str:
    if previous is None:
        return "NO_PRIOR"
    if previous == current == 0:
        return "NEUTRAL_PERSIST"
    if previous == 0:
        return "FROM_NEUTRAL"
    if current == 0:
        return "TO_NEUTRAL"
    return "PERSIST" if previous == current else "FLIP"


def build_h41_features(
    frame: pd.DataFrame,
    *,
    prior_deltas: int = 104,
) -> pd.DataFrame:
    if prior_deltas <= 0:
        raise RuntimeError("DCLB-864 H.4.1 rank width must be positive")
    records: list[dict[str, Any]] = []
    history: list[float] = []
    previous_level: float | None = None
    previous_rank_sign: int | None = None
    previous_emitted_num: int | None = None
    for row in frame.itertuples(index=False):
        level = float(row.net_liquidity_usd_millions)
        delta = None if previous_level is None else float(np.log(level / previous_level))
        number: int | None = None
        centered: int | None = None
        sign: int | None = None
        transition = "NO_PRIOR"
        stale_num = previous_emitted_num
        if delta is not None:
            if len(history) >= prior_deltas:
                prior = history[-prior_deltas:]
                number = midrank_numerator(delta, prior)
                centered = number - prior_deltas
                sign = int(np.sign(centered))
                transition = _transition(previous_rank_sign, sign)
                previous_rank_sign = sign
                previous_emitted_num = number
            history.append(delta)
        records.append(
            {
                "release_date": str(row.release_date),
                "available_at_utc": pd.Timestamp(row.available_at_utc),
                "delta": delta,
                "rank_num": number,
                "center_num": centered,
                "relief_sign": sign,
                "transition": transition,
                "previous_emitted_rank_num": stale_num,
            }
        )
        previous_level = level
    return pd.DataFrame.from_records(records)


def _decision_time(release_date: str) -> pd.Timestamp:
    local = pd.Timestamp(f"{release_date} 17:00:00", tz=NEW_YORK)
    return local.tz_convert("UTC")


def _entry_time(release_date: str) -> pd.Timestamp:
    local = pd.Timestamp(f"{release_date} 17:05:00", tz=NEW_YORK)
    return local.tz_convert("UTC")


def build_rrp_interval_features(
    rrp: pd.DataFrame,
    h8: pd.DataFrame,
    *,
    prior_deltas: int = 13,
) -> pd.DataFrame:
    if prior_deltas <= 0:
        raise RuntimeError("DCLB-864 ON RRP rank width must be positive")
    decisions = [_decision_time(str(value)) for value in h8["release_date"]]
    records: list[dict[str, Any]] = []
    history: list[float] = []
    previous_level: float | None = None
    previous_rank_sign: int | None = None
    previous_emitted_num: int | None = None
    segment = 0
    available = rrp["result_available_at_utc"]
    for index, decision in enumerate(decisions):
        if index == 0:
            records.append(
                {
                    "decision_time": decision,
                    "interval_count": 0,
                    "complete": False,
                    "latest_available_at": pd.NaT,
                    "level": None,
                    "delta": None,
                    "rank_num": None,
                    "center_num": None,
                    "relief_sign": None,
                    "transition": "NO_PRIOR",
                    "previous_emitted_rank_num": None,
                    "segment": segment,
                }
            )
            continue
        previous_decision = decisions[index - 1]
        mask = available.gt(previous_decision) & available.le(decision)
        interval = rrp.loc[mask]
        count = len(interval)
        complete = bool(
            3 <= count <= 7
            and count > 0
            and interval["source_complete"].all()
            and interval["quarantine_reason"].astype("string").eq("").all()
            and np.isfinite(
                interval["total_amount_accepted_usd"].to_numpy(dtype=float)
            ).all()
        )
        latest_available = (
            pd.Timestamp(interval["result_available_at_utc"].max())
            if count
            else pd.NaT
        )
        if not complete:
            history.clear()
            previous_level = None
            previous_rank_sign = None
            previous_emitted_num = None
            segment += 1
            records.append(
                {
                    "decision_time": decision,
                    "interval_count": count,
                    "complete": False,
                    "latest_available_at": latest_available,
                    "level": None,
                    "delta": None,
                    "rank_num": None,
                    "center_num": None,
                    "relief_sign": None,
                    "transition": "NO_PRIOR",
                    "previous_emitted_rank_num": None,
                    "segment": segment,
                }
            )
            continue
        amount = interval["total_amount_accepted_usd"].to_numpy(dtype=float)
        level = float(np.log1p(float(np.mean(amount)) / 1_000_000_000.0))
        delta = None if previous_level is None else level - previous_level
        number: int | None = None
        centered: int | None = None
        relief_sign: int | None = None
        transition = "NO_PRIOR"
        stale_num = previous_emitted_num
        if delta is not None:
            if len(history) >= prior_deltas:
                number = midrank_numerator(delta, history[-prior_deltas:])
                centered = number - prior_deltas
                relief_sign = int(np.sign(-centered))
                transition = _transition(previous_rank_sign, relief_sign)
                previous_rank_sign = relief_sign
                previous_emitted_num = number
            history.append(delta)
        records.append(
            {
                "decision_time": decision,
                "interval_count": count,
                "complete": True,
                "latest_available_at": latest_available,
                "level": level,
                "delta": delta,
                "rank_num": number,
                "center_num": centered,
                "relief_sign": relief_sign,
                "transition": transition,
                "previous_emitted_rank_num": stale_num,
                "segment": segment,
            }
        )
        previous_level = level
    return pd.DataFrame.from_records(records)


def _robust_z(current: float, prior: Sequence[float]) -> float | None:
    values = np.asarray(prior, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(current):
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    denominator = 1.4826 * mad
    if not np.isfinite(denominator) or denominator <= 0.0:
        return None
    value = float((current - median) / denominator)
    return value if np.isfinite(value) else None


def build_h8_features(
    frame: pd.DataFrame,
    *,
    adjustment: str,
    prior_observations: int = 104,
) -> pd.DataFrame:
    if adjustment not in {"sa", "nsa"}:
        raise RuntimeError("DCLB-864 H.8 adjustment must be SA or NSA")
    if prior_observations <= 0:
        raise RuntimeError("DCLB-864 H.8 robust width must be positive")
    histories: list[list[float]] = [[], [], []]
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        large_prior = float(
            getattr(row, f"{adjustment}_large_other_deposits_prior")
        )
        large_latest = float(
            getattr(row, f"{adjustment}_large_other_deposits_latest")
        )
        small_prior = float(
            getattr(row, f"{adjustment}_small_other_deposits_prior")
        )
        small_latest = float(
            getattr(row, f"{adjustment}_small_other_deposits_latest")
        )
        borrowing_prior = float(
            getattr(row, f"{adjustment}_small_borrowings_prior")
        )
        borrowing_latest = float(
            getattr(row, f"{adjustment}_small_borrowings_latest")
        )
        cash_prior = float(
            getattr(row, f"{adjustment}_small_cash_assets_prior")
        )
        cash_latest = float(
            getattr(row, f"{adjustment}_small_cash_assets_latest")
        )
        components = [
            float(
                10_000.0 * np.log(large_latest / large_prior)
                - 10_000.0 * np.log(small_latest / small_prior)
            ),
            float(10_000.0 * np.log(borrowing_latest / borrowing_prior)),
            float(-10_000.0 * np.log(cash_latest / cash_prior)),
        ]
        zscores: list[float | None] = []
        for component, history in zip(components, histories, strict=True):
            zscores.append(
                _robust_z(component, history[-prior_observations:])
                if len(history) >= prior_observations
                else None
            )
        valid_z = all(value is not None for value in zscores)
        stress = (
            float(np.mean(np.asarray(zscores, dtype=float))) if valid_z else None
        )
        stress_sign = (
            int(np.sign(stress))
            if stress is not None and np.isfinite(stress) and stress != 0.0
            else 0
        )
        agreement = (
            int(
                np.sum(
                    np.sign(np.asarray(zscores, dtype=float)) == stress_sign
                )
            )
            if valid_z and stress_sign
            else 0
        )
        valid = bool(valid_z and stress_sign and agreement >= 2)
        relief_sign = -stress_sign if valid else None
        records.append(
            {
                "release_date": str(row.release_date),
                "release_time_utc": pd.Timestamp(row.release_time_utc),
                "components": components,
                "zscores": zscores,
                "stress": stress,
                "relief_sign": relief_sign,
                "agreement": agreement,
                "valid": valid,
            }
        )
        for component, history in zip(components, histories, strict=True):
            history.append(component)
    return pd.DataFrame.from_records(records)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _macro_state(h41_center: int, rrp_center: int) -> dict[str, Any]:
    h41_sign = int(np.sign(h41_center))
    rrp_sign = int(np.sign(-rrp_center))
    macro_integer = 13 * h41_center - 104 * rrp_center
    side_sign = int(np.sign(macro_integer))
    if h41_sign == 0 or rrp_sign == 0:
        relation = "HAS_NEUTRAL_COMPONENT"
    elif h41_sign == rrp_sign:
        relation = "MACRO_CONCORDANT"
    elif side_sign == 0:
        relation = "MACRO_BALANCED_OPPOSITION"
    elif side_sign == h41_sign:
        relation = "MACRO_DISCORDANT_H41_DOMINANT"
    elif side_sign == rrp_sign:
        relation = "MACRO_DISCORDANT_RRP_DOMINANT"
    else:
        raise RuntimeError("DCLB-864 macro dominance is undefined")
    return {
        "h41_sign": h41_sign,
        "rrp_sign": rrp_sign,
        "macro_integer": macro_integer,
        "side_sign": side_sign,
        "macro_relation": relation,
        "macro_strength": (
            "WEAK" if abs(macro_integer) <= 13 * 52 else "STRONG"
        ),
    }


def _bank_relation(h8_relief_sign: int, side_sign: int) -> str:
    if h8_relief_sign not in (-1, 1) or side_sign not in (-1, 1):
        raise RuntimeError("DCLB-864 bank relation received neutral sign")
    return (
        "BANK_SUPPORTS"
        if h8_relief_sign == side_sign
        else "BANK_OPPOSES"
    )


def _h41_age_bucket(decision: pd.Timestamp, available: pd.Timestamp) -> str:
    decision_date = decision.tz_convert(NEW_YORK).date()
    available_date = available.tz_convert(NEW_YORK).date()
    age = (decision_date - available_date).days
    if age < 0:
        raise RuntimeError("DCLB-864 H.4.1 age is negative")
    if age == 0:
        return "SAME_DAY"
    if age == 1:
        return "ONE_DAY"
    if age <= 3:
        return "TWO_TO_THREE_DAYS"
    return "FOUR_PLUS_DAYS"


def _rrp_count_bucket(count: int) -> str:
    if count in (3, 4):
        return "THREE_TO_FOUR"
    if count == 5:
        return "FIVE"
    if count in (6, 7):
        return "SIX_TO_SEVEN"
    raise RuntimeError("DCLB-864 complete ON RRP count is outside 3..7")


def build_joint_states(
    h41: pd.DataFrame,
    rrp: pd.DataFrame,
    h8: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    h41_features = build_h41_features(h41)
    rrp_features = build_rrp_interval_features(rrp, h8)
    sa = build_h8_features(h8, adjustment="sa")
    nsa = build_h8_features(h8, adjustment="nsa")
    h41_available = h41_features["available_at_utc"]
    records: list[dict[str, Any]] = []
    prior_primary_side: int | None = None
    for index, h8_row in enumerate(h8.itertuples(index=False)):
        release_date = str(h8_row.release_date)
        release_time = pd.Timestamp(h8_row.release_time_utc)
        decision = _decision_time(release_date)
        entry = _entry_time(release_date)
        exit_time = entry + HOLD
        previous_decision = (
            _decision_time(str(h8.iloc[index - 1]["release_date"]))
            if index
            else None
        )
        h41_index = int(h41_available.searchsorted(decision, side="right") - 1)
        h41_row = (
            h41_features.iloc[h41_index] if h41_index >= 0 else None
        )
        h41_available_at = (
            pd.Timestamp(h41_row["available_at_utc"])
            if h41_row is not None
            else pd.NaT
        )
        h41_fresh = bool(
            h41_row is not None
            and previous_decision is not None
            and h41_available_at > previous_decision
            and h41_available_at <= decision
        )
        h41_num = (
            _optional_int(h41_row["rank_num"])
            if h41_row is not None
            else None
        )
        h41_center = (
            _optional_int(h41_row["center_num"])
            if h41_row is not None
            else None
        )
        h41_stale_num = (
            _optional_int(h41_row["previous_emitted_rank_num"])
            if h41_row is not None
            else None
        )
        rrp_row = rrp_features.iloc[index]
        rrp_num = _optional_int(rrp_row["rank_num"])
        rrp_center = _optional_int(rrp_row["center_num"])
        rrp_stale_num = _optional_int(rrp_row["previous_emitted_rank_num"])
        rrp_latest = (
            pd.Timestamp(rrp_row["latest_available_at"])
            if not pd.isna(rrp_row["latest_available_at"])
            else pd.NaT
        )
        common_valid = bool(
            release_date not in EXCLUDED_H8_RELEASES
            and decision > release_time
            and h41_fresh
            and h41_num is not None
            and h41_center is not None
            and bool(rrp_row["complete"])
            and rrp_num is not None
            and rrp_center is not None
            and not pd.isna(rrp_latest)
        )
        macro = (
            _macro_state(h41_center, rrp_center)
            if h41_center is not None and rrp_center is not None
            else None
        )
        sa_row = sa.iloc[index]
        nsa_row = nsa.iloc[index]
        sa_valid = bool(sa_row["valid"])
        nsa_valid = bool(nsa_row["valid"])
        primary_eligible = bool(
            common_valid
            and sa_valid
            and macro is not None
            and macro["side_sign"] != 0
        )
        prior_transition = "NO_PRIOR"
        if primary_eligible:
            current_side = int(macro["side_sign"])
            prior_transition = (
                "NO_PRIOR"
                if prior_primary_side is None
                else "PERSIST"
                if prior_primary_side == current_side
                else "FLIP"
            )
            prior_primary_side = current_side
        signal_available = pd.NaT
        if common_valid:
            signal_available = max(h41_available_at, rrp_latest, release_time)
            if signal_available > decision:
                raise RuntimeError(
                    "DCLB-864 selected source was unavailable at decision"
                )
        records.append(
            {
                "h8_index": index,
                "release_date": release_date,
                "release_time": release_time,
                "decision_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "signal_available_time": signal_available,
                "excluded": release_date in EXCLUDED_H8_RELEASES,
                "common_valid": common_valid,
                "primary_eligible": primary_eligible,
                "h41_num": h41_num,
                "h41_center": h41_center,
                "h41_stale_num": h41_stale_num,
                "h41_direction": (
                    _direction(int(np.sign(h41_center)))
                    if h41_center is not None
                    else None
                ),
                "h41_transition": (
                    str(h41_row["transition"])
                    if h41_row is not None
                    else "NO_PRIOR"
                ),
                "h41_age_bucket": (
                    _h41_age_bucket(decision, h41_available_at)
                    if common_valid
                    else None
                ),
                "rrp_num": rrp_num,
                "rrp_center": rrp_center,
                "rrp_stale_num": rrp_stale_num,
                "rrp_segment": int(rrp_row["segment"]),
                "rrp_direction": (
                    _direction(int(np.sign(-rrp_center)))
                    if rrp_center is not None
                    else None
                ),
                "rrp_transition": str(rrp_row["transition"]),
                "rrp_count_bucket": (
                    _rrp_count_bucket(int(rrp_row["interval_count"]))
                    if common_valid
                    else None
                ),
                "macro_integer": (
                    int(macro["macro_integer"]) if macro is not None else None
                ),
                "side_sign": (
                    int(macro["side_sign"]) if macro is not None else None
                ),
                "macro_relation": (
                    str(macro["macro_relation"]) if macro is not None else None
                ),
                "macro_strength": (
                    str(macro["macro_strength"]) if macro is not None else None
                ),
                "sa_valid": sa_valid,
                "sa_relief_sign": _optional_int(sa_row["relief_sign"]),
                "sa_agreement": int(sa_row["agreement"]),
                "nsa_valid": nsa_valid,
                "nsa_relief_sign": _optional_int(nsa_row["relief_sign"]),
                "nsa_agreement": int(nsa_row["agreement"]),
                "prior_side_transition": prior_transition,
            }
        )
    states = pd.DataFrame.from_records(records)
    funnel = {
        "h41_source_rows": len(h41),
        "h41_finite_deltas": int(h41_features["delta"].notna().sum()),
        "h41_rank_complete_rows": int(h41_features["rank_num"].notna().sum()),
        "rrp_source_rows": len(rrp),
        "rrp_quarantined_rows": int((~rrp["source_complete"]).sum()),
        "rrp_complete_intervals": int(rrp_features["complete"].sum()),
        "rrp_finite_deltas": int(rrp_features["delta"].notna().sum()),
        "rrp_rank_complete_intervals": int(
            rrp_features["rank_num"].notna().sum()
        ),
        "h8_source_rows": len(h8),
        "h8_sa_valid_rows": int(sa["valid"].sum()),
        "h8_nsa_valid_rows": int(nsa["valid"].sum()),
        "common_causal_rows": int(states["common_valid"].sum()),
        "raw_primary_eligible_rows": int(states["primary_eligible"].sum()),
    }
    return states, funnel


def _side_name(sign: int) -> str:
    if sign == 1:
        return "LONG"
    if sign == -1:
        return "SHORT"
    raise RuntimeError("DCLB-864 event side must be nonzero")


def _event_signal_id(row: Mapping[str, Any]) -> str:
    ordered = [
        str(row[column])
        if column not in {
            "signal_available_time",
            "decision_time",
            "entry_time",
            "exit_time",
        }
        else _format_time(row[column])
        for column in CLOCK_COLUMNS
        if column != "signal_id"
    ]
    return hashlib.sha256("|".join(ordered).encode("utf-8")).hexdigest()[:24]


def _event_from_state(
    control: str,
    state: Mapping[str, Any],
    *,
    side_sign: int,
    h41_center: int | None = None,
    rrp_center: int | None = None,
    h8_relief_sign: int | None = None,
    h8_agreement: int | None = None,
    decision_time: pd.Timestamp | None = None,
    entry_time: pd.Timestamp | None = None,
    exit_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    h41_value = int(state["h41_center"]) if h41_center is None else h41_center
    rrp_value = int(state["rrp_center"]) if rrp_center is None else rrp_center
    macro = _macro_state(h41_value, rrp_value)
    relief = (
        int(state["sa_relief_sign"])
        if h8_relief_sign is None
        else int(h8_relief_sign)
    )
    agreement = (
        int(state["sa_agreement"])
        if h8_agreement is None
        else int(h8_agreement)
    )
    event = {
        "control": control,
        "signal_id": "",
        "signal_available_time": pd.Timestamp(state["signal_available_time"]),
        "decision_time": (
            pd.Timestamp(state["decision_time"])
            if decision_time is None
            else pd.Timestamp(decision_time)
        ),
        "entry_time": (
            pd.Timestamp(state["entry_time"])
            if entry_time is None
            else pd.Timestamp(entry_time)
        ),
        "exit_time": (
            pd.Timestamp(state["exit_time"])
            if exit_time is None
            else pd.Timestamp(exit_time)
        ),
        "side": _side_name(side_sign),
        "h41_direction": _direction(macro["h41_sign"]),
        "h41_transition": str(state["h41_transition"]),
        "rrp_direction": _direction(macro["rrp_sign"]),
        "rrp_transition": str(state["rrp_transition"]),
        "macro_relation": str(macro["macro_relation"]),
        "macro_strength": str(macro["macro_strength"]),
        "h8_relief": _direction(relief),
        "h8_agreement": (
            "THREE_OF_THREE" if agreement == 3 else "TWO_OF_THREE"
        ),
        "bank_relation": _bank_relation(relief, side_sign),
        "h41_age_bucket": str(state["h41_age_bucket"]),
        "rrp_count_bucket": str(state["rrp_count_bucket"]),
        "prior_side_transition": str(state["prior_side_transition"]),
    }
    event["signal_id"] = _event_signal_id(event)
    return event


def _event_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    records = list(rows)
    if not records:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in CLOCK_COLUMNS})
    frame = pd.DataFrame.from_records(records, columns=CLOCK_COLUMNS)
    for column in (
        "signal_available_time",
        "decision_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if not frame["side"].isin(("LONG", "SHORT")).all():
        raise RuntimeError("DCLB-864 clock contains invalid side")
    return frame


def reserve_nonoverlap(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    raw = _event_frame(rows).sort_values(
        ["entry_time", "signal_id"], kind="mergesort"
    )
    accepted: list[dict[str, Any]] = []
    active_until: pd.Timestamp | None = None
    for row in raw.to_dict("records"):
        entry = pd.Timestamp(row["entry_time"])
        exit_time = pd.Timestamp(row["exit_time"])
        if exit_time <= entry:
            raise RuntimeError("DCLB-864 event exit does not follow entry")
        if active_until is not None and entry < active_until:
            continue
        accepted.append(row)
        active_until = exit_time
    return _event_frame(accepted).reset_index(drop=True)


def _contained(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return frame.loc[
        frame["entry_time"].ge(start) & frame["exit_time"].le(end)
    ].copy()


def _split_contained(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = [_contained(frame, *WINDOWS[name]) for name in ("train", "selection")]
    nonempty = [piece for piece in pieces if not piece.empty]
    if not nonempty:
        return _event_frame([])
    return (
        pd.concat(nonempty, ignore_index=True)
        .sort_values(["entry_time", "signal_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _reidentity(
    row: Mapping[str, Any],
    *,
    control: str,
    side: str | None = None,
) -> dict[str, Any]:
    result = {column: row[column] for column in CLOCK_COLUMNS}
    result["control"] = control
    if side is not None:
        result["side"] = side
    result["signal_id"] = ""
    result["signal_id"] = _event_signal_id(result)
    return result


def build_controls(
    states: pd.DataFrame,
    h8: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[str, int]]:
    raw: dict[str, list[dict[str, Any]]] = {
        name: [] for name in INDEPENDENT_CONTROLS
    }
    state_records = states.to_dict("records")
    for state in state_records:
        common = bool(state["common_valid"])
        sa_valid = bool(state["sa_valid"])
        macro_side = _optional_int(state["side_sign"])
        h41_center = _optional_int(state["h41_center"])
        rrp_center = _optional_int(state["rrp_center"])
        if common and sa_valid and h41_center not in (None, 0):
            raw["h41_only"].append(
                _event_from_state(
                    "h41_only",
                    state,
                    side_sign=int(np.sign(h41_center)),
                )
            )
        rrp_relief_sign = (
            int(np.sign(-rrp_center)) if rrp_center is not None else 0
        )
        if common and sa_valid and rrp_relief_sign:
            raw["rrp_interval_only"].append(
                _event_from_state(
                    "rrp_interval_only",
                    state,
                    side_sign=rrp_relief_sign,
                )
            )
        if common and sa_valid:
            h8_side = _optional_int(state["sa_relief_sign"])
            if h8_side:
                raw["h8_only"].append(
                    _event_from_state(
                        "h8_only",
                        state,
                        side_sign=h8_side,
                    )
                )
        if bool(state["primary_eligible"]) and macro_side:
            primary = _event_from_state(
                "primary", state, side_sign=macro_side
            )
            raw["primary"].append(primary)
            relation = str(state["macro_relation"])
            if relation == "MACRO_CONCORDANT":
                raw["macro_concordant_only"].append(
                    _event_from_state(
                        "macro_concordant_only",
                        state,
                        side_sign=macro_side,
                    )
                )
            if relation.startswith("MACRO_DISCORDANT"):
                raw["macro_discordant_only"].append(
                    _event_from_state(
                        "macro_discordant_only",
                        state,
                        side_sign=macro_side,
                    )
                )
            bank = _bank_relation(int(state["sa_relief_sign"]), macro_side)
            raw[
                "bank_supports_only"
                if bank == "BANK_SUPPORTS"
                else "bank_opposes_only"
            ].append(
                _event_from_state(
                    (
                        "bank_supports_only"
                        if bank == "BANK_SUPPORTS"
                        else "bank_opposes_only"
                    ),
                    state,
                    side_sign=macro_side,
                )
            )
            host_index = int(state["h8_index"]) + 1
            if host_index < len(h8):
                host = h8.iloc[host_index]
                host_decision = _decision_time(str(host["release_date"]))
                host_entry = _entry_time(str(host["release_date"]))
                host_release = pd.Timestamp(host["release_time_utc"])
                if host_entry > host_release:
                    raw["one_h8_release_execution_delay"].append(
                        _event_from_state(
                            "one_h8_release_execution_delay",
                            state,
                            side_sign=macro_side,
                            decision_time=host_decision,
                            entry_time=host_entry,
                            exit_time=host_entry + HOLD,
                        )
                    )
        if common and sa_valid and rrp_center is not None:
            stale_h41_num = _optional_int(state["h41_stale_num"])
            if stale_h41_num is not None:
                stale_h41_center = stale_h41_num - 104
                stale_macro = _macro_state(stale_h41_center, rrp_center)
                if stale_macro["side_sign"]:
                    raw["stale_h41_one_release"].append(
                        _event_from_state(
                            "stale_h41_one_release",
                            state,
                            side_sign=int(stale_macro["side_sign"]),
                            h41_center=stale_h41_center,
                        )
                    )
        if common and sa_valid and h41_center is not None:
            stale_rrp_num = _optional_int(state["rrp_stale_num"])
            if stale_rrp_num is not None:
                stale_rrp_center = stale_rrp_num - 13
                stale_macro = _macro_state(h41_center, stale_rrp_center)
                if stale_macro["side_sign"]:
                    raw["stale_rrp_one_interval"].append(
                        _event_from_state(
                            "stale_rrp_one_interval",
                            state,
                            side_sign=int(stale_macro["side_sign"]),
                            rrp_center=stale_rrp_center,
                        )
                    )
        if (
            common
            and bool(state["nsa_valid"])
            and macro_side not in (None, 0)
        ):
            raw["nsa_h8"].append(
                _event_from_state(
                    "nsa_h8",
                    state,
                    side_sign=macro_side,
                    h8_relief_sign=int(state["nsa_relief_sign"]),
                    h8_agreement=int(state["nsa_agreement"]),
                )
            )
    raw_counts = {name: len(rows) for name, rows in raw.items()}
    reserved_global = {
        name: reserve_nonoverlap(raw[name]) for name in INDEPENDENT_CONTROLS
    }
    primary = reserved_global["primary"]
    flipped = [
        _reidentity(
            row,
            control="exact_direction_flip",
            side="SHORT" if row["side"] == "LONG" else "LONG",
        )
        for row in primary.to_dict("records")
    ]
    random_rows = []
    for row in primary.to_dict("records"):
        digest = hashlib.sha256(
            f"DCLB-864|{_format_time(row['entry_time'])}".encode("utf-8")
        ).digest()
        random_rows.append(
            _reidentity(
                row,
                control="deterministic_random_side",
                side="LONG" if digest[0] < 128 else "SHORT",
            )
        )
    reserved_global["exact_direction_flip"] = _event_frame(flipped)
    reserved_global["deterministic_random_side"] = _event_frame(random_rows)
    raw_counts["exact_direction_flip"] = len(primary)
    raw_counts["deterministic_random_side"] = len(primary)
    reserved_counts = {
        name: len(reserved_global[name]) for name in prereg.CONTROL_ORDER
    }
    controls = {
        name: _split_contained(reserved_global[name])
        for name in prereg.CONTROL_ORDER
    }
    return controls, raw_counts, reserved_counts


def _window(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = WINDOWS[name]
    return _contained(frame, start, end).sort_values(
        ["entry_time", "signal_id"], kind="mergesort"
    )


def _longest_run(values: Iterable[str]) -> int:
    best = 0
    current = 0
    previous: str | None = None
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        best = max(best, current)
    return best


def clock_stats(rows: pd.DataFrame) -> dict[str, Any]:
    ordered = rows.sort_values(
        ["entry_time", "signal_id"], kind="mergesort"
    )
    total = len(ordered)
    if not total:
        return {
            "events": 0,
            "long": 0,
            "short": 0,
            "long_share": None,
            "short_share": None,
            "active_months": 0,
            "maximum_month_share": None,
            "maximum_quarter_share": None,
            "maximum_gap_days": None,
            "maximum_same_side_run": 0,
        }
    sides = ordered["side"].value_counts().to_dict()
    local_entry = ordered["entry_time"].dt.tz_convert(NEW_YORK).dt.tz_localize(
        None
    )
    months = local_entry.dt.to_period("M").astype(str).value_counts()
    quarters = local_entry.dt.to_period("Q").astype(str).value_counts()
    local_dates = [value.date() for value in local_entry]
    calendar_gaps = [
        (current - previous).days
        for previous, current in zip(local_dates, local_dates[1:])
    ]
    return {
        "events": total,
        "long": int(sides.get("LONG", 0)),
        "short": int(sides.get("SHORT", 0)),
        "long_share": float(sides.get("LONG", 0) / total),
        "short_share": float(sides.get("SHORT", 0) / total),
        "active_months": int(len(months)),
        "maximum_month_share": float(months.max() / total),
        "maximum_quarter_share": float(quarters.max() / total),
        "maximum_gap_days": (
            float(max(calendar_gaps)) if calendar_gaps else None
        ),
        "maximum_same_side_run": _longest_run(ordered["side"]),
    }


def same_side_reproduction(
    primary: pd.DataFrame,
    control: pd.DataFrame,
    split: str,
) -> float | None:
    candidate = _window(primary, split)
    if candidate.empty:
        return None
    comparison = _window(control, split)
    lookup = {
        (pd.Timestamp(row.entry_time), str(row.side))
        for row in comparison.itertuples(index=False)
    }
    matched = sum(
        (pd.Timestamp(row.entry_time), str(row.side)) in lookup
        for row in candidate.itertuples(index=False)
    )
    return float(matched / len(candidate))


def _timing_integrity(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    if frame["signal_id"].duplicated().any():
        return False
    if not all(
        str(row["signal_id"]) == _event_signal_id(row)
        for row in frame.to_dict("records")
    ):
        return False
    if not frame["signal_available_time"].le(frame["decision_time"]).all():
        return False
    if not frame["decision_time"].lt(frame["entry_time"]).all():
        return False
    if not frame["exit_time"].sub(frame["entry_time"]).eq(HOLD).all():
        return False
    decision_local = frame["decision_time"].dt.tz_convert(NEW_YORK)
    entry_local = frame["entry_time"].dt.tz_convert(NEW_YORK)
    if not (
        decision_local.dt.hour.eq(17)
        & decision_local.dt.minute.eq(0)
        & decision_local.dt.second.eq(0)
    ).all():
        return False
    if not (
        entry_local.dt.hour.eq(17)
        & entry_local.dt.minute.eq(5)
        & entry_local.dt.second.eq(0)
    ).all():
        return False
    return True


def _same_clock_side_control_integrity(
    primary: pd.DataFrame,
    control: pd.DataFrame,
    *,
    mode: str,
) -> bool:
    if len(primary) != len(control):
        return False
    ordered_primary = primary.sort_values("entry_time", kind="mergesort")
    ordered_control = control.sort_values("entry_time", kind="mergesort")
    unchanged = [
        column
        for column in CLOCK_COLUMNS
        if column not in {"control", "signal_id", "side"}
    ]
    for original, placebo in zip(
        ordered_primary.to_dict("records"),
        ordered_control.to_dict("records"),
        strict=True,
    ):
        if any(original[column] != placebo[column] for column in unchanged):
            return False
        if mode == "flip":
            expected = "SHORT" if original["side"] == "LONG" else "LONG"
        elif mode == "random":
            digest = hashlib.sha256(
                f"DCLB-864|{_format_time(original['entry_time'])}".encode(
                    "utf-8"
                )
            ).digest()
            expected = "LONG" if digest[0] < 128 else "SHORT"
        else:
            raise RuntimeError("DCLB-864 unknown same-clock control mode")
        if placebo["side"] != expected:
            return False
    return True


def _reservation_integrity(frame: pd.DataFrame) -> bool:
    ordered = frame.sort_values(
        ["entry_time", "signal_id"], kind="mergesort"
    )
    if len(ordered) <= 1:
        return True
    entries = ordered["entry_time"].iloc[1:].reset_index(drop=True)
    exits = ordered["exit_time"].iloc[:-1].reset_index(drop=True)
    return bool(entries.ge(exits).all())


def _composition_metrics(
    primary: pd.DataFrame,
    controls: Mapping[str, pd.DataFrame],
    split: str,
) -> dict[str, Any]:
    rows = _window(primary, split)
    total = len(rows)

    def share(column: str, value: str) -> float | None:
        if not total:
            return None
        return float(rows[column].eq(value).sum() / total)

    discordant = (
        float(rows["macro_relation"].str.startswith("MACRO_DISCORDANT").sum() / total)
        if total
        else None
    )
    return {
        "events": total,
        "bank_supports_share": share("bank_relation", "BANK_SUPPORTS"),
        "bank_opposes_share": share("bank_relation", "BANK_OPPOSES"),
        "macro_concordant_share": share(
            "macro_relation", "MACRO_CONCORDANT"
        ),
        "macro_discordant_share": discordant,
        "weak_share": share("macro_strength", "WEAK"),
        "strong_share": share("macro_strength", "STRONG"),
        "two_of_three_share": share("h8_agreement", "TWO_OF_THREE"),
        "three_of_three_share": share("h8_agreement", "THREE_OF_THREE"),
        "h41_only_same_side_reproduction": same_side_reproduction(
            primary, controls["h41_only"], split
        ),
        "rrp_interval_only_same_side_reproduction": same_side_reproduction(
            primary, controls["rrp_interval_only"], split
        ),
        "stale_h41_one_release_same_side_reproduction": same_side_reproduction(
            primary, controls["stale_h41_one_release"], split
        ),
        "stale_rrp_one_interval_same_side_reproduction": (
            same_side_reproduction(
                primary, controls["stale_rrp_one_interval"], split
            )
        ),
        "deterministic_random_side_same_side_reproduction": (
            same_side_reproduction(
                primary, controls["deterministic_random_side"], split
            )
        ),
    }


def support_checks(
    controls: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any], dict[str, bool]]:
    gate = prereg.build_manifest()["source_support_gate"]
    primary = controls["primary"]
    statistics = {
        name: clock_stats(_window(primary, name)) for name in WINDOWS
    }
    train = statistics["train"]
    selection = statistics["selection"]
    train_gate = gate["train"]
    selection_gate = gate["selection"]
    source_checks: dict[str, bool] = {
        "train_events_min": train["events"] >= train_gate["events_min"],
        "train_each_year_events_min": all(
            statistics[str(year)]["events"]
            >= train_gate["each_year_events_min"]
            for year in (2020, 2021, 2022)
        ),
        "train_active_months_min": (
            train["active_months"] >= train_gate["active_months_min"]
        ),
        "train_side_support": bool(
            train["long_share"] is not None
            and train["short_share"] is not None
            and train["long_share"] >= train_gate["each_side_share_min"]
            and train["short_share"] >= train_gate["each_side_share_min"]
        ),
        "train_maximum_month_share": bool(
            train["maximum_month_share"] is not None
            and train["maximum_month_share"]
            <= train_gate["maximum_month_share"]
        ),
        "train_maximum_quarter_share": bool(
            train["maximum_quarter_share"] is not None
            and train["maximum_quarter_share"]
            <= train_gate["maximum_quarter_share"]
        ),
        "train_maximum_entry_gap": bool(
            train["maximum_gap_days"] is not None
            and train["maximum_gap_days"]
            <= train_gate["maximum_entry_gap_days"]
        ),
        "train_maximum_same_side_run": (
            train["maximum_same_side_run"]
            <= train_gate["maximum_same_side_run"]
        ),
        "selection_events_min": (
            selection["events"] >= selection_gate["events_min"]
        ),
        "selection_each_half_events_min": all(
            statistics[name]["events"]
            >= selection_gate["each_half_events_min"]
            for name in ("2023_h1", "2023_h2")
        ),
        "selection_each_quarter_events_min": all(
            statistics[name]["events"]
            >= selection_gate["each_quarter_events_min"]
            for name in ("2023_q1", "2023_q2", "2023_q3", "2023_q4")
        ),
        "selection_active_months_min": (
            selection["active_months"] >= selection_gate["active_months_min"]
        ),
        "selection_side_support": bool(
            selection["long_share"] is not None
            and selection["short_share"] is not None
            and selection["long_share"]
            >= selection_gate["each_side_share_min"]
            and selection["short_share"]
            >= selection_gate["each_side_share_min"]
        ),
        "selection_maximum_month_share": bool(
            selection["maximum_month_share"] is not None
            and selection["maximum_month_share"]
            <= selection_gate["maximum_month_share"]
        ),
        "selection_maximum_entry_gap": bool(
            selection["maximum_gap_days"] is not None
            and selection["maximum_gap_days"]
            <= selection_gate["maximum_entry_gap_days"]
        ),
        "selection_maximum_same_side_run": (
            selection["maximum_same_side_run"]
            <= selection_gate["maximum_same_side_run"]
        ),
    }
    for split in ("train", "selection"):
        for control in prereg.CONTROL_ORDER:
            source_checks[f"{split}:required_control:{control}"] = (
                not _window(controls[control], split).empty
            )
    source_checks.update(
        {
        "all_controls_timing_integrity": all(
            _timing_integrity(controls[name])
            for name in prereg.CONTROL_ORDER
        ),
        "all_controls_global_nonoverlap": all(
            _reservation_integrity(controls[name])
            for name in prereg.CONTROL_ORDER
        ),
        "exact_direction_flip_identity": _same_clock_side_control_integrity(
            primary,
            controls["exact_direction_flip"],
            mode="flip",
        ),
        "deterministic_random_side_identity": (
            _same_clock_side_control_integrity(
                primary,
                controls["deterministic_random_side"],
                mode="random",
            )
        ),
        "clock_has_no_raw_or_outcome_columns": not any(
            token in column.lower()
            for column in CLOCK_COLUMNS
            for token in FORBIDDEN_CLOCK_TOKENS
        ),
        }
    )
    composition = {
        split: _composition_metrics(primary, controls, split)
        for split in ("train", "selection")
    }
    composition_gate = gate["composition_each_split"]
    composition_checks: dict[str, bool] = {}
    for split, metrics in composition.items():
        lower_checks = {
            "bank_supports_share": "bank_supports_share_min",
            "bank_opposes_share": "bank_opposes_share_min",
            "macro_concordant_share": "macro_concordant_share_min",
            "macro_discordant_share": "macro_discordant_share_min",
            "weak_share": "weak_share_min",
            "strong_share": "strong_share_min",
            "two_of_three_share": "two_of_three_share_min",
            "three_of_three_share": "three_of_three_share_min",
        }
        for metric_name, gate_name in lower_checks.items():
            value = metrics[metric_name]
            composition_checks[f"{split}:{metric_name}"] = bool(
                value is not None and value >= composition_gate[gate_name]
            )
        reproduction_checks = {
            "h41_only_same_side_reproduction": (
                "h41_only_same_side_reproduction_max"
            ),
            "rrp_interval_only_same_side_reproduction": (
                "rrp_only_same_side_reproduction_max"
            ),
            "stale_h41_one_release_same_side_reproduction": (
                "each_stale_same_side_reproduction_max"
            ),
            "stale_rrp_one_interval_same_side_reproduction": (
                "each_stale_same_side_reproduction_max"
            ),
            "deterministic_random_side_same_side_reproduction": (
                "random_same_side_reproduction_max"
            ),
        }
        for metric_name, gate_name in reproduction_checks.items():
            value = metrics[metric_name]
            composition_checks[f"{split}:{metric_name}"] = bool(
                value is not None and value <= composition_gate[gate_name]
            )
    return statistics, source_checks, composition, composition_checks


def first_failure(
    source_checks: Mapping[str, bool],
    composition_checks: Mapping[str, bool],
    novelty_checks: Mapping[str, bool],
    *,
    artifact_eligible: bool,
) -> tuple[str, str | None]:
    for name, passed in source_checks.items():
        if not passed:
            return "source_support", name
    for name, passed in composition_checks.items():
        if not passed:
            return "relational_composition", name
    if not artifact_eligible:
        return "artifact_eligibility", "synthetic_or_injected_build"
    for name, passed in novelty_checks.items():
        if not passed:
            return "comparator_novelty", name
    if not novelty_checks:
        return "comparator_novelty", "required_comparator_checks_missing"
    return "none", None


def _combined_clock(controls: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [
        controls[name] for name in prereg.CONTROL_ORDER if not controls[name].empty
    ]
    if not frames:
        return _event_frame([])
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            ["entry_time", "control", "signal_id"], kind="mergesort"
        )
        .reset_index(drop=True)
    )


def deterministic_clock_bytes(
    controls: Mapping[str, pd.DataFrame],
) -> bytes:
    combined = _combined_clock(controls)
    if list(combined.columns) != list(CLOCK_COLUMNS):
        raise RuntimeError("DCLB-864 clock schema drift")
    serialized = combined.copy()
    for column in (
        "signal_available_time",
        "decision_time",
        "entry_time",
        "exit_time",
    ):
        serialized[column] = serialized[column].map(_format_time)
    text = serialized.to_csv(
        index=False,
        columns=CLOCK_COLUMNS,
        lineterminator="\n",
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer,
        mode="wb",
        filename="",
        mtime=0,
    ) as zipped:
        zipped.write(text)
    return buffer.getvalue()


def maximum_tolerant_matches(
    left: Iterable[pd.Timestamp],
    right: Iterable[pd.Timestamp],
    tolerance: pd.Timedelta,
) -> int:
    first = sorted(pd.Timestamp(value) for value in left)
    second = sorted(pd.Timestamp(value) for value in right)
    left_index = right_index = matched = 0
    while left_index < len(first) and right_index < len(second):
        if first[left_index] < second[right_index] - tolerance:
            left_index += 1
        elif second[right_index] < first[left_index] - tolerance:
            right_index += 1
        else:
            matched += 1
            left_index += 1
            right_index += 1
    return matched


def exact_entry_jaccard(left: pd.DataFrame, right: pd.DataFrame) -> float:
    first = set(left["entry_time"])
    second = set(right["entry_time"])
    union = first | second
    return float(len(first & second) / len(union)) if union else 1.0


def tolerant_entry_jaccard(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: pd.Timedelta,
) -> float:
    matched = maximum_tolerant_matches(
        left["entry_time"], right["entry_time"], tolerance
    )
    denominator = len(left) + len(right) - matched
    return float(matched / denominator) if denominator else 1.0


def _side_sign(row: Any) -> int:
    if hasattr(row, "side_sign"):
        return int(row.side_sign)
    return 1 if str(row.side) == "LONG" else -1


def _signed_occupancy(
    rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    grid_size = int((end - start) / BAR)
    occupancy = np.zeros(grid_size, dtype=np.int8)
    ordered = rows.sort_values("entry_time", kind="mergesort")
    for row in ordered.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if (entry - start) % BAR or (exit_time - start) % BAR:
            raise RuntimeError("DCLB-864 comparator interval off five-minute grid")
        left = int((entry - start) / BAR)
        right = int((exit_time - start) / BAR)
        if left < 0 or right > grid_size or left >= right:
            raise RuntimeError("DCLB-864 comparator interval outside common window")
        if np.any(occupancy[left:right] != 0):
            raise RuntimeError("DCLB-864 selected clock overlaps itself")
        occupancy[left:right] = _side_sign(row)
    return occupancy


def occupancy_metrics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    start: pd.Timestamp = COMMON_START,
    end: pd.Timestamp = COMMON_END,
) -> float | None:
    first = _signed_occupancy(left, start, end)
    second = _signed_occupancy(right, start, end)
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return None
    correlation = float(np.corrcoef(first, second)[0, 1])
    if not np.isfinite(correlation):
        return None
    return abs(correlation)


def _validate_raw_group(rows: pd.DataFrame, *, key: str) -> pd.DataFrame:
    ordered = rows.sort_values("entry_time", kind="mergesort").reset_index(
        drop=True
    )
    if ordered["entry_time"].duplicated().any():
        raise RuntimeError(f"DCLB-864 comparator entries duplicated: {key}")
    if not ordered["exit_time"].gt(ordered["entry_time"]).all():
        raise RuntimeError(f"DCLB-864 comparator interval invalid: {key}")
    if len(ordered) > 1:
        entries = ordered["entry_time"].iloc[1:].reset_index(drop=True)
        exits = ordered["exit_time"].iloc[:-1].reset_index(drop=True)
        if not entries.ge(exits).all():
            raise RuntimeError(f"DCLB-864 comparator overlaps itself: {key}")
    return ordered


def _comparator_group_key(
    artifact_id: str,
    filters: Mapping[str, str],
) -> str:
    suffix = ",".join(f"{key}={value}" for key, value in filters.items())
    return f"{artifact_id}:{suffix}"


def _read_comparator_groups_impl(
    payload: Mapping[str, Any],
    decoded_row_counter: list[int],
) -> tuple[dict[str, dict[str, Any]], int]:
    groups: dict[str, dict[str, Any]] = {}
    decoded_rows = 0
    for contract in payload["novelty_contract"]["comparators"]:
        if sha256_file(contract["path"]) != contract["sha256"]:
            raise RuntimeError(
                f"DCLB-864 comparator hash drift: {contract['id']}"
            )
        if sha256_csv_header(contract["path"]) != contract["header_sha256"]:
            raise RuntimeError(
                f"DCLB-864 comparator header hash drift: {contract['id']}"
            )
        header = csv_header(contract["path"])
        if not set(contract["usecols"]).issubset(header):
            raise RuntimeError(
                f"DCLB-864 comparator usecols missing: {contract['id']}"
            )
        raw = pd.read_csv(
            _path(contract["path"]),
            usecols=list(contract["usecols"]),
            dtype="string",
            keep_default_na=False,
            na_filter=False,
        )
        raw = raw.loc[:, list(contract["usecols"])]
        decoded_rows += len(raw)
        decoded_row_counter[0] = decoded_rows
        parser = contract["parser"]
        artifact_id = str(contract["id"])
        entry_column = str(parser["entry_column"])
        exit_column = str(parser["exit_column"])
        side_column = str(parser["side_column"])
        parsed_entry = _parse_timestamp_strings(
            raw[entry_column],
            label=f"{artifact_id} all-row entry",
        )
        parsed_exit = _parse_timestamp_strings(
            raw[exit_column],
            label=f"{artifact_id} all-row exit",
        )
        side_mapping = {
            str(name): int(value)
            for name, value in parser["side_mapping"].items()
        }
        parsed_side = raw[side_column].map(side_mapping)
        if parsed_side.isna().any() or not parsed_side.isin((-1, 1)).all():
            raise RuntimeError(
                f"DCLB-864 comparator side invalid before filtering: "
                f"{artifact_id}"
            )
        if not parsed_exit.gt(parsed_entry).all():
            raise RuntimeError(
                f"DCLB-864 comparator interval invalid before filtering: "
                f"{artifact_id}"
            )
        parsed = raw.copy()
        parsed["_parsed_entry"] = parsed_entry
        parsed["_parsed_exit"] = parsed_exit
        parsed["_parsed_side"] = parsed_side.astype(int)
        for group_contract in contract["groups"]:
            filters = {
                str(name): str(value)
                for name, value in group_contract["filter"].items()
            }
            selected = parsed.copy()
            for column, value in filters.items():
                selected = selected.loc[selected[column].eq(value)]
            key = _comparator_group_key(artifact_id, filters)
            if selected.empty:
                raise RuntimeError(f"DCLB-864 comparator group empty: {key}")
            selected_rows = _validate_raw_group(
                pd.DataFrame(
                    {
                        "entry_time": selected["_parsed_entry"],
                        "exit_time": selected["_parsed_exit"],
                        "side_sign": selected["_parsed_side"].astype(int),
                    }
                ),
                key=key,
            )
            before = selected_rows["exit_time"].le(COMMON_START)
            after = selected_rows["entry_time"].ge(COMMON_END)
            contained_mask = selected_rows["entry_time"].ge(
                COMMON_START
            ) & selected_rows["exit_time"].le(COMMON_END)
            crossing = ~(before | after | contained_mask)
            contained = selected_rows.loc[contained_mask].reset_index(drop=True)
            minimum = int(group_contract["minimum_contained_rows"])
            if len(contained) < minimum:
                raise RuntimeError(
                    f"DCLB-864 comparator below contained floor: {key}"
                )
            _signed_occupancy(contained, COMMON_START, COMMON_END)
            groups[key] = {
                "artifact_id": str(contract["id"]),
                "filters": filters,
                "clock_family": str(contract["clock_family"]),
                "minimum_contained_rows": minimum,
                "rows": contained,
                "counts": {
                    "raw_selected_rows": len(selected_rows),
                    "fully_contained_rows": len(contained),
                    "before_window_rows": int(before.sum()),
                    "after_window_rows": int(after.sum()),
                    "boundary_crossing_rows": int(crossing.sum()),
                },
            }
    return groups, decoded_rows


def _read_comparator_groups(
    payload: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], int]:
    decoded_row_counter = [0]
    try:
        return _read_comparator_groups_impl(payload, decoded_row_counter)
    except ComparatorContractFailure:
        raise
    except Exception as error:
        raise ComparatorContractFailure(
            "comparator_artifact_contract",
            decoded_row_counter[0],
            str(error),
        ) from error


def _same_entry_same_side_candidate_share(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
) -> float:
    if candidate.empty:
        raise RuntimeError("DCLB-864 candidate denominator is empty")
    lookup = {
        (pd.Timestamp(row.entry_time), int(row.side_sign))
        for row in comparator.itertuples(index=False)
    }
    matched = sum(
        (pd.Timestamp(row.entry_time), _side_sign(row)) in lookup
        for row in candidate.itertuples(index=False)
    )
    return float(matched / len(candidate))


def _evaluate_novelty_impl(
    primary: pd.DataFrame,
    payload: Mapping[str, Any],
    decoded_row_counter: list[int],
) -> tuple[dict[str, Any], dict[str, bool], int]:
    candidate = _contained(primary, COMMON_START, COMMON_END)
    if candidate.empty:
        raise RuntimeError("DCLB-864 primary empty in novelty common window")
    _signed_occupancy(candidate, COMMON_START, COMMON_END)
    groups, decoded_rows = _read_comparator_groups(payload)
    decoded_row_counter[0] = decoded_rows
    novelty = payload["novelty_contract"]
    report: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for key, group in groups.items():
        comparator = group["rows"]
        exact = exact_entry_jaccard(candidate, comparator)
        reproduction = _same_entry_same_side_candidate_share(
            candidate, comparator
        )
        correlation = occupancy_metrics(
            candidate,
            comparator,
            COMMON_START,
            COMMON_END,
        )
        seven_day = tolerant_entry_jaccard(
            candidate, comparator, pd.Timedelta(days=7)
        )
        family = group["clock_family"]
        item = {
            "artifact_id": group["artifact_id"],
            "filters": group["filters"],
            "clock_family": family,
            "common_window": [
                _format_time(COMMON_START),
                _format_time(COMMON_END),
            ],
            "candidate_rows": len(candidate),
            "comparator_counts": group["counts"],
            "minimum_contained_rows": group["minimum_contained_rows"],
            "exact_entry_jaccard": exact,
            "same_entry_same_side_reproduction": reproduction,
            "absolute_signed_occupancy_pearson": correlation,
            "seven_calendar_day_tolerant_jaccard_report_only": seven_day,
        }
        if family == "same_h8_anchor":
            thresholds = novelty["same_h8_anchor_thresholds"]
            checks[f"{key}:exact_entry_jaccard"] = (
                exact <= thresholds["exact_entry_jaccard_max"]
            )
            checks[f"{key}:same_entry_same_side_reproduction"] = (
                reproduction
                <= thresholds["same_entry_same_side_reproduction_max"]
            )
            checks[f"{key}:signed_occupancy_pearson"] = bool(
                correlation is not None
                and correlation
                <= thresholds["absolute_signed_occupancy_pearson_max"]
            )
        elif family == "asynchronous":
            thresholds = novelty["asynchronous_thresholds"]
            six_hour = tolerant_entry_jaccard(
                candidate, comparator, pd.Timedelta(hours=6)
            )
            item["six_hour_tolerant_jaccard"] = six_hour
            checks[f"{key}:exact_entry_jaccard"] = (
                exact <= thresholds["exact_entry_jaccard_max"]
            )
            checks[f"{key}:six_hour_tolerant_jaccard"] = (
                six_hour <= thresholds["six_hour_one_to_one_jaccard_max"]
            )
            checks[f"{key}:signed_occupancy_pearson"] = bool(
                correlation is not None
                and correlation
                <= thresholds["absolute_signed_occupancy_pearson_max"]
            )
        else:
            raise RuntimeError(
                f"DCLB-864 comparator clock family invalid: {family}"
            )
        report[key] = item
    return report, checks, decoded_rows


def evaluate_novelty(
    primary: pd.DataFrame,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], int]:
    decoded_row_counter = [0]
    try:
        return _evaluate_novelty_impl(
            primary,
            payload,
            decoded_row_counter,
        )
    except ComparatorContractFailure:
        raise
    except Exception as error:
        raise ComparatorContractFailure(
            "novelty_metric_contract",
            decoded_row_counter[0],
            str(error),
        ) from error


def _control_report(
    controls: Mapping[str, pd.DataFrame],
    raw_counts: Mapping[str, int],
    reserved_counts: Mapping[str, int],
) -> dict[str, Any]:
    primary = controls["primary"]
    return {
        name: {
            "raw_rows": int(raw_counts[name]),
            "globally_reserved_rows": int(reserved_counts[name]),
            "split_contained_rows": len(controls[name]),
            "train": clock_stats(_window(controls[name], "train")),
            "selection": clock_stats(_window(controls[name], "selection")),
            "same_side_reproduction_to_primary": {
                split: same_side_reproduction(
                    primary, controls[name], split
                )
                for split in ("train", "selection")
            },
        }
        for name in prereg.CONTROL_ORDER
    }


def _core_payload(
    states: pd.DataFrame,
    feature_funnel: Mapping[str, Any],
    controls: Mapping[str, pd.DataFrame],
    raw_counts: Mapping[str, int],
    reserved_counts: Mapping[str, int],
    source_audit: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    clock_bytes: bytes,
    *,
    clock_path: str | Path,
    artifact_eligible: bool,
    protocol_git_subprocess_calls: int,
) -> dict[str, Any]:
    statistics, source_checks, composition, composition_checks = support_checks(
        controls
    )
    source_passed = all(source_checks.values())
    composition_passed = bool(
        source_passed and all(composition_checks.values())
    )
    novelty_report: dict[str, Any] = {}
    novelty_checks: dict[str, bool] = {}
    comparator_rows_decoded = 0
    comparator_status = "not_opened_source_support_or_composition_failed"
    if composition_passed and artifact_eligible:
        try:
            (
                novelty_report,
                novelty_checks,
                comparator_rows_decoded,
            ) = evaluate_novelty(controls["primary"], preregistration)
            comparator_status = (
                "opened_after_complete_source_and_composition_pass"
            )
        except ComparatorContractFailure as error:
            comparator_rows_decoded = error.rows_decoded
            novelty_report = {
                "contract_failure": {
                    "code": error.code,
                    "message": str(error),
                    "comparator_rows_decoded": error.rows_decoded,
                }
            }
            novelty_checks = {
                f"comparator_contract:{error.code}": False,
            }
            comparator_status = (
                "opened_after_complete_source_and_composition_pass_"
                "then_failed_closed"
            )
    elif composition_passed:
        comparator_status = "synthetic_build_not_authorized"
    novelty_passed = bool(
        composition_passed
        and artifact_eligible
        and novelty_checks
        and all(novelty_checks.values())
    )
    first_stage, first_check = first_failure(
        source_checks,
        composition_checks,
        novelty_checks,
        artifact_eligible=artifact_eligible,
    )
    if not source_passed or not composition_passed:
        decision = (
            "retire_DCLB_864_unchanged_before_comparators_and_outcomes"
        )
    elif not artifact_eligible:
        decision = "synthetic_build_cannot_authorize_comparators_or_outcomes"
    elif not novelty_passed:
        decision = "retire_DCLB_864_unchanged_before_outcomes"
    else:
        decision = (
            "advance_to_separately_frozen_strict_economic_RLLM_evaluator"
        )
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": prereg.Policy().policy_id,
        "artifact_eligible": artifact_eligible,
        "outcomes_opened": False,
        "post_entry_return_computed": False,
        "funding_loaded": False,
        "source_incidence_opened": True,
        "source_rows_decoded": int(
            source_audit["h41_rows"]
            + source_audit["rrp_rows"]
            + source_audit["h8_rows"]
        ),
        "comparator_rows_decoded": comparator_rows_decoded,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
            "builder_v2": {
                "path": str(PREREGISTRATION_BUILDER),
                "sha256": PREREGISTRATION_BUILDER_SHA256,
            },
            "builder_v1": {
                "path": str(BASE_PREREGISTRATION_BUILDER),
                "sha256": BASE_PREREGISTRATION_BUILDER_SHA256,
            },
        },
        "implementation": {
            "source": str(SCRIPT_PATH),
            "source_sha256": sha256_file(SCRIPT_PATH),
            "test": str(TEST_PATH),
            "test_sha256": sha256_file(TEST_PATH),
            "contract": str(IMPLEMENTATION_CONTRACT),
            "contract_sha256": IMPLEMENTATION_CONTRACT_SHA256,
            "committed_clean_before_real_source": artifact_eligible,
        },
        "source_audit": dict(source_audit),
        "feature_funnel": dict(feature_funnel),
        "joint_state_rows": len(states),
        "controls": _control_report(
            controls, raw_counts, reserved_counts
        ),
        "source_statistics": statistics,
        "source_support_checks": source_checks,
        "source_support_passed": source_passed,
        "relational_composition": composition,
        "relational_composition_checks": composition_checks,
        "relational_composition_passed": composition_passed,
        "comparator_status": comparator_status,
        "novelty": novelty_report,
        "novelty_checks": novelty_checks,
        "novelty_passed": novelty_passed,
        "first_failing_stage": first_stage,
        "first_failing_check": first_check,
        "decision": decision,
        "clock": {
            "path": str(clock_path),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "rows": sum(len(controls[name]) for name in prereg.CONTROL_ORDER),
            "columns": list(CLOCK_COLUMNS),
            "control_counts": {
                name: len(controls[name]) for name in prereg.CONTROL_ORDER
            },
            "deterministic_gzip_mtime_zero": True,
        },
        "outcome_boundary": {
            "btc_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "future_return_rows_computed": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_source_rows_loaded": 0,
            "network_calls": 0,
            "external_data_subprocess_calls": 0,
            "protocol_git_subprocess_calls": protocol_git_subprocess_calls,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def _build_support(
    h41: pd.DataFrame,
    rrp: pd.DataFrame,
    h8: pd.DataFrame,
    *,
    source_audit: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    clock_path: str | Path,
    artifact_eligible: bool,
    protocol_git_subprocess_calls: int,
) -> tuple[dict[str, Any], bytes]:
    states, funnel = build_joint_states(h41, rrp, h8)
    controls, raw_counts, reserved_counts = build_controls(states, h8)
    clock_bytes = deterministic_clock_bytes(controls)
    report = _core_payload(
        states,
        funnel,
        controls,
        raw_counts,
        reserved_counts,
        source_audit,
        preregistration,
        clock_bytes,
        clock_path=clock_path,
        artifact_eligible=artifact_eligible,
        protocol_git_subprocess_calls=protocol_git_subprocess_calls,
    )
    return report, clock_bytes


def build_support_from_frames(
    h41: pd.DataFrame,
    rrp: pd.DataFrame,
    h8: pd.DataFrame,
) -> tuple[dict[str, Any], bytes]:
    validated_h41 = validate_h41_source(h41)
    validated_rrp = validate_rrp_source(rrp)
    validated_h8 = validate_h8_source(h8)
    source_audit = {
        "kind": "synthetic_or_injected",
        "h41_rows": len(validated_h41),
        "rrp_rows": len(validated_rrp),
        "h8_rows": len(validated_h8),
        "bindings": {},
    }
    return _build_support(
        validated_h41,
        validated_rrp,
        validated_h8,
        source_audit=source_audit,
        preregistration=prereg.build_manifest(),
        clock_path=DEFAULT_CLOCK_OUTPUT,
        artifact_eligible=False,
        protocol_git_subprocess_calls=0,
    )


def validate_report(payload: Mapping[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("DCLB-864 support report manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("DCLB-864 support report opened outcomes")
    if payload.get("post_entry_return_computed") is not False:
        raise RuntimeError("DCLB-864 support report computed a future return")
    if payload.get("funding_loaded") is not False:
        raise RuntimeError("DCLB-864 support report loaded funding")
    boundary = payload["outcome_boundary"]
    for name in (
        "btc_market_rows_loaded",
        "funding_rows_loaded",
        "future_return_rows_computed",
        "return_or_pnl_fields_read",
        "post_2023_source_rows_loaded",
        "network_calls",
        "external_data_subprocess_calls",
    ):
        if boundary.get(name) != 0:
            raise RuntimeError(f"DCLB-864 forbidden evidence opened: {name}")
    if (
        not payload["source_support_passed"]
        or not payload["relational_composition_passed"]
    ) and payload["comparator_rows_decoded"] != 0:
        raise RuntimeError(
            "DCLB-864 comparator rows opened before source/composition pass"
        )


def _output_relative_path(path: str | Path) -> Path:
    candidate = Path(path)
    raw = str(path)
    if (
        raw.startswith("~")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("DCLB-864 output path must be repository-relative")
    return candidate


def _open_output_parent(candidate: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        current_fd = os.open(REPOSITORY_ROOT, flags)
    except OSError as error:
        raise RuntimeError("DCLB-864 repository root is unsafe") from error
    try:
        for part in candidate.parent.parts:
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError as error:
                raise RuntimeError(
                    "DCLB-864 output parent missing or symlinked"
                ) from error
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        raise
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError("DCLB-864 output is symlinked or invalid") from error
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("DCLB-864 output is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_once(path: str | Path, payload: bytes) -> str:
    output = _output_relative_path(path)
    directory_fd = _open_output_parent(output)
    temporary_name = (
        f".{output.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    temporary_created = False
    try:
        try:
            existing = _read_regular_at(directory_fd, output.name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != payload:
                raise RuntimeError("DCLB-864 existing artifact is noncanonical")
            return "verified_existing"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                output.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_regular_at(directory_fd, output.name) != payload:
                raise RuntimeError("DCLB-864 artifact race drift")
            return "verified_existing"
        os.fsync(directory_fd)
        return "created"
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            os.fsync(directory_fd)
        os.close(directory_fd)


def canonical_report_bytes(report: Mapping[str, Any]) -> bytes:
    validate_report(report)
    return (
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def run(
    *,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK_OUTPUT,
) -> dict[str, Any]:
    _assert_protocol_committed()
    preregistration = validate_preregistration()
    bindings = verify_pre_source_bindings(preregistration)
    h41 = load_h41_source()
    rrp = load_rrp_source()
    h8 = load_h8_source()
    source_audit = {
        "kind": "frozen_repository_sources",
        "h41_rows": len(h41),
        "rrp_rows": len(rrp),
        "h8_rows": len(h8),
        "bindings": bindings,
        "source_value_allowlists": {
            "h41": list(prereg.H41_ALLOWLIST),
            "rrp": list(prereg.RRP_ALLOWLIST),
            "h8": list(prereg.H8_ALLOWLIST),
        },
    }
    report, clock_bytes = _build_support(
        h41,
        rrp,
        h8,
        source_audit=source_audit,
        preregistration=preregistration,
        clock_path=clock_output,
        artifact_eligible=True,
        protocol_git_subprocess_calls=2,
    )
    report_bytes = canonical_report_bytes(report)
    _write_once(clock_output, clock_bytes)
    _write_once(report_output, report_bytes)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_OUTPUT))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK_OUTPUT))
    args = parser.parse_args()
    report = run(
        report_output=args.report_output,
        clock_output=args.clock_output,
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
