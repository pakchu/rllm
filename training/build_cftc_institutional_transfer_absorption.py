"""Freeze CFTC Bitcoin TFF positioning for CITA-1 without market outcomes.

The source is the official annual Traders in Financial Futures futures-only
archive. Published participant changes are checked against adjacent report
levels. Availability is deliberately conservative and includes the documented
2023 ION publication backlog.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ARCHIVE_PAGE_URL = (
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/"
    "HistoricalCompressed/index.htm"
)
SCHEDULE_URL = (
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/"
    "ReleaseSchedule/index.htm"
)
SPECIAL_ANNOUNCEMENTS_URL = (
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/"
    "HistoricalSpecialAnnouncements/index.htm"
)
VARIABLE_NAMES_URL = (
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/"
    "HistoricalViewable/cotvariablestfm.html"
)
EXPLANATORY_NOTES_URL = (
    "https://www.cftc.gov/idc/groups/public/%40commitmentsoftraders/"
    "documents/file/tfmexplanatorynotes.pdf"
)
ZIP_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
USER_AGENT = "rllm-causal-research/1.0 (+https://github.com/pakchu/rllm)"
CONTRACT_CODE = "133741"
MARKET_NAME = "BITCOIN - CHICAGO MERCANTILE EXCHANGE"
EXPECTED_ROWS_BY_YEAR = {
    2018: 39,
    2019: 52,
    2020: 52,
    2021: 52,
    2022: 52,
    2023: 52,
}
SPECIAL_PUBLICATION_DATES = {
    date(2023, 1, 31): date(2023, 2, 24),
    date(2023, 2, 7): date(2023, 3, 3),
    date(2023, 2, 14): date(2023, 3, 8),
    date(2023, 2, 21): date(2023, 3, 10),
    date(2023, 2, 28): date(2023, 3, 14),
    date(2023, 3, 7): date(2023, 3, 16),
    date(2023, 3, 14): date(2023, 3, 21),
}
FetchBytes = Callable[[str], bytes]


@dataclass(frozen=True)
class BuildConfig:
    start_year: int = 2018
    end_year: int = 2023
    output_dir: str = "data/cftc_institutional_transfer_absorption_2018_2023"
    timeout_seconds: int = 60
    retries: int = 5
    cache_dir: str = "/tmp/rllm_cftc_tff_cache"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def write_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))


def fetch_bytes(url: str, *, timeout_seconds: int = 60, retries: int = 5) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/zip,application/octet-stream;q=0.9,*/*;q=0.1",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                return response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to retrieve {url}") from last_error


def _normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            column: column.strip().lower().replace("-", "_")
            for column in frame.columns
        }
    )


def parse_annual_zip(payload: bytes, *, year: int) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if names != ["FinFutYY.txt"]:
            raise ValueError(f"unexpected CFTC TFF archive members for {year}: {names}")
        frame = pd.read_csv(
            io.BytesIO(archive.read(names[0])),
            dtype=str,
            na_values=["."],
            keep_default_na=True,
        )
    frame = _normalized_columns(frame)
    required = {
        "market_and_exchange_names",
        "report_date_as_yyyy_mm_dd",
        "cftc_contract_market_code_quotes",
        "open_interest_all",
        "dealer_positions_long_all",
        "dealer_positions_short_all",
        "asset_mgr_positions_long_all",
        "asset_mgr_positions_short_all",
        "lev_money_positions_long_all",
        "lev_money_positions_short_all",
        "change_in_dealer_long_all",
        "change_in_dealer_short_all",
        "change_in_asset_mgr_long_all",
        "change_in_asset_mgr_short_all",
        "change_in_lev_money_long_all",
        "change_in_lev_money_short_all",
    }
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"CFTC TFF schema changed for {year}: {missing}")
    code = (
        frame["cftc_contract_market_code_quotes"]
        .astype(str)
        .str.strip()
        .str.strip('"')
    )
    name = frame["market_and_exchange_names"].astype(str).str.strip()
    selected = frame.loc[code.eq(CONTRACT_CODE) & name.eq(MARKET_NAME)].copy()
    expected = EXPECTED_ROWS_BY_YEAR.get(year)
    if expected is not None and len(selected) != expected:
        raise ValueError(
            f"unexpected CFTC Bitcoin row count for {year}: {len(selected)}"
        )
    selected["source_year"] = year
    selected["official_zip_url"] = ZIP_URL_TEMPLATE.format(year=year)
    return selected.reset_index(drop=True)


def conservative_available_time_utc(report_date: date) -> pd.Timestamp:
    publication = SPECIAL_PUBLICATION_DATES.get(report_date)
    if publication is not None:
        # The special-announcement page binds the actual catch-up publication
        # date. Midnight on the following UTC day is after the U.S. release day.
        value = datetime.combine(
            publication + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
    else:
        # CFTC states Friday 15:30 ET normally, with federal holidays delaying
        # release one or two days. Tuesday report date +8d 00:00 UTC is later
        # than even a two-day-delayed release and avoids historical clock guesswork.
        value = datetime.combine(
            report_date + timedelta(days=8),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
    return cast(pd.Timestamp, pd.Timestamp(value))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))
    finite = values.dropna()
    if not finite.empty and not bool((finite % 1.0).eq(0.0).all()):
        raise ValueError(f"CFTC position column is not integral: {column}")
    return values


def build_panel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("no CFTC annual frames")
    raw = pd.concat(frames, ignore_index=True)
    raw["report_date"] = pd.to_datetime(
        raw["report_date_as_yyyy_mm_dd"], errors="raise"
    )
    raw = raw.sort_values("report_date").reset_index(drop=True)
    if raw["report_date"].duplicated().any():
        raise ValueError("CFTC Bitcoin report dates are duplicated")
    if not raw["report_date"].is_monotonic_increasing:
        raise ValueError("CFTC Bitcoin reports are not chronological")

    output = pd.DataFrame(
        {
            "report_date": raw["report_date"].dt.strftime("%Y-%m-%d"),
            "market_and_exchange_names": raw["market_and_exchange_names"]
            .astype(str)
            .str.strip(),
            "cftc_contract_market_code": CONTRACT_CODE,
            "source_year": raw["source_year"].astype(int),
            "official_zip_url": raw["official_zip_url"],
        }
    )
    output["available_time_utc"] = [
        conservative_available_time_utc(timestamp.date()).isoformat()
        for timestamp in raw["report_date"]
    ]
    output["special_publication_override"] = [
        timestamp.date() in SPECIAL_PUBLICATION_DATES
        for timestamp in raw["report_date"]
    ]

    source_columns = {
        "open_interest": "open_interest_all",
        "dealer_long": "dealer_positions_long_all",
        "dealer_short": "dealer_positions_short_all",
        "asset_mgr_long": "asset_mgr_positions_long_all",
        "asset_mgr_short": "asset_mgr_positions_short_all",
        "lev_money_long": "lev_money_positions_long_all",
        "lev_money_short": "lev_money_positions_short_all",
        "dealer_long_published_change": "change_in_dealer_long_all",
        "dealer_short_published_change": "change_in_dealer_short_all",
        "asset_mgr_long_published_change": "change_in_asset_mgr_long_all",
        "asset_mgr_short_published_change": "change_in_asset_mgr_short_all",
        "lev_money_long_published_change": "change_in_lev_money_long_all",
        "lev_money_short_published_change": "change_in_lev_money_short_all",
    }
    for output_name, source_name in source_columns.items():
        output[output_name] = _numeric(raw, source_name)

    consistency: list[str] = []
    for participant in ("dealer", "asset_mgr", "lev_money"):
        output[f"{participant}_net"] = (
            output[f"{participant}_long"] - output[f"{participant}_short"]
        )
        output[f"{participant}_published_net_change"] = (
            output[f"{participant}_long_published_change"]
            - output[f"{participant}_short_published_change"]
        )
        output[f"{participant}_arithmetic_net_change"] = output[
            f"{participant}_net"
        ].diff()
        for side in ("long", "short"):
            name = f"{participant}_{side}_change_consistent"
            output[name] = output[
                f"{participant}_{side}_published_change"
            ].eq(output[f"{participant}_{side}"].diff())
            consistency.append(name)
        net_name = f"{participant}_net_change_consistent"
        output[net_name] = output[f"{participant}_published_net_change"].eq(
            output[f"{participant}_arithmetic_net_change"]
        )
        consistency.append(net_name)

    required_numeric = [
        "open_interest",
        "asset_mgr_published_net_change",
        "lev_money_published_net_change",
    ]
    output["source_complete"] = (
        output[required_numeric].notna().all(axis=1)
        & output[consistency].all(axis=1)
        & output["open_interest"].gt(0.0)
    )
    available = pd.to_datetime(output["available_time_utc"], utc=True)
    if not available.is_monotonic_increasing or available.duplicated().any():
        raise ValueError("CFTC conservative availability clock is invalid")
    return output


def build(config: BuildConfig) -> dict[str, Any]:
    if config.start_year != 2018 or config.end_year != 2023:
        raise ValueError("CITA-1 source horizon is frozen to 2018-2023")
    output_dir = Path(config.output_dir)
    raw_dir = output_dir / "raw"
    cache_dir = Path(config.cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    payloads: dict[int, bytes] = {}
    frames: list[pd.DataFrame] = []
    for year in range(config.start_year, config.end_year + 1):
        url = ZIP_URL_TEMPLATE.format(year=year)
        cache_path = cache_dir / f"fut_fin_txt_{year}.zip"
        if cache_path.exists():
            payload = cache_path.read_bytes()
        else:
            payload = fetch_bytes(
                url,
                timeout_seconds=config.timeout_seconds,
                retries=config.retries,
            )
            cache_path.write_bytes(payload)
        payloads[year] = payload
        frames.append(parse_annual_zip(payload, year=year))
        (raw_dir / f"fut_fin_txt_{year}.zip").write_bytes(payload)

    panel = build_panel(frames)
    output_path = output_dir / "cftc_institutional_transfer_absorption_2018_2023.csv.gz"
    write_gzip(output_path, panel.to_csv(index=False, lineterminator="\n").encode())

    source_manifest = {
        "protocol_version": "cftc_institutional_transfer_absorption_source_v1",
        "official_urls": {
            "historical_compressed": ARCHIVE_PAGE_URL,
            "release_schedule": SCHEDULE_URL,
            "special_announcements": SPECIAL_ANNOUNCEMENTS_URL,
            "variable_names": VARIABLE_NAMES_URL,
            "explanatory_notes": EXPLANATORY_NOTES_URL,
            "annual_archives": {
                str(year): ZIP_URL_TEMPLATE.format(year=year)
                for year in payloads
            },
        },
        "source_contract": (
            "official annual TFF futures-only rows for CME Bitcoin contract 133741; "
            "published changes reconciled to adjacent position levels"
        ),
        "availability_contract": (
            "report date +8d 00:00 UTC, except the seven documented 2023 ION "
            "catch-up reports, which use actual publication date + next UTC midnight"
        ),
        "special_publication_dates": {
            report.isoformat(): publication.isoformat()
            for report, publication in SPECIAL_PUBLICATION_DATES.items()
        },
        "annual_zip_sha256": {
            str(year): sha256_bytes(payload) for year, payload in payloads.items()
        },
        "raw_snapshots": {
            str(raw_dir / f"fut_fin_txt_{year}.zip"): sha256_file(
                raw_dir / f"fut_fin_txt_{year}.zip"
            )
            for year in payloads
        },
        "known_bitcoin_revision_notice": None,
        "archive_revision_limitation": (
            "annual compressed files are official consolidated archives, not "
            "byte captures from every original publication; no contract-133741 "
            "revision was identified in the official special-announcement ledger"
        ),
    }
    source_manifest_path = output_dir / "source_manifest.json"
    source_manifest_path.write_bytes(canonical_json(source_manifest))

    year_counts = panel["report_date"].str[:4].value_counts().sort_index()
    quarantined = panel.loc[~panel["source_complete"].astype(bool), "report_date"]
    build_manifest = {
        "protocol_version": "cftc_institutional_transfer_absorption_build_v1",
        "config": {
            key: value for key, value in asdict(config).items() if key != "cache_dir"
        },
        "rows": int(len(panel)),
        "rows_by_report_year": {
            str(year): int(count) for year, count in year_counts.items()
        },
        "first_report_date": str(panel.iloc[0]["report_date"]),
        "last_report_date": str(panel.iloc[-1]["report_date"]),
        "first_available_time_utc": str(panel.iloc[0]["available_time_utc"]),
        "last_available_time_utc": str(panel.iloc[-1]["available_time_utc"]),
        "source_complete_rows": int(panel["source_complete"].sum()),
        "source_quarantined_rows": int((~panel["source_complete"]).sum()),
        "quarantined_report_dates": quarantined.astype(str).tolist(),
        "special_publication_overrides": int(
            panel["special_publication_override"].sum()
        ),
        "market_or_funding_rows_read": 0,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
    }
    build_manifest["manifest_hash"] = sha256_bytes(canonical_json(build_manifest))
    (output_dir / "build_manifest.json").write_bytes(canonical_json(build_manifest))
    return build_manifest


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=BuildConfig.start_year)
    parser.add_argument("--end-year", type=int, default=BuildConfig.end_year)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--cache-dir", default=BuildConfig.cache_dir)
    return BuildConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
