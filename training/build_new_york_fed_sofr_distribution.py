"""Freeze a causal, source-only New York Fed SOFR distribution panel.

The official historical endpoint reports an ``effectiveDate`` for the repo
transactions summarized by SOFR.  That rate is published on the next SOFR
business day.  Because the New York Fed may revise a published rate later on
the publication day, the rate is conservatively timestamped at 15:00
America/New_York, after the documented 14:30 ET revision window.

The historical endpoint can expose quarterly-updated percentile and volume
summary statistics.  They therefore receive a separate, deliberately lagged
availability timestamp: the start of the second quarter after their effective
quarter.  A downstream evaluator must never substitute the earlier rate
timestamp for the summary-statistics timestamp.

No crypto price, return, label, or portfolio outcome is imported or derived.
The builder is deliberately capped at 2023 so the current pre-2024 research
protocol cannot accidentally open a sealed outcome period.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as wall_time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo


BASE_URL = "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
USER_AGENT = "rllm-sofr-source-freeze/1.0"
SCHEMA_VERSION = 1
MAX_RESEARCH_YEAR = 2023
SOURCE_SNAPSHOT_DATE = "2026-07-17"
NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

FROZEN_YEAR_COVERAGE: dict[int, tuple[int, str, str]] = {
    2018: (188, "2018-04-02", "2018-12-31"),
    2019: (250, "2019-01-02", "2019-12-31"),
    2020: (251, "2020-01-02", "2020-12-31"),
    2021: (250, "2021-01-04", "2021-12-31"),
    2022: (249, "2022-01-03", "2022-12-30"),
    2023: (249, "2023-01-03", "2023-12-29"),
}
FROZEN_RESPONSE_SHA256 = {
    2018: "0652bd78da1372b5a4a89eb10b54148b6cc0dabf528a139a2ac23b0f7052e46f",
    2019: "8ef5f77a6bcf145b83ffc3600beffc1f96ee7d3be17ce16d892a7bcb78926c60",
    2020: "6303924b8ba2eb14fb35c84586cc7ccfa9b62227307f145492902ccdcd86f1d5",
    2021: "1d81e7212f3791a9691e77d552bc933b17b1aacdf11a0200a056961e60c3f8cd",
    2022: "db885b1e5fcb68859e56428441835e8b37e2ebbe227ddc06cc95f62fd1fddd0c",
    2023: "e2fcdcea3fbf9dd6b160d8e2ae5ce27463a9fba0642bfdcfbbe38b415c0e461c",
}

RAW_FIELDS = (
    "effectiveDate",
    "type",
    "percentRate",
    "percentPercentile1",
    "percentPercentile25",
    "percentPercentile75",
    "percentPercentile99",
    "volumeInBillions",
    "revisionIndicator",
)
OUTPUT_COLUMNS = (
    "effective_date",
    "publication_date",
    "sofr_available_at_utc",
    "summary_available_at_utc",
    "sofr_percent",
    "percentile_1_percent",
    "percentile_25_percent",
    "percentile_75_percent",
    "percentile_99_percent",
    "volume_usd_billions",
    "revision_indicator",
    "source_complete",
)


@dataclass(frozen=True)
class BuildConfig:
    start_year: int = 2018
    end_year: int = 2023
    output_dir: str = "data/new_york_fed_sofr_distribution_2018_2023"
    retries: int = 5
    timeout_seconds: int = 60


def annual_url(year: int, *, base_url: str = BASE_URL) -> str:
    params = {
        "startDate": f"{year:04d}-01-01",
        "endDate": f"{year:04d}-12-31",
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def _fetch_bytes(url: str, *, retries: int, timeout: int) -> bytes:
    error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt + 1 < retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts") from error


def _parse_date(value: Any, *, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string, got {value!r}")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date string, got {value!r}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} is not a canonical ISO date: {value!r}")
    return parsed


def _parse_finite(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite numeric, got {value!r}")
    return number


def _parse_optional_finite(value: Any, *, field: str) -> float | None:
    if value in (None, "", "NA"):
        return None
    return _parse_finite(value, field=field)


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    if value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def parse_annual_response(payload: bytes, *, year: int) -> list[dict[str, Any]]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("NY Fed response is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict) or not isinstance(document.get("refRates"), list):
        raise ValueError("NY Fed response must contain a refRates list")

    normalized: list[dict[str, Any]] = []
    for raw in document["refRates"]:
        if not isinstance(raw, dict):
            raise ValueError(f"NY Fed refRates row must be an object, got {raw!r}")
        missing = set(RAW_FIELDS).difference(raw)
        if missing:
            raise ValueError(f"NY Fed row is missing required fields: {sorted(missing)}")
        effective = _parse_date(raw["effectiveDate"], field="effectiveDate")
        if effective.year != year:
            raise ValueError(
                f"NY Fed row {effective} falls outside requested year {year}"
            )
        if raw["type"] != "SOFR":
            raise ValueError(f"expected SOFR row type, got {raw['type']!r}")

        rate = _parse_finite(raw["percentRate"], field="percentRate")
        p1 = _parse_optional_finite(
            raw["percentPercentile1"], field="percentPercentile1"
        )
        p25 = _parse_optional_finite(
            raw["percentPercentile25"], field="percentPercentile25"
        )
        p75 = _parse_optional_finite(
            raw["percentPercentile75"], field="percentPercentile75"
        )
        p99 = _parse_optional_finite(
            raw["percentPercentile99"], field="percentPercentile99"
        )
        volume = _parse_finite(raw["volumeInBillions"], field="volumeInBillions")
        if volume <= 0.0:
            raise ValueError(f"volumeInBillions must be positive, got {volume!r}")
        quantiles = (p1, p25, rate, p75, p99)
        optional_quantiles = (p1, p25, p75, p99)
        present_count = sum(value is not None for value in optional_quantiles)
        if present_count not in (0, len(optional_quantiles)):
            raise ValueError(
                f"SOFR distribution is partially missing on {effective}: "
                f"{optional_quantiles}"
            )
        complete = present_count == len(optional_quantiles)
        if complete and not all(
            left <= right
            for left, right in zip(quantiles, quantiles[1:])
        ):
            raise ValueError(
                f"SOFR distribution is not ordered on {effective}: {quantiles}"
            )
        revision = raw["revisionIndicator"]
        if revision is None:
            revision = ""
        if not isinstance(revision, str):
            raise ValueError(
                f"revisionIndicator must be text or null, got {revision!r}"
            )
        normalized.append(
            {
                "effective_date": effective,
                "sofr_percent": rate,
                "percentile_1_percent": p1,
                "percentile_25_percent": p25,
                "percentile_75_percent": p75,
                "percentile_99_percent": p99,
                "volume_usd_billions": volume,
                "revision_indicator": revision.strip(),
                "source_complete": complete,
            }
        )
    normalized.sort(key=lambda row: row["effective_date"])
    dates = [row["effective_date"] for row in normalized]
    if len(set(dates)) != len(dates):
        raise ValueError(f"NY Fed response contains duplicate dates for {year}")
    return normalized


def _validate_frozen_coverage(rows: list[dict[str, Any]], *, year: int) -> None:
    expected = FROZEN_YEAR_COVERAGE.get(year)
    if expected is None:
        raise ValueError(f"no frozen source coverage contract for year {year}")
    count, first, last = expected
    actual = (
        len(rows),
        rows[0]["effective_date"].isoformat() if rows else None,
        rows[-1]["effective_date"].isoformat() if rows else None,
    )
    if actual != (count, first, last):
        raise ValueError(
            f"NY Fed {year} coverage changed: expected={(count, first, last)}, "
            f"actual={actual}"
        )


def _validate_frozen_payload(payload: bytes, *, year: int) -> str:
    actual = hashlib.sha256(payload).hexdigest()
    expected = FROZEN_RESPONSE_SHA256.get(year)
    if expected is None:
        raise ValueError(f"no frozen response hash contract for year {year}")
    if actual != expected:
        raise ValueError(
            f"NY Fed {year} response changed from the frozen snapshot: "
            f"expected={expected}, actual={actual}"
        )
    return actual


def causal_panel(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: row["effective_date"])
    dates = [row["effective_date"] for row in ordered]
    if len(set(dates)) != len(dates):
        raise ValueError("combined SOFR source contains duplicate effective dates")
    gaps = [(right - left).days for left, right in zip(dates, dates[1:])]
    if any(gap < 1 or gap > 4 for gap in gaps):
        raise ValueError(f"combined SOFR source has an unexpected business-day gap: {gaps}")

    output: list[dict[str, str]] = []
    # The last observation is retained in source provenance but not emitted:
    # its next publication day is outside the bounded source panel.
    for row, publication_date in zip(ordered, dates[1:]):
        rate_available = datetime.combine(
            publication_date,
            wall_time(hour=15),
            tzinfo=NEW_YORK,
        )
        effective = row["effective_date"]
        quarter_start_month = 3 * ((effective.month - 1) // 3) + 1
        summary_month_index = (effective.year * 12 + quarter_start_month - 1) + 6
        summary_year, summary_month_zero = divmod(summary_month_index, 12)
        summary_available = datetime(
            summary_year,
            summary_month_zero + 1,
            1,
            21,
            tzinfo=UTC,
        )
        output.append(
            {
                "effective_date": row["effective_date"].isoformat(),
                "publication_date": publication_date.isoformat(),
                "sofr_available_at_utc": rate_available.astimezone(UTC).isoformat(),
                "summary_available_at_utc": summary_available.isoformat(),
                "sofr_percent": _format_number(row["sofr_percent"]),
                "percentile_1_percent": _format_number(row["percentile_1_percent"]),
                "percentile_25_percent": _format_number(row["percentile_25_percent"]),
                "percentile_75_percent": _format_number(row["percentile_75_percent"]),
                "percentile_99_percent": _format_number(row["percentile_99_percent"]),
                "volume_usd_billions": _format_number(row["volume_usd_billions"]),
                "revision_indicator": row["revision_indicator"],
                "source_complete": "true" if row["source_complete"] else "false",
            }
        )
    return output


def _write_gzip_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text,
                    fieldnames=list(OUTPUT_COLUMNS),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)


def _validate_config(cfg: BuildConfig) -> None:
    if cfg.start_year < 2018:
        raise ValueError("SOFR source starts in 2018")
    if cfg.start_year > cfg.end_year:
        raise ValueError("start_year must not exceed end_year")
    if cfg.end_year > MAX_RESEARCH_YEAR:
        raise ValueError("2024+ is sealed by the current research protocol")
    if cfg.retries < 1 or cfg.timeout_seconds < 1:
        raise ValueError("retries and timeout_seconds must be positive")


def build(
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    _validate_config(cfg)
    all_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for year in range(cfg.start_year, cfg.end_year + 1):
        url = annual_url(year)
        payload = fetcher(url, retries=cfg.retries, timeout=cfg.timeout_seconds)
        rows = parse_annual_response(payload, year=year)
        _validate_frozen_coverage(rows, year=year)
        response_sha256 = _validate_frozen_payload(payload, year=year)
        all_rows.extend(rows)
        output_dir = Path(cfg.output_dir)
        raw_path = output_dir / "raw" / f"new_york_fed_sofr_{year}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payload)
        sources.append(
            {
                "year": year,
                "url": url,
                "snapshot_date": SOURCE_SNAPSHOT_DATE,
                "raw_path": str(raw_path),
                "response_sha256": response_sha256,
                "rows": len(rows),
                "first_effective_date": rows[0]["effective_date"].isoformat(),
                "last_effective_date": rows[-1]["effective_date"].isoformat(),
            }
        )

    panel = causal_panel(all_rows)
    output_path = output_dir / (
        "new_york_fed_sofr_distribution_"
        f"{panel[0]['effective_date']}_{panel[-1]['effective_date']}.csv.gz"
    )
    _write_gzip_csv(output_path, panel)
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    complete_rows = sum(row["source_complete"] == "true" for row in panel)
    revised_rows = sum(bool(row["revision_indicator"]) for row in panel)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": asdict(cfg),
        "protocol": {
            "source": "Federal Reserve Bank of New York Markets Data API",
            "measure": "SOFR rate distribution and transaction volume",
            "effective_date_semantics": "date of underlying repo transactions",
            "publication_date_rule": "next observed SOFR effective date",
            "sofr_rate_availability_rule": (
                "15:00 America/New_York on publication_date, after the documented "
                "same-day 14:30 ET revision window"
            ),
            "summary_availability_rule": (
                "21:00 UTC on the first day of the second quarter after the "
                "effective quarter; this is a conservative vintage guard for "
                "the NY Fed's quarterly-updated percentiles and volume"
            ),
            "last_fetched_row_emitted": False,
            "missing_distribution_policy": "retain row as source_complete=false; never fill",
            "future_year_guard": "2024+ fetches are rejected",
            "crypto_market_fields_opened": False,
            "outcomes_opened": False,
        },
        "output": str(output_path),
        "output_sha256": output_sha256,
        "fetched_rows": len(all_rows),
        "rows": len(panel),
        "complete_rows": complete_rows,
        "incomplete_rows": len(panel) - complete_rows,
        "revised_rows": revised_rows,
        "dropped_without_bounded_publication_date": len(all_rows) - len(panel),
        "first_effective_date": panel[0]["effective_date"],
        "last_effective_date": panel[-1]["effective_date"],
        "first_sofr_available_at_utc": panel[0]["sofr_available_at_utc"],
        "last_sofr_available_at_utc": panel[-1]["sofr_available_at_utc"],
        "first_summary_available_at_utc": panel[0]["summary_available_at_utc"],
        "last_summary_available_at_utc": panel[-1]["summary_available_at_utc"],
        "columns": list(OUTPUT_COLUMNS),
        "sources": sources,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=BuildConfig.start_year)
    parser.add_argument("--end-year", type=int, default=BuildConfig.end_year)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument(
        "--timeout-seconds", type=int, default=BuildConfig.timeout_seconds
    )
    manifest = build(BuildConfig(**vars(parser.parse_args())))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
