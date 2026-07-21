"""Build the outcome-blind DEWH-144 Deribit strike-wall source panel."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from training import download_deribit_btc_option_deliveries as delivery


POLICY_ID = "DEWH-144"
PROTOCOL_VERSION = "deribit_expiry_wall_handoff_source_v1"
MECHANISM_DOCUMENT = Path(
    "docs/deribit-expiry-wall-handoff-mechanism-decision-2026-07-21.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "f9c0029a6b9a7f0aa3015bc352974ec76238a2f3eb34887ed649d02d5cfaf0bb"
)
SHARED_DOWNLOADER = Path("training/download_deribit_btc_option_deliveries.py")
SHARED_DOWNLOADER_SHA256 = (
    "aa925828cf8350ed522c0ac559c64faed90fc049b99228d60b349d2771b1cd4c"
)
FROZEN_PRE_REFACTOR_DOWNLOADER_SHA256 = (
    "1e698db869ef263b692a950a3ecc4f4fafb834dd99db8476fd4da11bc1852cda"
)
SOURCE_COLUMNS = (
    "expiry_time",
    "delivery_event_time",
    "source_observation_earliest",
    "index_price",
    "distinct_strike_count",
    "total_position",
    "dominant_strike",
    "dominant_strike_position",
    "wall_share",
    "strike_position_hhi",
    "largest_individual_instrument_share",
    "local_log_spacing",
    "signed_normalized_wall_distance",
    "wall_tie_count",
    "delivery_delay_seconds",
    "maximum_event_row_span_seconds",
)


@dataclass(frozen=True)
class Config:
    output_csv: str = (
        "data/deribit_btc_expiry_wall_2019_2023/"
        "BTC_deribit_expiry_wall_2019-01-01_2023-12-31.csv.gz"
    )
    manifest_output: str = (
        "data/deribit_btc_expiry_wall_2019_2023/build_manifest.json"
    )
    start: str = "2019-01-01"
    end_exclusive: str = "2024-01-01"
    currency: str = "BTC"
    settlement_type: str = "delivery"
    page_size: int = 1000
    timeout_sec: float = 30.0
    request_pause_sec: float = 0.20
    maximum_retries: int = 8
    maximum_event_row_span_seconds: float = 5.0


Fetch = Callable[[dict[str, Any]], dict[str, Any]]


def _settlement_rows_commitment(rows: list[dict[str, Any]]) -> str:
    """Commit stable source rows, excluding request-specific API timing fields."""
    return delivery.canonical_hash(rows)


def _delivery_config(cfg: Config) -> delivery.Config:
    return delivery.Config(
        output_csv=cfg.output_csv,
        manifest_output=cfg.manifest_output,
        start=cfg.start,
        end_exclusive=cfg.end_exclusive,
        currency=cfg.currency,
        settlement_type=cfg.settlement_type,
        page_size=cfg.page_size,
        timeout_sec=cfg.timeout_sec,
        request_pause_sec=cfg.request_pause_sec,
        maximum_retries=cfg.maximum_retries,
        maximum_event_row_span_seconds=cfg.maximum_event_row_span_seconds,
    )


def _normalised_options(
    rows: list[dict[str, Any]],
    cfg: delivery.Config,
) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, Any]] = []
    futures = 0
    for row in rows:
        parsed = delivery._normalise_option(row, cfg)
        if parsed is None:
            futures += 1
        else:
            records.append(parsed)
    if not records:
        raise RuntimeError("DEWH source has no BTC option delivery rows")
    options = pd.DataFrame.from_records(records)
    if options.duplicated(["expiry_time", "instrument_name"]).any():
        raise RuntimeError("DEWH source contains duplicate instruments")
    options = options.sort_values(
        ["expiry_time", "strike", "option_type"],
        kind="mergesort",
        ignore_index=True,
    )
    return options, futures


def _year_counts(values: pd.Series) -> dict[str, int]:
    timestamps = pd.to_datetime(values, utc=True, errors="raise")
    return {
        str(year): int(count)
        for year, count in timestamps.dt.year.value_counts().sort_index().items()
    }


def aggregate_wall_deliveries(
    rows: list[dict[str, Any]],
    cfg: delivery.Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate raw delivery rows by expiry without retaining instruments."""
    options, futures = _normalised_options(rows, cfg)
    records: list[dict[str, Any]] = []
    invalid_too_few_strikes = 0
    invalid_tied_wall = 0
    invalid_spacing = 0
    total_expiries = 0
    maximum_span = 0.0
    maximum_delay = 0.0
    delayed_expiries = 0

    for expiry_time, group in options.groupby("expiry_time", sort=True):
        total_expiries += 1
        raw_span = float(
            (
                cast(pd.Timestamp, group["raw_timestamp"].max())
                - cast(pd.Timestamp, group["raw_timestamp"].min())
            ).total_seconds()
        )
        if raw_span > cfg.maximum_event_row_span_seconds:
            raise RuntimeError("DEWH option rows disagree on delivery clock")
        maximum_span = max(maximum_span, raw_span)
        delivery_event_time = cast(pd.Timestamp, group["raw_timestamp"].max())
        expiry = cast(pd.Timestamp, expiry_time)
        delay = float((delivery_event_time - expiry).total_seconds())
        maximum_delay = max(maximum_delay, delay)
        delayed_expiries += int(delay > cfg.maximum_event_row_span_seconds)

        index_prices = group["index_price"].to_numpy(float)
        if not np.allclose(index_prices, index_prices[0], rtol=1e-12, atol=1e-8):
            raise RuntimeError("DEWH option rows disagree on delivery index")
        index_price = float(index_prices[0])
        positions = group["position"].to_numpy(float)
        total_position = float(positions.sum())
        if not math.isfinite(total_position) or total_position <= 0.0:
            raise RuntimeError("DEWH expiry total position is invalid")

        by_strike = cast(
            pd.Series,
            group.groupby("strike", sort=True)["position"].sum(),
        )
        if len(by_strike) < 3:
            invalid_too_few_strikes += 1
            continue
        strike_positions = by_strike.to_numpy(float)
        maximum_position = float(strike_positions.max())
        tied = np.flatnonzero(strike_positions == maximum_position)
        if len(tied) != 1:
            invalid_tied_wall += 1
            continue
        wall_offset = int(tied[0])
        strikes = by_strike.index.to_numpy(float)
        wall_strike = float(strikes[wall_offset])
        other_strikes = np.delete(strikes, wall_offset)
        log_spacings = np.abs(np.log(other_strikes / wall_strike))
        local_spacing = float(log_spacings.min())
        if not math.isfinite(local_spacing) or local_spacing <= 0.0:
            invalid_spacing += 1
            continue

        wall_share = maximum_position / total_position
        hhi = float(np.square(strike_positions / total_position).sum())
        largest_instrument_share = float(positions.max() / total_position)
        signed_distance = float(
            math.log(index_price / wall_strike) / local_spacing
        )
        if not all(
            math.isfinite(value)
            for value in (
                wall_share,
                hhi,
                largest_instrument_share,
                signed_distance,
            )
        ):
            raise RuntimeError("DEWH derived wall feature is non-finite")
        if not (
            0.0 < largest_instrument_share <= wall_share <= 1.0
            and 0.0 < hhi <= 1.0
        ):
            raise RuntimeError("DEWH wall concentration escaped its bounds")

        records.append(
            {
                "expiry_time": expiry,
                "delivery_event_time": delivery_event_time,
                "source_observation_earliest": delivery_event_time
                + pd.Timedelta(minutes=65),
                "index_price": index_price,
                "distinct_strike_count": int(len(by_strike)),
                "total_position": total_position,
                "dominant_strike": wall_strike,
                "dominant_strike_position": maximum_position,
                "wall_share": wall_share,
                "strike_position_hhi": hhi,
                "largest_individual_instrument_share": (
                    largest_instrument_share
                ),
                "local_log_spacing": local_spacing,
                "signed_normalized_wall_distance": signed_distance,
                "wall_tie_count": 1,
                "delivery_delay_seconds": delay,
                "maximum_event_row_span_seconds": raw_span,
            }
        )

    if not records:
        raise RuntimeError("DEWH source has no wall-valid expiries")
    aggregate = pd.DataFrame.from_records(records, columns=SOURCE_COLUMNS)
    if aggregate["expiry_time"].duplicated().any():
        raise AssertionError("DEWH wall-valid expiry clock is not unique")
    if not aggregate["expiry_time"].is_monotonic_increasing:
        raise AssertionError("DEWH wall-valid expiry clock is not chronological")
    if total_expiries != (
        len(aggregate)
        + invalid_too_few_strikes
        + invalid_tied_wall
        + invalid_spacing
    ):
        raise AssertionError("DEWH source exclusion accounting does not reconcile")

    audit = {
        "option_rows_selected": int(len(options)),
        "futures_rows_excluded": int(futures),
        "source_expiry_events": total_expiries,
        "wall_valid_expiry_events": int(len(aggregate)),
        "invalid_too_few_strikes": invalid_too_few_strikes,
        "invalid_tied_wall": invalid_tied_wall,
        "invalid_spacing": invalid_spacing,
        "first_source_expiry": cast(
            pd.Timestamp, options["expiry_time"].iloc[0]
        ).isoformat(),
        "last_source_expiry": cast(
            pd.Timestamp, options["expiry_time"].iloc[-1]
        ).isoformat(),
        "first_wall_valid_expiry": cast(
            pd.Timestamp, aggregate["expiry_time"].iloc[0]
        ).isoformat(),
        "last_wall_valid_expiry": cast(
            pd.Timestamp, aggregate["expiry_time"].iloc[-1]
        ).isoformat(),
        "all_positions_positive": bool(options["position"].gt(0.0).all()),
        "unique_instrument_per_expiry": True,
        "all_scheduled_expiries_at_08_utc": bool(
            options["expiry_time"].dt.hour.eq(8).all()
            and options["expiry_time"].dt.minute.eq(0).all()
        ),
        "maximum_delivery_delay_seconds": maximum_delay,
        "delayed_expiry_events": delayed_expiries,
        "maximum_event_row_span_seconds": maximum_span,
        "option_rows_by_year": _year_counts(
            cast(pd.Series, options["expiry_time"])
        ),
        "source_expiries_by_year": _year_counts(
            cast(
                pd.Series,
                options.drop_duplicates("expiry_time")["expiry_time"],
            )
        ),
        "wall_valid_expiries_by_year": _year_counts(
            cast(pd.Series, aggregate["expiry_time"])
        ),
    }
    return aggregate, audit


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    if delivery.sha256_file(MECHANISM_DOCUMENT) != MECHANISM_DOCUMENT_SHA256:
        raise ValueError("DEWH mechanism document changed")
    if delivery.sha256_file(SHARED_DOWNLOADER) != SHARED_DOWNLOADER_SHA256:
        raise ValueError("DEWH shared Deribit downloader changed")
    delivery_cfg = _delivery_config(cfg)
    frame, audit = delivery.download(
        delivery_cfg,
        fetch=fetch,
        sleep=sleep if sleep is not None else time.sleep,
        aggregate=aggregate_wall_deliveries,
        page_commitment=_settlement_rows_commitment,
    )
    audit["page_commitment_scope"] = "ordered settlements rows only"
    audit["excluded_dynamic_response_fields"] = [
        "result.continuation",
        "usIn",
        "usOut",
        "usDiff",
    ]
    output = Path(cfg.output_csv)
    delivery._write_deterministic_csv(output, frame)
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "configuration": asdict(cfg),
        "mechanism_binding": {
            "path": str(MECHANISM_DOCUMENT),
            "sha256": MECHANISM_DOCUMENT_SHA256,
        },
        "shared_downloader_binding": {
            "path": str(SHARED_DOWNLOADER),
            "sha256": SHARED_DOWNLOADER_SHA256,
            "pre_aggregator_refactor_sha256": (
                FROZEN_PRE_REFACTOR_DOWNLOADER_SHA256
            ),
        },
        "source_audit": audit,
        "aggregate": {
            "path": str(output),
            "sha256": delivery.sha256_file(output),
            "bytes": output.stat().st_size,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        },
        "retained_source_fields": list(SOURCE_COLUMNS),
        "forbidden_fields_retained": [],
        "forbidden_primary_fields": [
            "mark_price",
            "profit_loss",
            "session_profit_loss",
            "option_type",
            "terminal_state",
            "net_release_position",
            "release_side",
            "itm_call_position",
            "itm_put_position",
        ],
        "causal_availability": {
            "deribit_publication_sla_known": False,
            "historical_observation_earliest": (
                "delivery_event_time + 65 minutes"
            ),
            "live_rule": (
                "two identical canonical delivery sets five minutes apart "
                "after a sixty-minute embargo; later arrival delays or cancels"
            ),
        },
        "outcome_boundary": {
            "binance_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "future_return_rows_loaded": 0,
            "performance_artifacts_parsed": 0,
            "return_or_pnl_fields_retained": 0,
            "economic_outcomes_computed": False,
            "raw_deribit_rows_persisted": False,
        },
        "candidate_incidence_computed": False,
        "parameter_search_performed": False,
    }
    manifest = {**core, "manifest_hash": delivery.canonical_hash(core)}
    _atomic_json(Path(cfg.manifest_output), manifest)
    return manifest


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", default=Config.output_csv)
    parser.add_argument("--manifest-output", default=Config.manifest_output)
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end-exclusive", default=Config.end_exclusive)
    parser.add_argument("--page-size", type=int, default=Config.page_size)
    parser.add_argument("--timeout-sec", type=float, default=Config.timeout_sec)
    parser.add_argument(
        "--request-pause-sec",
        type=float,
        default=Config.request_pause_sec,
    )
    parser.add_argument(
        "--maximum-retries",
        type=int,
        default=Config.maximum_retries,
    )
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    manifest = run(parse_args())
    print(
        json.dumps(
            {
                "aggregate": manifest["aggregate"],
                "manifest": manifest["configuration"]["manifest_output"],
                "manifest_hash": manifest["manifest_hash"],
                "source_expiries": manifest["source_audit"][
                    "source_expiry_events"
                ],
                "wall_valid_expiries": manifest["source_audit"][
                    "wall_valid_expiry_events"
                ],
                "outcomes_computed": manifest["outcome_boundary"][
                    "economic_outcomes_computed"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
