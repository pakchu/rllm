"""Build the outcome-blind BAFR-24F clock and temporal-support decision."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.aggressor_frustration import BAR_COLUMNS


BASELINE_BARS = 8_640
BASELINE_MIN_PERIODS = 2_016
SCORE_QUANTILE = 0.90
HOLD_BARS = 24
POST_GAP_QUARANTINE_BARS = 24
SELECTION_END = pd.Timestamp("2024-01-01")
CLOCK_COLUMNS = (
    "signal_position",
    "entry_position",
    "exit_position",
    "signal_date",
    "entry_date",
    "exit_date",
    "side",
    "hold_bars",
)


@dataclass(frozen=True)
class SupportConfig:
    features: str = (
        "data/binance_um_aggressor_frustration_btc_2020_2023/"
        "BTCUSDT_aggressor_frustration_5m_2020-01-01_2023-12-31.csv.gz"
    )
    feature_manifest: str = (
        "data/binance_um_aggressor_frustration_btc_2020_2023/build_manifest.json"
    )
    market: str = (
        "data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    clock_output: str = "results/binance_aggressor_frustration_clock_2026-07-20.csv"
    result_output: str = "results/binance_aggressor_frustration_support_2026-07-20.json"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_complete_grid(dates: pd.Series) -> None:
    if dates.empty:
        raise ValueError("market timestamp grid is empty")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("market timestamps must be unique and monotonic")
    deltas = dates.diff().dropna()
    if len(deltas) and not deltas.eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("market timestamps must form a complete five-minute grid")


def _allclose(left: pd.Series, right: pd.Series) -> bool:
    return bool(
        np.allclose(
            pd.to_numeric(left, errors="coerce").to_numpy(float),
            pd.to_numeric(right, errors="coerce").to_numpy(float),
            rtol=1e-9,
            atol=1e-3,
            equal_nan=False,
        )
    )


def validate_feature_identities(features: pd.DataFrame) -> dict[str, bool]:
    checks = {
        "schema": tuple(features.columns) == BAR_COLUMNS,
        "timestamps": not features["date"].duplicated().any()
        and features["date"].is_monotonic_increasing,
        "event_count_partition": bool(
            (
                features["agg_trade_count"]
                == features["classified_tick_count"] + features["unavailable_tick_count"]
            ).all()
        ),
        "aggressor_notional_partition": _allclose(
            features["quote_notional"],
            features["buy_quote_notional"] + features["sell_quote_notional"],
        ),
        "signed_notional_identity": _allclose(
            features["signed_quote_notional"],
            features["buy_quote_notional"] - features["sell_quote_notional"],
        ),
        "tick_notional_partition": _allclose(
            features["classified_quote_notional"],
            features["up_tick_notional"] + features["down_tick_notional"],
        ),
        "buy_frustration_identity": _allclose(
            features["buy_frustrated_notional"],
            features["strict_buy_frustrated_notional"]
            + features["carried_buy_frustrated_notional"],
        ),
        "sell_frustration_identity": _allclose(
            features["sell_frustrated_notional"],
            features["strict_sell_frustrated_notional"]
            + features["carried_sell_frustrated_notional"],
        ),
    }
    quote = pd.to_numeric(features["quote_notional"], errors="coerce")
    classified = pd.to_numeric(features["classified_quote_notional"], errors="coerce")
    total_frustrated = (
        pd.to_numeric(features["buy_frustrated_notional"], errors="coerce")
        + pd.to_numeric(features["sell_frustrated_notional"], errors="coerce")
    )
    expected_share = total_frustrated.divide(quote).fillna(0.0)
    expected_score = (
        pd.to_numeric(features["sell_frustrated_notional"], errors="coerce")
        - pd.to_numeric(features["buy_frustrated_notional"], errors="coerce")
    ).divide(quote).fillna(0.0)
    expected_tick = (
        pd.to_numeric(features["up_tick_notional"], errors="coerce")
        - pd.to_numeric(features["down_tick_notional"], errors="coerce")
    ).divide(classified).fillna(0.0)
    checks.update(
        {
            "frustrated_share_identity": _allclose(
                features["frustrated_notional_share"], expected_share
            ),
            "frustration_score_identity": _allclose(
                features["frustration_score"], expected_score
            ),
            "tick_imbalance_identity": _allclose(
                features["tick_notional_imbalance"], expected_tick
            ),
            "score_bounds": bool(
                pd.to_numeric(features["frustration_score"], errors="coerce")
                .abs()
                .le(1.0 + 1e-9)
                .all()
            ),
            "nonnegative_notionals": bool(
                features[
                    [
                        column
                        for column in features.columns
                        if column.endswith("notional") and not column.startswith("signed_")
                    ]
                ]
                .apply(pd.to_numeric, errors="coerce")
                .ge(0.0)
                .all()
                .all()
            ),
        }
    )
    return checks


def prior_clean_quantile(
    values: pd.Series,
    clean: pd.Series,
    *,
    quantile: float = SCORE_QUANTILE,
    window: int = BASELINE_BARS,
    min_periods: int = BASELINE_MIN_PERIODS,
) -> pd.Series:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if not 1 <= min_periods <= window:
        raise ValueError("rolling window/minimum are invalid")
    numeric = pd.to_numeric(values, errors="coerce")
    valid = clean.astype(bool) & numeric.notna() & np.isfinite(numeric)
    observed = numeric.loc[valid]
    observed_threshold = (
        observed.shift(1).rolling(window, min_periods=min_periods).quantile(quantile)
    )
    output = pd.Series(np.nan, index=values.index, dtype=float)
    output.loc[observed_threshold.index] = observed_threshold.to_numpy(float)
    return output


def quarantine_mask(
    source_available: pd.Series,
    source_gap_day: pd.Series,
    *,
    post_gap_bars: int = POST_GAP_QUARANTINE_BARS,
) -> pd.Series:
    invalid = (~source_available.astype(bool)) | source_gap_day.astype(bool)
    return (
        invalid.astype(np.int8)
        .rolling(post_gap_bars + 1, min_periods=1)
        .max()
        .astype(bool)
    )


def _source_gap_days(manifest: dict[str, Any]) -> list[str]:
    days = {
        archive["date"]
        for month in manifest.get("months", [])
        for archive in month.get("archives", [])
        if int(archive.get("state_reset_count", 0)) > 0
    }
    return sorted(days)


def validate_manifest_chain(manifest: dict[str, Any]) -> dict[str, bool]:
    months = manifest.get("months")
    checks = {
        "manifest_months_present": isinstance(months, list) and bool(months),
        "monthly_artifact_hashes": True,
        "monthly_rows": True,
        "archive_date_lists": True,
        "within_month_state_chain": True,
        "month_warmup_chain": True,
        "archive_terminal_ids": True,
    }
    if not checks["manifest_months_present"]:
        return checks

    assert isinstance(months, list)
    ordered = sorted(months, key=lambda item: str(item.get("month", "")))
    if ordered != months:
        checks["archive_date_lists"] = False
    monthly_row_total = 0
    previous_archive: dict[str, Any] | None = None
    for month in months:
        output = Path(str(month.get("output", "")))
        if not output.is_file() or _sha256(output) != month.get("output_sha256"):
            checks["monthly_artifact_hashes"] = False
        monthly_row_total += int(month.get("rows", 0))
        archives = month.get("archives")
        requested = month.get("requested_dates")
        if (
            not isinstance(archives, list)
            or not archives
            or not isinstance(requested, list)
            or requested != [archive.get("date") for archive in archives]
        ):
            checks["archive_date_lists"] = False
            continue
        warmup = month.get("warmup")
        first_state = archives[0].get("state_in")
        if not isinstance(warmup, dict):
            checks["month_warmup_chain"] = False
        elif warmup.get("status") == "verified":
            if warmup.get("state_out") != first_state:
                checks["month_warmup_chain"] = False
            if previous_archive is not None and (
                warmup.get("date") != previous_archive.get("date")
                or warmup.get("archive_sha256") != previous_archive.get("archive_sha256")
                or warmup.get("state_out") != previous_archive.get("state_out")
            ):
                checks["month_warmup_chain"] = False
        elif warmup.get("status") == "unavailable":
            if first_state != {
                "previous_price": None,
                "last_nonzero_tick": 0,
                "previous_agg_trade_id": None,
            }:
                checks["month_warmup_chain"] = False
        else:
            checks["month_warmup_chain"] = False

        for index, archive in enumerate(archives):
            if index and archive.get("state_in") != archives[index - 1].get("state_out"):
                checks["within_month_state_chain"] = False
            state_out = archive.get("state_out")
            if (
                not isinstance(state_out, dict)
                or state_out.get("previous_agg_trade_id") != archive.get("last_agg_trade_id")
            ):
                checks["archive_terminal_ids"] = False
        previous_archive = archives[-1]
    checks["monthly_rows"] = monthly_row_total == int(manifest.get("rows", -1))
    return checks


def load_support_frame(
    cfg: SupportConfig,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, bool]]:
    manifest_path = Path(cfg.feature_manifest)
    features_path = Path(cfg.features)
    market_path = Path(cfg.market)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("BAFR source manifest opened outcomes")
    feature_hash = _sha256(features_path)
    if feature_hash != manifest.get("combined_sha256"):
        raise ValueError("BAFR feature hash differs from its manifest")

    manifest_checks = validate_manifest_chain(manifest)
    if not all(manifest_checks.values()):
        failed = sorted(name for name, passed in manifest_checks.items() if not passed)
        raise ValueError(f"BAFR manifest integrity failed: {failed}")

    features = pd.read_csv(features_path, compression="gzip", parse_dates=["date"])
    checks = {**manifest_checks, **validate_feature_identities(features)}
    checks.update(
        {
            "manifest_rows": int(manifest.get("rows", -1)) == len(features),
            "manifest_columns": manifest.get("columns") == list(features.columns),
            "pre2024_cutoff": bool(features["date"].max() < SELECTION_END),
        }
    )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"BAFR source integrity failed: {failed}")

    # The market file is hashed in full but only its timestamp column is parsed.
    market_hash = _sha256(market_path)
    market = pd.read_csv(
        market_path,
        compression="gzip",
        usecols=["date"],
        parse_dates=["date"],
    )
    _assert_complete_grid(market["date"])
    if market["date"].max() >= SELECTION_END:
        raise ValueError("BAFR support market contains 2024+ timestamps")

    frame = market.merge(features, on="date", how="left", validate="one_to_one")
    required = [column for column in BAR_COLUMNS if column != "date"]
    frame["source_available"] = frame[required].notna().all(axis=1)
    gap_days = _source_gap_days(manifest)
    frame["source_gap_day"] = frame["date"].dt.strftime("%Y-%m-%d").isin(gap_days)
    frame["quarantined"] = quarantine_mask(
        frame["source_available"], frame["source_gap_day"]
    )
    metadata = {
        "feature_sha256": feature_hash,
        "feature_manifest_sha256": _sha256(manifest_path),
        "market_sha256": market_hash,
        "market_columns_loaded": ["date"],
        "price_or_outcome_columns_loaded": [],
        "feature_rows": int(len(features)),
        "market_rows": int(len(market)),
        "missing_feature_bars": int((~frame["source_available"]).sum()),
        "source_gap_days": gap_days,
        "quarantined_bars": int(frame["quarantined"].sum()),
        "quarantined_fraction": float(frame["quarantined"].mean()),
        "first_date": str(frame["date"].min()),
        "last_date": str(frame["date"].max()),
    }
    return frame, metadata, checks


def build_schedule(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    clean = ~frame["quarantined"].astype(bool)
    score = pd.to_numeric(frame["frustration_score"], errors="coerce")
    threshold = prior_clean_quantile(score.abs(), clean)
    raw_signal = clean & score.ne(0.0) & score.abs().ge(threshold)
    sides = np.sign(score).fillna(0.0).to_numpy(np.int8)
    dates = frame["date"]
    quarantined = frame["quarantined"].to_numpy(bool)
    rows: list[dict[str, Any]] = []
    previous_exit = -1
    for signal_position in np.flatnonzero(raw_signal.to_numpy(bool)):
        entry_position = int(signal_position + 1)
        exit_position = int(entry_position + HOLD_BARS)
        if exit_position >= len(frame):
            continue
        if entry_position < previous_exit:
            continue
        if quarantined[signal_position : exit_position + 1].any():
            continue
        if dates.iloc[exit_position] >= SELECTION_END:
            continue
        rows.append(
            {
                "signal_position": int(signal_position),
                "entry_position": entry_position,
                "exit_position": exit_position,
                "signal_date": str(dates.iloc[signal_position]),
                "entry_date": str(dates.iloc[entry_position]),
                "exit_date": str(dates.iloc[exit_position]),
                "side": int(sides[signal_position]),
                "hold_bars": HOLD_BARS,
            }
        )
        previous_exit = exit_position
    diagnostics = {
        "clean_rows": int(clean.sum()),
        "threshold_available_rows": int(threshold.notna().sum()),
        "raw_signal_count": int(raw_signal.sum()),
        "nonoverlap_count": int(len(rows)),
    }
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS), diagnostics


def support_metrics(clock: pd.DataFrame) -> dict[str, Any]:
    if clock.empty:
        return {
            "total": 0,
            "by_year": {str(year): 0 for year in range(2020, 2024)},
            "2023_h1": 0,
            "2023_h2": 0,
            "long_share": 0.0,
            "short_share": 0.0,
            "maximum_single_month_share": 0.0,
            "passes": False,
        }
    dates = pd.to_datetime(clock["entry_date"], errors="raise")
    total = int(len(clock))
    by_year = {
        str(year): int(dates.dt.year.eq(year).sum()) for year in range(2020, 2024)
    }
    h1 = int(((dates >= "2023-01-01") & (dates < "2023-07-01")).sum())
    h2 = int(((dates >= "2023-07-01") & (dates < "2024-01-01")).sum())
    long_share = float(clock["side"].eq(1).mean())
    short_share = float(clock["side"].eq(-1).mean())
    maximum_month_share = float(dates.dt.to_period("M").value_counts().max() / total)
    passes = (
        total >= 250
        and all(value >= 40 for value in by_year.values())
        and h1 >= 20
        and h2 >= 20
        and 0.25 <= long_share <= 0.75
        and 0.25 <= short_share <= 0.75
        and maximum_month_share <= 0.20
    )
    return {
        "total": total,
        "by_year": by_year,
        "2023_h1": h1,
        "2023_h2": h2,
        "long_share": long_share,
        "short_share": short_share,
        "maximum_single_month_share": maximum_month_share,
        "passes": bool(passes),
    }


def run_support(cfg: SupportConfig) -> dict[str, Any]:
    frame, source, source_checks = load_support_frame(cfg)
    clock, diagnostics = build_schedule(frame)
    support = support_metrics(clock)
    source_quality_passes = source["quarantined_fraction"] <= 0.02
    passed = bool(source_quality_passes and support["passes"])

    clock_path = Path(cfg.clock_output)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    clock.to_csv(clock_path, index=False)
    result = {
        "as_of": "2026-07-20",
        "candidate": "BAFR-24F",
        "stage": "outcome_blind_support",
        "outcomes_opened": False,
        "policy": {
            "baseline_clean_observations": BASELINE_BARS,
            "baseline_minimum_observations": BASELINE_MIN_PERIODS,
            "absolute_score_quantile": SCORE_QUANTILE,
            "entry": "next five-minute open",
            "hold_bars": HOLD_BARS,
            "side": "sign(frustration_score)",
            "post_gap_quarantine_bars": POST_GAP_QUARANTINE_BARS,
        },
        "config": asdict(cfg),
        "source": source,
        "source_checks": source_checks,
        "source_quality_passes": source_quality_passes,
        "diagnostics": diagnostics,
        "support": support,
        "clock": {
            "path": str(clock_path),
            "sha256": _sha256(clock_path),
            "rows": int(len(clock)),
            "columns": list(clock.columns),
        },
        "passed": passed,
        "next_stage": "outcome_blind_novelty_gate" if passed else "reject_without_outcomes",
    }
    output_path = Path(cfg.result_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=SupportConfig.features)
    parser.add_argument("--feature-manifest", default=SupportConfig.feature_manifest)
    parser.add_argument("--market", default=SupportConfig.market)
    parser.add_argument("--clock-output", default=SupportConfig.clock_output)
    parser.add_argument("--result-output", default=SupportConfig.result_output)
    result = run_support(SupportConfig(**vars(parser.parse_args())))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
