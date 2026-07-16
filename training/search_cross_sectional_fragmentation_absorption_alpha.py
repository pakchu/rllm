"""Search XFA-1: cross-sectional fragmented-flow absorption before 2026.

The strategy has no BTC leg.  At each completed UTC hour it looks for an alt
whose aggressive flow is extreme, whose factor-adjusted price response is
small, and whose average trade size is unusually low.  It trades the signal
alt against ETH in the direction opposite the aggressive flow, with causal
rolling beta-neutral weights.  All thresholds and outcomes in this module are
restricted to 2023-2025 development data; 2026 is not read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import develop_causal_residual_expert_switcher_pre2026 as strict
from training import evaluate_leave_one_out_residual_continuation_2025 as lorc
from training import select_leave_one_out_residual_exhaustion_pre2025 as lore
from training.select_leave_one_out_residual_exhaustion_pre2025 import (
    weekly_cluster_signflip,
)


SYMBOLS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
CANDIDATE_SYMBOLS = tuple(symbol for symbol in SYMBOLS if symbol != "ETHUSDT")
HEDGE_SYMBOL = "ETHUSDT"
LORE_DIR = Path("data/binance_um_lore_2023_2024")
LORC_DIR = Path("data/binance_um_lorc_2024_2025")
OUTPUT = Path("results/cross_sectional_fragmentation_absorption_pre2026_2026-07-17.json")
DOCS_OUTPUT = Path("docs/cross-sectional-fragmentation-absorption-pre2026-2026-07-17.md")
BASE_COST_BP = 6.0
STRESS_COST_BP = 10.0
ROLLING_HOURS = 30 * 24
MINIMUM_ROLLING_HOURS = 14 * 24
RESIDUAL_VOL_HOURS = 7 * 24
MINIMUM_RESIDUAL_VOL_HOURS = 3 * 24
SIGNFLIP_SEED = 2_026_071_8
SIGNFLIP_SAMPLES = 20_000


@dataclass(frozen=True)
class Policy:
    policy_id: str
    minimum_abs_flow_z: float
    maximum_abs_residual_z: float
    maximum_average_trade_size_z: float
    hold_hours: int


POLICIES = (
    Policy("XFA01", 2.00, 0.50, -0.50, 3),
    Policy("XFA02", 2.00, 0.50, -0.50, 6),
    Policy("XFA03", 2.50, 0.75, -0.50, 3),
    Policy("XFA04", 2.50, 0.75, -0.50, 6),
    Policy("XFA05", 2.00, 0.35, -0.25, 3),
    Policy("XFA06", 2.00, 0.35, -0.25, 6),
    Policy("XFA07", 1.75, 0.50, -0.75, 3),
    Policy("XFA08", 1.75, 0.50, -0.75, 6),
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prior_zscore(series: pd.Series, window: int, minimum: int) -> pd.Series:
    mean = series.rolling(window, min_periods=minimum).mean().shift(1)
    std = series.rolling(window, min_periods=minimum).std(ddof=1).shift(1)
    return ((series - mean) / std.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def _load_hourly_symbol(
    directory: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    suffix = "2023_2024" if directory == LORE_DIR else "2024_2025"
    path = directory / f"{symbol}_5m_{suffix}.csv.gz"
    columns = (
        "date",
        "open",
        "high",
        "low",
        "close",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_quote",
        "tic",
    )
    raw = pd.read_csv(path, usecols=list(columns), parse_dates=["date"])
    raw = raw.loc[(raw["date"] >= start) & (raw["date"] < end)].copy()
    expected = pd.date_range(start, end - pd.Timedelta(minutes=5), freq="5min")
    if not pd.DatetimeIndex(raw["date"]).equals(expected):
        raise RuntimeError(f"{symbol} XFA physical 5m grid failure")
    if not raw["tic"].astype(str).eq(symbol).all():
        raise RuntimeError(f"{symbol} XFA identity failure")
    raw = raw.set_index("date")
    hourly = pd.DataFrame(
        {
            "open": raw["open"].resample("1h", closed="left", label="right").first(),
            "high": raw["high"].resample("1h", closed="left", label="right").max(),
            "low": raw["low"].resample("1h", closed="left", label="right").min(),
            "close": raw["close"].resample("1h", closed="left", label="right").last(),
            "quote_volume": raw["quote_asset_volume"]
            .resample("1h", closed="left", label="right")
            .sum(),
            "trade_count": raw["number_of_trades"]
            .resample("1h", closed="left", label="right")
            .sum(),
            "taker_buy_quote": raw["taker_buy_quote"]
            .resample("1h", closed="left", label="right")
            .sum(),
            "bar_count": raw["close"].resample("1h", closed="left", label="right").count(),
            "positive_activity_count": (
                raw["quote_asset_volume"].gt(0) & raw["number_of_trades"].gt(0)
            )
            .resample("1h", closed="left", label="right")
            .sum(),
        }
    )
    expected_hourly = pd.date_range(start + pd.Timedelta(hours=1), end, freq="1h")
    if not hourly.index.equals(expected_hourly) or not hourly["bar_count"].eq(12).all():
        raise RuntimeError(f"{symbol} XFA completed-hour grid failure")
    hourly["quality"] = hourly["positive_activity_count"].eq(12)
    numeric = hourly.drop(columns=["bar_count", "quality"]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{symbol} XFA non-finite source")
    if (hourly[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{symbol} XFA non-positive price source")
    return hourly


def build_features(
    directory: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    hourly = {
        symbol: _load_hourly_symbol(directory, symbol, start, end)
        for symbol in SYMBOLS
    }
    returns = pd.DataFrame(
        {
            symbol: np.log(frame["close"] / frame["open"]).where(frame["quality"])
            for symbol, frame in hourly.items()
        }
    )
    factors = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    betas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for symbol in SYMBOLS:
        factor = returns.drop(columns=symbol).median(axis=1)
        factors[symbol] = factor
        covariance = returns[symbol].rolling(
            ROLLING_HOURS, min_periods=MINIMUM_ROLLING_HOURS
        ).cov(factor)
        variance = factor.rolling(
            ROLLING_HOURS, min_periods=MINIMUM_ROLLING_HOURS
        ).var()
        betas[symbol] = (covariance / variance.replace(0.0, np.nan)).shift(1).clip(
            0.25, 2.5
        )
    rows: list[pd.DataFrame] = []
    for symbol in SYMBOLS:
        frame = hourly[symbol]
        residual = returns[symbol] - betas[symbol] * factors[symbol]
        residual_vol = residual.rolling(
            RESIDUAL_VOL_HOURS, min_periods=MINIMUM_RESIDUAL_VOL_HOURS
        ).std(ddof=1).shift(1)
        residual_z = residual / residual_vol.replace(0.0, np.nan)
        flow = (2.0 * frame["taker_buy_quote"] - frame["quote_volume"]) / frame[
            "quote_volume"
        ]
        flow = flow.where(frame["quality"])
        flow_z = _prior_zscore(flow, ROLLING_HOURS, MINIMUM_ROLLING_HOURS)
        average_trade_size = (frame["quote_volume"] / frame["trade_count"]).where(
            frame["quality"]
        )
        average_trade_size_z = _prior_zscore(
            np.log(average_trade_size), ROLLING_HOURS, MINIMUM_ROLLING_HOURS
        )
        log_range = np.log(frame["high"] / frame["low"]).where(frame["quality"])
        range_rms = log_range.rolling(
            3 * 24, min_periods=24
        ).apply(lambda values: float(np.sqrt(np.mean(values**2))), raw=True).shift(1)
        rows.append(
            pd.DataFrame(
                {
                    "signal_time": frame.index,
                    "symbol": symbol,
                    "beta": betas[symbol],
                    "factor_return": factors[symbol],
                    "residual_return": residual,
                    "residual_z": residual_z,
                    "flow": flow,
                    "flow_z": flow_z,
                    "average_trade_size": average_trade_size,
                    "average_trade_size_z": average_trade_size_z,
                    "range_risk": range_rms,
                }
            )
        )
    features = pd.concat(rows, ignore_index=True)
    features["feature_available_time"] = features["signal_time"]
    return features.sort_values(["signal_time", "symbol"]).reset_index(drop=True)


def _event_score(row: pd.Series, *, require_fragmentation: bool) -> float:
    fragmentation = (
        max(-float(row["average_trade_size_z"]), 0.0)
        if require_fragmentation
        else 1.0
    )
    return float(
        abs(row["flow_z"])
        / (0.25 + abs(row["residual_z"]))
        * fragmentation
    )


def build_clock(
    features: pd.DataFrame,
    policy: Policy,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    require_fragmentation: bool = True,
) -> pd.DataFrame:
    by_time = {time: frame for time, frame in features.groupby("signal_time", sort=True)}
    previous_exit: pd.Timestamp | None = None
    rows: list[dict[str, Any]] = []
    for signal_time, frame in by_time.items():
        signal = pd.Timestamp(signal_time)
        if signal < start or signal >= end:
            continue
        if previous_exit is not None and signal + pd.Timedelta(minutes=5) < previous_exit:
            continue
        candidates = frame.loc[frame["symbol"].isin(CANDIDATE_SYMBOLS)].copy()
        finite_columns = ["beta", "residual_z", "flow_z", "average_trade_size_z", "range_risk"]
        candidates = candidates.loc[np.isfinite(candidates[finite_columns]).all(axis=1)]
        candidates = candidates.loc[
            candidates["flow_z"].abs().ge(policy.minimum_abs_flow_z)
            & candidates["residual_z"].abs().le(policy.maximum_abs_residual_z)
        ]
        if require_fragmentation:
            candidates = candidates.loc[
                candidates["average_trade_size_z"].le(policy.maximum_average_trade_size_z)
            ]
        if candidates.empty:
            continue
        candidates["event_score"] = candidates.apply(
            lambda row: _event_score(
                row, require_fragmentation=require_fragmentation
            ),
            axis=1,
        )
        event = candidates.sort_values(
            ["event_score", "symbol"], ascending=[False, True]
        ).iloc[0]
        hedge = frame.loc[frame["symbol"].eq(HEDGE_SYMBOL)]
        if len(hedge) != 1 or not np.isfinite(hedge[["beta", "range_risk"]]).all(axis=None):
            continue
        hedge_row = hedge.iloc[0]
        signal_beta = float(event["beta"])
        hedge_beta = float(hedge_row["beta"])
        denominator = signal_beta + hedge_beta
        signal_weight = hedge_beta / denominator
        hedge_weight = signal_beta / denominator
        signal_symbol = str(event["symbol"])
        if float(event["flow_z"]) > 0.0:
            long_symbol, short_symbol = HEDGE_SYMBOL, signal_symbol
            long_weight, short_weight = hedge_weight, signal_weight
            long_beta, short_beta = hedge_beta, signal_beta
            side = "absorbed_buy_short_signal"
        else:
            long_symbol, short_symbol = signal_symbol, HEDGE_SYMBOL
            long_weight, short_weight = signal_weight, hedge_weight
            long_beta, short_beta = signal_beta, hedge_beta
            side = "absorbed_sell_long_signal"
        entry = signal + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=policy.hold_hours)
        if exit_time >= end:
            continue
        rows.append(
            {
                "policy_id": policy.policy_id,
                "signal_time": signal,
                "feature_available_time": signal,
                "entry_time": entry,
                "exit_time": exit_time,
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "long_weight": long_weight,
                "short_weight_abs": short_weight,
                "long_beta": long_beta,
                "short_beta": short_beta,
                "choice": side,
                "gross_scale": 1.0,
                "predicted_edge": float(event["event_score"]),
                "confidence_threshold": 0.0,
                "signal_symbol": signal_symbol,
                "flow_z": float(event["flow_z"]),
                "residual_z": float(event["residual_z"]),
                "average_trade_size_z": float(event["average_trade_size_z"]),
                "range_risk": max(float(event["range_risk"]), float(hedge_row["range_risk"])),
            }
        )
        previous_exit = exit_time
    clock = pd.DataFrame(rows)
    if clock.empty:
        return clock
    exposure = clock["long_weight"] * clock["long_beta"] - clock[
        "short_weight_abs"
    ] * clock["short_beta"]
    if not np.allclose(exposure, 0.0, atol=1e-12):
        raise RuntimeError("XFA clock lost beta neutrality")
    if not np.allclose(clock["long_weight"] + clock["short_weight_abs"], 1.0):
        raise RuntimeError("XFA clock lost gross-one sizing")
    if not (clock["feature_available_time"] < clock["entry_time"]).all():
        raise RuntimeError("XFA feature availability crossed entry")
    if (clock["entry_time"].iloc[1:].reset_index(drop=True) < clock["exit_time"].iloc[:-1].reset_index(drop=True)).any():
        raise RuntimeError("XFA clock overlaps")
    return clock.reset_index(drop=True)


def _segments_for_policy(
    policy: Policy,
    feature_2023_2024: pd.DataFrame,
    feature_2025: pd.DataFrame,
    bundle_2023_2024: Any,
    bundle_2025: Any,
    *,
    require_fragmentation: bool = True,
) -> list[strict.Segment]:
    clock_2023_2024 = build_clock(
        feature_2023_2024,
        policy,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2025-01-01"),
        require_fragmentation=require_fragmentation,
    )
    clock_2025 = build_clock(
        feature_2025,
        policy,
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2026-01-01"),
        require_fragmentation=require_fragmentation,
    )
    return [
        strict.Segment(
            "2023_2024",
            bundle_2023_2024,
            clock_2023_2024,
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2025-01-01"),
        ),
        strict.Segment(
            "2025",
            bundle_2025,
            clock_2025,
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2026-01-01"),
        ),
    ]


def _slim_evaluation(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "annual": {
            key: strict._slim(value) for key, value in raw["annual"].items()
        },
        "combined_2024_2025": strict._slim(raw["combined_2024_2025"]),
    }


def _evaluate(segments: list[strict.Segment], cost_bp: float) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = strict._evaluate_clock(segments, cost_bp)
    return _slim_evaluation(raw), raw


def _policy_checks(
    primary: dict[str, Any],
    stress: dict[str, Any],
    delayed: dict[str, Any],
    opposite: dict[str, Any],
    no_fragmentation: dict[str, Any],
    signflip: dict[str, Any],
) -> dict[str, bool]:
    annual = primary["annual"]
    combined = primary["combined_2024_2025"]
    return {
        "each_year_absolute_return_positive": all(
            annual[year]["absolute_return_pct"] > 0.0 for year in ("2023", "2024", "2025")
        ),
        "minimum_annual_ratio_at_least_1": min(
            annual[year]["cagr_to_strict_mdd"] for year in ("2023", "2024", "2025")
        ) >= 1.0,
        "combined_2024_2025_ratio_at_least_3": combined["cagr_to_strict_mdd"] >= 3.0,
        "combined_strict_mdd_at_most_15": combined["strict_mdd_pct"] <= 15.0,
        "combined_trades_at_least_40": combined["trades"] >= 40,
        "ten_bp_stress_positive": stress["combined_2024_2025"]["absolute_return_pct"] > 0.0,
        "delay_five_minutes_positive": delayed["combined_2024_2025"]["absolute_return_pct"] > 0.0,
        "direction_flip_cagr_lower": opposite["combined_2024_2025"]["cagr_pct"] < combined["cagr_pct"],
        "fragmentation_beats_unfiltered_ratio": combined["cagr_to_strict_mdd"] > no_fragmentation[
            "combined_2024_2025"
        ]["cagr_to_strict_mdd"],
        "weekly_cluster_signflip_p_at_most_0_10": signflip["raw_p_value"] <= 0.10,
    }


def _ranking_key(result: dict[str, Any]) -> tuple[Any, ...]:
    annual = result["primary"]["annual"]
    combined = result["primary"]["combined_2024_2025"]
    return (
        result["passes"],
        min(annual[year]["cagr_to_strict_mdd"] for year in ("2023", "2024", "2025")),
        combined["cagr_to_strict_mdd"],
        combined["absolute_return_pct"],
        combined["trades"],
    )


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# XFA-1 cross-sectional fragmentation absorption — pre-2026",
        "",
        "## Decision",
        "",
        f"- Development status: **{result['status']}**.",
        "- 2026 post-entry outcomes were not read.",
        "- Eight disclosed mechanism variants were evaluated on research-seen 2023-2025 only.",
        "- XFA trades one idiosyncratic alt against ETH with causal factor-beta-neutral weights; it has no BTC leg.",
        "",
        "## Ranked development policies",
        "",
        "| Rank | Policy | Parameters | 2023 | 2024 | 2025 | 2024-25 combined | Pass |",
        "|---:|---|---|---:|---:|---:|---:|:---:|",
    ]
    for rank, row in enumerate(result["ranked"], start=1):
        primary = row["primary"]
        def cell(stats: dict[str, Any]) -> str:
            return (
                f"{stats['absolute_return_pct']:.2f}/{stats['cagr_pct']:.2f}/"
                f"{stats['strict_mdd_pct']:.2f}/{stats['cagr_to_strict_mdd']:.2f}/"
                f"{stats['trades']}"
            )
        lines.append(
            f"| {rank} | {row['policy']['policy_id']} | flow>={row['policy']['minimum_abs_flow_z']}, "
            f"resid<={row['policy']['maximum_abs_residual_z']}, size<={row['policy']['maximum_average_trade_size_z']}, "
            f"hold={row['policy']['hold_hours']}h | {cell(primary['annual']['2023'])} | "
            f"{cell(primary['annual']['2024'])} | {cell(primary['annual']['2025'])} | "
            f"{cell(primary['combined_2024_2025'])} | {'PASS' if row['passes'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Metric cells are absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades.",
            "",
            "## Causal and accounting contract",
            "",
            "- Features use exactly twelve completed 5-minute bars per UTC hour; entry is the following +5-minute open.",
            "- Flow, average-trade-size and beta standardizers use strictly prior rolling history.",
            "- A signal requires extreme taker flow, muted factor-adjusted price response and unusually small average trade size.",
            "- Direction is opposite the aggressive flow; ETH is the factor hedge, sized to zero estimated factor beta.",
            "- Strict MDD includes global/pre-entry HWM, funding, favorable-before-adverse held OHLC and hypothetical liquidation cost.",
            "- Controls include 10 bp/side, +5m entry/exit, exact direction flip and the same rule without fragmentation.",
            "",
            "A development pass would only authorize a separately frozen 2026 one-shot replay. It would not authorize live trading.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: str | Path = OUTPUT, docs_output: str | Path = DOCS_OUTPUT) -> dict[str, Any]:
    feature_2023_2024 = build_features(
        LORE_DIR, pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")
    )
    feature_2025 = build_features(
        LORC_DIR, pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")
    )
    bundle_2023_2024 = lore.load_bundle()
    bundle_2025 = lorc.load_bundle()
    evaluated: list[dict[str, Any]] = []
    for policy in POLICIES:
        segments = _segments_for_policy(
            policy,
            feature_2023_2024,
            feature_2025,
            bundle_2023_2024,
            bundle_2025,
        )
        primary, primary_raw = _evaluate(segments, BASE_COST_BP)
        stress, _ = _evaluate(segments, STRESS_COST_BP)
        delayed_segments = [
            strict.Segment(
                segment.name,
                segment.bundle,
                strict._delay_clock(segment.clock, 5),
                segment.start,
                segment.end,
            )
            for segment in segments
        ]
        opposite_segments = [
            strict.Segment(
                segment.name,
                segment.bundle,
                strict._direction_flip(segment.clock),
                segment.start,
                segment.end,
            )
            for segment in segments
        ]
        delayed, _ = _evaluate(delayed_segments, BASE_COST_BP)
        opposite, _ = _evaluate(opposite_segments, BASE_COST_BP)
        no_fragmentation_segments = _segments_for_policy(
            policy,
            feature_2023_2024,
            feature_2025,
            bundle_2023_2024,
            bundle_2025,
            require_fragmentation=False,
        )
        no_fragmentation, _ = _evaluate(no_fragmentation_segments, BASE_COST_BP)
        signflip = weekly_cluster_signflip(
            primary_raw["combined_2024_2025"]["trade_rows"],
            seed=SIGNFLIP_SEED + int(policy.policy_id[-2:]),
            samples=SIGNFLIP_SAMPLES,
        )
        checks = _policy_checks(
            primary, stress, delayed, opposite, no_fragmentation, signflip
        )
        evaluated.append(
            {
                "policy": asdict(policy),
                "primary": primary,
                "stress_10bp": stress,
                "delay_five_minutes": delayed,
                "direction_flip": opposite,
                "no_fragmentation_control": no_fragmentation,
                "weekly_cluster_signflip": signflip,
                "checks": checks,
                "passes": all(checks.values()),
            }
        )
    ranked = sorted(evaluated, key=_ranking_key, reverse=True)
    passing = [row for row in ranked if row["passes"]]
    core: dict[str, Any] = {
        "protocol_version": "xfa_v1_pre2026_development_2026-07-17",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_boundary": {
            "development_windows": ["2023", "2024", "2025"],
            "development_outcomes_research_seen": True,
            "post_entry_2026_outcomes_read": False,
            "policies_tested": len(POLICIES),
            "live_promotion_allowed": False,
        },
        "source_hashes": {
            "lore_eth_market": _sha256(LORE_DIR / "ETHUSDT_5m_2023_2024.csv.gz"),
            "lorc_eth_market": _sha256(LORC_DIR / "ETHUSDT_5m_2024_2025.csv.gz"),
        },
        "feature_contract": {
            "universe": list(SYMBOLS),
            "candidate_symbols": list(CANDIDATE_SYMBOLS),
            "hedge_symbol": HEDGE_SYMBOL,
            "rolling_hours": ROLLING_HOURS,
            "minimum_rolling_hours": MINIMUM_ROLLING_HOURS,
            "residual_vol_hours": RESIDUAL_VOL_HOURS,
            "entry_delay_minutes": 5,
            "base_cost_bp_per_side": BASE_COST_BP,
            "stress_cost_bp_per_side": STRESS_COST_BP,
        },
        "status": (
            "development_pass_freeze_before_2026" if passing else "reject_before_2026"
        ),
        "selected_policy": passing[0]["policy"] if passing else None,
        "ranked": ranked,
    }
    core["manifest_hash"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(core, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    docs_path = Path(docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_markdown(core))
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--docs-output", default=str(DOCS_OUTPUT))
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.docs_output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
