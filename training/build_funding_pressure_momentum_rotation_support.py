"""Build FPMR-1's outcome-blind 2023-2024 support clock."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.build_dispersion_conditioned_residual_momentum_support import (
    _lexical_extreme,
    feature_panels,
)
from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    sha256_file,
)
from training.preregister_funding_pressure_momentum_rotation import (
    canonical_hash,
    protocol,
)


SYMBOLS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")
PREREGISTRATION = Path(
    "results/funding_pressure_momentum_rotation_preregistration_2026-07-18.json"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "7a7d4d174862d02389fb9b3d95f4e31bcc6d722c89ae7565d42b9d606f8f6f10"
)
EXPECTED_PROTOCOL_HASH = "41c33c9a4028e78ae9007e419dcc40c64a66e5309cf1b8ea2623f83262b20cc1"
DEFAULT_MARKET_DIR = Path("data/binance_um_pool_5m_2023_2026")
DEFAULT_AUX_DIR = Path("data/binance_um_aux_2023_2026")
DEFAULT_CLOCK = Path("data/funding_pressure_momentum_rotation_clock_2023_2024.csv.gz")
DEFAULT_OUTPUT = Path(
    "results/funding_pressure_momentum_rotation_support_2026-07-18.json"
)
DEFAULT_DOCS = Path("docs/funding-pressure-momentum-rotation-support-2026-07-18.md")

CLOCK_COLUMNS = (
    "policy_id",
    "decision_time",
    "last_price_time",
    "funding_cutoff_time",
    "entry_time",
    "exit_time",
    "long_symbol",
    "short_symbol",
    "long_weight",
    "short_weight_abs",
    "long_beta",
    "short_beta",
    "long_score",
    "short_score",
    "long_residual_level_z",
    "short_residual_level_z",
    "long_residual_rotation",
    "short_residual_rotation",
    "long_funding_pressure_change_z",
    "short_funding_pressure_change_z",
    "price_only_long_symbol",
    "price_only_short_symbol",
    "price_only_long_weight",
    "price_only_short_weight_abs",
    "funding_only_long_symbol",
    "funding_only_short_symbol",
    "funding_only_long_weight",
    "funding_only_short_weight_abs",
    "static_residual_long_symbol",
    "static_residual_short_symbol",
    "static_residual_long_weight",
    "static_residual_short_weight_abs",
)
FORBIDDEN_OUTCOME_TOKENS = (
    "pnl",
    "equity",
    "trade_return",
    "absolute_return",
    "cagr",
    "drawdown",
    "entry_price",
    "exit_price",
)


def _verify_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError("FPMR-1 preregistration file changed")
    payload = json.loads(PREREGISTRATION.read_text())
    if payload.get("protocol_hash") != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("FPMR-1 preregistration identity changed")
    if canonical_hash(payload["protocol"]) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("FPMR-1 preregistration body changed")
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("FPMR-1 implementation protocol drifted")
    boundary = payload["protocol"]["evidence_boundary"]
    if boundary["exact_fpmr_score_or_post_entry_return_opened"]:
        raise RuntimeError("FPMR-1 outcomes opened before support")
    return payload


def load_predictors(
    market_dir: str | Path = DEFAULT_MARKET_DIR,
    aux_dir: str | Path = DEFAULT_AUX_DIR,
    symbols: Iterable[str] = SYMBOLS,
) -> tuple[pd.DataFrame, dict[str, pd.Series], list[dict[str, Any]]]:
    """Load only predictor closes and settled funding from the frozen prefix."""

    close_panel: dict[str, pd.Series] = {}
    funding_panel: dict[str, pd.Series] = {}
    records: list[dict[str, Any]] = []
    market_root = Path(market_dir)
    aux_root = Path(aux_dir)
    for symbol in symbols:
        market_path = market_root / f"{symbol}_5m_2023-01_2026-05.csv.gz"
        funding_path = aux_root / f"{symbol}_funding_2023-01-01_2026-06-01.csv.gz"
        market = pd.read_csv(market_path, usecols=["date", "close"])
        market["date"] = pd.to_datetime(market["date"], errors="raise")
        market["close"] = pd.to_numeric(market["close"], errors="raise")
        market = market.loc[market["date"].ge(START) & market["date"].lt(END)]
        if (
            market.empty
            or market["date"].duplicated().any()
            or not market["date"].is_monotonic_increasing
            or not np.isfinite(market["close"]).all()
            or market["close"].le(0.0).any()
        ):
            raise ValueError(f"FPMR-1 invalid market predictor prefix: {symbol}")
        funding = pd.read_csv(funding_path, usecols=["date", "funding_rate"])
        funding["date"] = pd.to_datetime(funding["date"], errors="raise")
        funding["funding_rate"] = pd.to_numeric(
            funding["funding_rate"], errors="raise"
        )
        funding = funding.loc[funding["date"].ge(START) & funding["date"].lt(END)]
        if (
            funding.empty
            or funding["date"].duplicated().any()
            or not funding["date"].is_monotonic_increasing
            or not np.isfinite(funding["funding_rate"]).all()
        ):
            raise ValueError(f"FPMR-1 invalid funding predictor prefix: {symbol}")
        close_panel[symbol] = market.set_index("date")["close"]
        funding_panel[symbol] = funding.set_index("date")["funding_rate"]
        records.extend(
            [
                {
                    "kind": "market_close_predictor",
                    "symbol": symbol,
                    "path": str(market_path),
                    "sha256": sha256_file(market_path),
                    "rows_read": int(len(market)),
                    "columns_read": ["date", "close"],
                    "first_timestamp": market["date"].iloc[0].isoformat(),
                    "last_timestamp": market["date"].iloc[-1].isoformat(),
                    "rows_at_or_after_2025_read": 0,
                },
                {
                    "kind": "settled_funding_predictor",
                    "symbol": symbol,
                    "path": str(funding_path),
                    "sha256": sha256_file(funding_path),
                    "rows_read": int(len(funding)),
                    "columns_read": ["date", "funding_rate"],
                    "first_timestamp": funding["date"].iloc[0].isoformat(),
                    "last_timestamp": funding["date"].iloc[-1].isoformat(),
                    "rows_at_or_after_2025_read": 0,
                },
            ]
        )
    close = pd.DataFrame(close_panel).sort_index()
    if tuple(close.columns) != tuple(symbols):
        raise RuntimeError("FPMR-1 symbol order changed")
    return close, funding_panel, records


def _cross_sectional_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    scale = float(numeric.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("FPMR-1 cross-sectional scale collapsed")
    return (numeric - float(numeric.mean())) / scale


def _pair(values: pd.Series, beta: pd.Series) -> dict[str, Any]:
    long_symbol = _lexical_extreme(values, maximum=True)
    short_symbol = _lexical_extreme(values, maximum=False)
    long_beta = float(beta[long_symbol])
    short_beta = float(beta[short_symbol])
    denominator = long_beta + short_beta
    return {
        "long_symbol": long_symbol,
        "short_symbol": short_symbol,
        "long_weight": short_beta / denominator,
        "short_weight_abs": long_beta / denominator,
    }


def build_clock(close: pd.DataFrame, funding: dict[str, pd.Series]) -> pd.DataFrame:
    """Create the exact predictor-only FPMR-1 schedule and frozen controls."""

    _, _, factor_30d, beta = feature_panels(close)
    rows: list[dict[str, Any]] = []
    for boundary in pd.date_range("2023-01-02", "2024-12-30", freq="W-MON"):
        last_price = boundary - pd.Timedelta(minutes=5)
        momentum_start = last_price - pd.Timedelta(days=30)
        previous_boundary = boundary - pd.Timedelta(days=7)
        previous_last = previous_boundary - pd.Timedelta(minutes=5)
        previous_start = previous_last - pd.Timedelta(days=30)
        required = (last_price, momentum_start, previous_last, previous_start)
        if any(timestamp not in close.index for timestamp in required):
            continue
        if any(timestamp not in factor_30d.index for timestamp in (last_price, previous_last)):
            continue
        current_beta = beta.loc[last_price]
        previous_beta = beta.loc[previous_last]
        current_factor = factor_30d.loc[last_price]
        previous_factor = factor_30d.loc[previous_last]
        current_momentum = np.log(close.loc[last_price] / close.loc[momentum_start])
        previous_momentum = np.log(close.loc[previous_last] / close.loc[previous_start])
        vectors = (
            current_beta,
            previous_beta,
            current_factor,
            previous_factor,
            current_momentum,
            previous_momentum,
        )
        if any(vector.isna().any() for vector in vectors):
            continue
        current_residual = current_momentum - current_beta * current_factor
        previous_residual = previous_momentum - previous_beta * previous_factor
        residual_level_z = _cross_sectional_z(current_residual)
        previous_residual_z = _cross_sectional_z(previous_residual)
        residual_rotation = residual_level_z - previous_residual_z
        current_funding = pd.Series(
            {
                symbol: series.loc[
                    (series.index > boundary - pd.Timedelta(days=28))
                    & (series.index <= boundary)
                ].sum()
                for symbol, series in funding.items()
            }
        )
        prior_funding = pd.Series(
            {
                symbol: series.loc[
                    (series.index > boundary - pd.Timedelta(days=35))
                    & (series.index <= boundary - pd.Timedelta(days=7))
                ].sum()
                for symbol, series in funding.items()
            }
        )
        funding_pressure_change_z = _cross_sectional_z(
            current_funding - prior_funding
        )
        price_only = residual_level_z + residual_rotation
        funding_only = -funding_pressure_change_z
        primary_score = price_only + funding_only
        primary_pair = _pair(primary_score, current_beta)
        price_pair = _pair(price_only, current_beta)
        funding_pair = _pair(funding_only, current_beta)
        static_pair = _pair(residual_level_z, current_beta)
        long_symbol = str(primary_pair["long_symbol"])
        short_symbol = str(primary_pair["short_symbol"])
        rows.append(
            {
                "policy_id": "FPMR01",
                "decision_time": boundary + pd.Timedelta(minutes=5),
                "last_price_time": last_price,
                "funding_cutoff_time": boundary,
                "entry_time": boundary + pd.Timedelta(minutes=10),
                "exit_time": boundary + pd.Timedelta(days=7, minutes=10),
                **primary_pair,
                "long_beta": float(current_beta[long_symbol]),
                "short_beta": float(current_beta[short_symbol]),
                "long_score": float(primary_score[long_symbol]),
                "short_score": float(primary_score[short_symbol]),
                "long_residual_level_z": float(residual_level_z[long_symbol]),
                "short_residual_level_z": float(residual_level_z[short_symbol]),
                "long_residual_rotation": float(residual_rotation[long_symbol]),
                "short_residual_rotation": float(residual_rotation[short_symbol]),
                "long_funding_pressure_change_z": float(
                    funding_pressure_change_z[long_symbol]
                ),
                "short_funding_pressure_change_z": float(
                    funding_pressure_change_z[short_symbol]
                ),
                "price_only_long_symbol": price_pair["long_symbol"],
                "price_only_short_symbol": price_pair["short_symbol"],
                "price_only_long_weight": price_pair["long_weight"],
                "price_only_short_weight_abs": price_pair["short_weight_abs"],
                "funding_only_long_symbol": funding_pair["long_symbol"],
                "funding_only_short_symbol": funding_pair["short_symbol"],
                "funding_only_long_weight": funding_pair["long_weight"],
                "funding_only_short_weight_abs": funding_pair["short_weight_abs"],
                "static_residual_long_symbol": static_pair["long_symbol"],
                "static_residual_short_symbol": static_pair["short_symbol"],
                "static_residual_long_weight": static_pair["long_weight"],
                "static_residual_short_weight_abs": static_pair["short_weight_abs"],
            }
        )
    clock = pd.DataFrame(rows, columns=CLOCK_COLUMNS)
    assert_clock_contract(clock)
    return clock


def assert_clock_contract(clock: pd.DataFrame) -> None:
    if tuple(clock.columns) != CLOCK_COLUMNS:
        raise RuntimeError("FPMR-1 clock schema changed")
    forbidden = [
        column
        for column in clock.columns
        if any(token in column.lower() for token in FORBIDDEN_OUTCOME_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"FPMR-1 outcome columns escaped support: {forbidden}")
    if clock.empty:
        return
    times = clock.copy()
    for column in (
        "decision_time",
        "last_price_time",
        "funding_cutoff_time",
        "entry_time",
        "exit_time",
    ):
        times[column] = pd.to_datetime(times[column], errors="raise")
    if not times["decision_time"].eq(
        times["funding_cutoff_time"] + pd.Timedelta(minutes=5)
    ).all():
        raise RuntimeError("FPMR-1 decision delay changed")
    if not times["entry_time"].eq(
        times["funding_cutoff_time"] + pd.Timedelta(minutes=10)
    ).all():
        raise RuntimeError("FPMR-1 entry delay changed")
    if not times["exit_time"].eq(
        times["entry_time"] + pd.Timedelta(days=7)
    ).all():
        raise RuntimeError("FPMR-1 hold changed")
    if not times["last_price_time"].lt(times["funding_cutoff_time"]).all():
        raise RuntimeError("FPMR-1 price cutoff crossed funding boundary")
    if not times["funding_cutoff_time"].lt(times["decision_time"]).all():
        raise RuntimeError("FPMR-1 funding settlement was not observed before decision")
    gross = clock["long_weight"] + clock["short_weight_abs"]
    if not np.allclose(gross, 1.0, atol=1e-12):
        raise RuntimeError("FPMR-1 gross changed")
    beta_exposure = (
        clock["long_weight"] * clock["long_beta"]
        - clock["short_weight_abs"] * clock["short_beta"]
    )
    if not np.allclose(beta_exposure, 0.0, atol=1e-12):
        raise RuntimeError("FPMR-1 beta neutrality changed")


def support_summary(clock: pd.DataFrame) -> dict[str, Any]:
    times = pd.to_datetime(clock["decision_time"], errors="raise")
    years = times.dt.year.astype(str)
    halves = years + "H" + np.where(times.dt.month.le(6), "1", "2")
    pairs = clock.groupby(["long_symbol", "short_symbol"], sort=True).size()
    long_share = clock["long_symbol"].value_counts(normalize=True)
    short_share = clock["short_symbol"].value_counts(normalize=True)
    support = {
        "events": int(len(clock)),
        "events_by_year": {str(key): int(value) for key, value in years.value_counts().sort_index().items()},
        "events_by_half": {str(key): int(value) for key, value in halves.value_counts().sort_index().items()},
        "unique_ordered_pairs": int(len(pairs)),
        "maximum_ordered_pair_share": float(pairs.max() / len(clock)),
        "maximum_long_symbol_share": float(long_share.max()),
        "maximum_short_symbol_share": float(short_share.max()),
        "long_symbols": sorted(clock["long_symbol"].unique().tolist()),
        "short_symbols": sorted(clock["short_symbol"].unique().tolist()),
    }
    gates = {
        "events_2023_2024_at_least_90": support["events"] >= 90,
        "events_each_year_at_least_45": min(support["events_by_year"].values()) >= 45,
        "events_each_half_at_least_20": min(support["events_by_half"].values()) >= 20,
        "unique_ordered_pairs_at_least_15": support["unique_ordered_pairs"] >= 15,
        "maximum_ordered_pair_share_at_most_0_15": support["maximum_ordered_pair_share"] <= 0.15,
        "maximum_symbol_side_share_at_most_0_40": max(
            support["maximum_long_symbol_share"], support["maximum_short_symbol_share"]
        )
        <= 0.40,
        "all_six_symbols_long_and_short": (
            support["long_symbols"] == sorted(SYMBOLS)
            and support["short_symbols"] == sorted(SYMBOLS)
        ),
        "outcome_columns_forbidden": not any(
            any(token in column.lower() for token in FORBIDDEN_OUTCOME_TOKENS)
            for column in clock.columns
        ),
    }
    return {**support, "gates": gates, "passes_support": all(gates.values())}


def _markdown(payload: dict[str, Any]) -> str:
    support = payload["support"]
    return f"""# FPMR-1 outcome-blind support freeze — 2026-07-18

## Decision

**{'PASS' if support['passes_support'] else 'REJECT'}**. No post-entry return,
PnL, equity, CAGR, or drawdown was calculated.

- events: `{support['events']}`
- years: `{support['events_by_year']}`
- halves: `{support['events_by_half']}`
- unique ordered pairs: `{support['unique_ordered_pairs']}`
- maximum ordered-pair share: `{support['maximum_ordered_pair_share']:.4f}`
- maximum long/short symbol share: `{support['maximum_long_symbol_share']:.4f}` / `{support['maximum_short_symbol_share']:.4f}`
- all six symbols appear on both sides: `{support['gates']['all_six_symbols_long_and_short']}`

The clock includes the three frozen mechanism controls but no outcome field.
Only a passing support artifact authorizes a separately committed strict 2023
evaluator.

Clock SHA-256: `{payload['clock_sha256']}`
Manifest hash: `{payload['manifest_hash']}`
"""


def run(
    market_dir: str | Path = DEFAULT_MARKET_DIR,
    aux_dir: str | Path = DEFAULT_AUX_DIR,
    clock_output: str | Path = DEFAULT_CLOCK,
    output: str | Path = DEFAULT_OUTPUT,
    docs_output: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    registration = _verify_preregistration()
    close, funding, sources = load_predictors(market_dir, aux_dir)
    clock = build_clock(close, funding)
    clock_path = Path(clock_output)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    deterministic_csv_gz(clock, clock_path)
    support = support_summary(clock)
    core = {
        "protocol_version": "fpmr_support_v1_2026-07-18",
        "policy_id": "FPMR01",
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "protocol_hash": registration["protocol_hash"],
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "post_entry_returns_or_pnl_calculated": False,
        "source_records": sources,
        "clock_path": str(clock_path),
        "clock_sha256": sha256_file(clock_path),
        "clock_columns": list(clock.columns),
        "support": support,
        "advance_to_strict_2023_evaluator": support["passes_support"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = {**core, "manifest_hash": canonical_hash(core)}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    docs_path = Path(docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_markdown(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-dir", default=str(DEFAULT_MARKET_DIR))
    parser.add_argument("--aux-dir", default=str(DEFAULT_AUX_DIR))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--docs-output", default=str(DEFAULT_DOCS))
    args = parser.parse_args()
    payload = run(
        args.market_dir,
        args.aux_dir,
        args.clock_output,
        args.output,
        args.docs_output,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
