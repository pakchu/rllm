"""Build DCRM-1's outcome-blind 2023-2024 weekly support clock."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    resolve,
    sha256_file,
)
from training.preregister_dispersion_conditioned_residual_momentum import (
    canonical_hash,
    protocol,
)


SYMBOLS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2025-01-01 00:00:00")
POLICY_ID = "DCRM01"
PREREGISTRATION = Path(
    "results/dispersion_conditioned_residual_momentum_preregistration_2026-07-17.json"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "bfbdea9e288c2553e33ebdef1a6b4ccd6bdaf7d3c46b5b7fbc2d9d42413ca145"
)
EXPECTED_PROTOCOL_HASH = "e41f3acdb7297c6704db2f225eea0764d2e8252285713f282d07bdc8a6ffb4eb"
DEFAULT_SOURCE_DIR = Path("data/binance_um_lore_2023_2024")
DEFAULT_CLOCK = Path(
    "data/dispersion_conditioned_residual_momentum_support_clock_2023_2024.csv.gz"
)
DEFAULT_MANIFEST = Path(
    "results/dispersion_conditioned_residual_momentum_support_2026-07-17.json"
)
DEFAULT_DOCS = Path(
    "docs/dispersion-conditioned-residual-momentum-support-2026-07-17.md"
)
LORE_CLOCK = Path("data/leave_one_out_residual_exhaustion_v1_support_clocks_2023_2024.csv.gz")

CLOCK_COLUMNS = (
    "policy_id",
    "decision_time",
    "last_feature_time",
    "entry_time",
    "exit_time",
    "long_symbol",
    "short_symbol",
    "long_weight",
    "short_weight_abs",
    "base_long_weight",
    "base_short_weight_abs",
    "gross_scale",
    "long_beta",
    "short_beta",
    "long_momentum_30d",
    "short_momentum_30d",
    "long_factor_30d",
    "short_factor_30d",
    "long_score",
    "short_score",
    "score_dispersion",
    "prior_dispersion_q80",
)
FORBIDDEN_OUTCOME_TOKENS = (
    "pnl",
    "equity",
    "trade_return",
    "absolute_return",
    "cagr",
    "drawdown",
    "exit_price",
    "entry_price",
)


def _verify_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError("DCRM-1 preregistration file changed")
    payload = json.loads(PREREGISTRATION.read_text())
    if payload.get("protocol_hash") != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("DCRM-1 preregistration identity changed")
    if canonical_hash(payload["protocol"]) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("DCRM-1 preregistration body changed")
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("DCRM-1 implementation protocol drifted")
    if payload["protocol"]["evidence_boundary"]["post_entry_returns_or_equity_opened"]:
        raise RuntimeError("DCRM-1 outcomes opened before support")
    return payload


def load_close_panel(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    symbols: Iterable[str] = SYMBOLS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Read only timestamps and closes from the physically frozen prefix."""

    panel: dict[str, pd.Series] = {}
    records: list[dict[str, Any]] = []
    for symbol in symbols:
        path = resolve(Path(source_dir) / f"{symbol}_5m_2023_2024.csv.gz")
        frame = pd.read_csv(path, usecols=["date", "close"])
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise").dt.tz_convert(None)
        frame["close"] = pd.to_numeric(frame["close"], errors="raise")
        if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
            raise ValueError(f"{symbol} duplicate or unsorted timestamps")
        frame = frame.loc[(frame["date"] >= START) & (frame["date"] < END)]
        if frame.empty or not np.isfinite(frame["close"]).all() or (frame["close"] <= 0).any():
            raise ValueError(f"{symbol} invalid close prefix")
        panel[symbol] = frame.set_index("date")["close"]
        records.append(
            {
                "symbol": symbol,
                "path": str(path),
                "sha256": sha256_file(path),
                "rows_read": int(len(frame)),
                "columns_read": ["date", "close"],
                "first_timestamp": frame["date"].iloc[0].isoformat(),
                "last_timestamp": frame["date"].iloc[-1].isoformat(),
                "rows_at_or_after_2025_read": 0,
            }
        )
    close = pd.DataFrame(panel).sort_index()
    if tuple(close.columns) != tuple(symbols):
        raise ValueError("DCRM-1 symbol order changed")
    return close, records


def feature_panels(
    close: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build causal LOO betas and 30-day factor returns from completed hours."""

    hourly = close.loc[close.index.minute == 55].copy()
    if hourly.empty:
        raise ValueError("no completed xx:55 closes")
    full_grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq="1h")
    hourly = hourly.reindex(full_grid)
    hourly_return = np.log(hourly / hourly.shift(1))
    factors: dict[str, pd.Series] = {}
    betas: dict[str, pd.Series] = {}
    for symbol in close.columns:
        others = [candidate for candidate in close.columns if candidate != symbol]
        other_returns = hourly_return[others]
        factor = other_returns.median(axis=1, skipna=False)
        covariance = hourly_return[symbol].rolling(720, min_periods=336).cov(factor)
        variance = factor.rolling(720, min_periods=336).var()
        factors[symbol] = factor
        betas[symbol] = (covariance / variance).shift(1).clip(0.25, 2.5)
    factor_hourly = pd.DataFrame(factors)
    beta = pd.DataFrame(betas)
    factor_30d = factor_hourly.rolling(720, min_periods=720).sum()
    return hourly_return, factor_hourly, factor_30d, beta


def _lexical_extreme(values: pd.Series, *, maximum: bool) -> str:
    ordered = sorted(
        ((float(value), str(symbol)) for symbol, value in values.items()),
        key=(lambda item: (-item[0], item[1])) if maximum else (lambda item: (item[0], item[1])),
    )
    return ordered[0][1]


def candidate_states(close: pd.DataFrame) -> pd.DataFrame:
    """Return weekly causal states before the eight-state dispersion warm-up."""

    _, _, factor_30d, beta = feature_panels(close)
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range("2023-01-02", "2024-12-30", freq="W-MON"):
        last_feature = decision - pd.Timedelta(minutes=5)
        momentum_start = last_feature - pd.Timedelta(days=30)
        if (
            last_feature not in close.index
            or momentum_start not in close.index
            or last_feature not in factor_30d.index
            or last_feature not in beta.index
        ):
            continue
        momentum = np.log(close.loc[last_feature] / close.loc[momentum_start])
        current_beta = beta.loc[last_feature]
        current_factor = factor_30d.loc[last_feature]
        if momentum.isna().any() or current_beta.isna().any() or current_factor.isna().any():
            continue
        score = momentum - current_beta * current_factor
        long_symbol = _lexical_extreme(score, maximum=True)
        short_symbol = _lexical_extreme(score, maximum=False)
        long_beta = float(current_beta[long_symbol])
        short_beta = float(current_beta[short_symbol])
        beta_sum = long_beta + short_beta
        rows.append(
            {
                "policy_id": POLICY_ID,
                "decision_time": decision,
                "last_feature_time": last_feature,
                "entry_time": decision + pd.Timedelta(minutes=5),
                "exit_time": decision + pd.Timedelta(days=7, minutes=5),
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "base_long_weight": short_beta / beta_sum,
                "base_short_weight_abs": long_beta / beta_sum,
                "long_beta": long_beta,
                "short_beta": short_beta,
                "long_momentum_30d": float(momentum[long_symbol]),
                "short_momentum_30d": float(momentum[short_symbol]),
                "long_factor_30d": float(current_factor[long_symbol]),
                "short_factor_30d": float(current_factor[short_symbol]),
                "long_score": float(score[long_symbol]),
                "short_score": float(score[short_symbol]),
                "score_dispersion": float(score.std(ddof=0)),
            }
        )
    states = pd.DataFrame(rows)
    if states.empty:
        return states
    states["prior_dispersion_q80"] = (
        states["score_dispersion"]
        .rolling(26, min_periods=8)
        .quantile(0.8, interpolation="linear")
        .shift(1)
    )
    return states


def build_clock(close: pd.DataFrame) -> pd.DataFrame:
    states = candidate_states(close)
    if states.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    clock = states.loc[states["prior_dispersion_q80"].notna()].copy()
    clock["gross_scale"] = np.where(
        clock["score_dispersion"] <= clock["prior_dispersion_q80"], 1.0, 0.25
    )
    clock["long_weight"] = clock["base_long_weight"] * clock["gross_scale"]
    clock["short_weight_abs"] = clock["base_short_weight_abs"] * clock["gross_scale"]
    return clock.loc[:, CLOCK_COLUMNS].reset_index(drop=True)


def assert_clock_contract(clock: pd.DataFrame) -> None:
    if tuple(clock.columns) != CLOCK_COLUMNS:
        raise RuntimeError("DCRM-1 clock schema changed")
    forbidden = [
        column
        for column in clock.columns
        if any(token in column.lower() for token in FORBIDDEN_OUTCOME_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"DCRM-1 outcome columns escaped into support: {forbidden}")
    if clock.empty:
        return
    checked = clock.copy()
    times = ("decision_time", "last_feature_time", "entry_time", "exit_time")
    for column in times:
        checked[column] = pd.to_datetime(checked[column], errors="raise")
    if not checked["policy_id"].eq(POLICY_ID).all():
        raise RuntimeError("DCRM-1 policy identity changed")
    if checked["decision_time"].duplicated().any() or not checked["decision_time"].is_monotonic_increasing:
        raise RuntimeError("DCRM-1 weekly decisions duplicate or reorder")
    if not (
        (checked["decision_time"].dt.weekday == 0)
        & (checked["decision_time"].dt.hour == 0)
        & (checked["decision_time"].dt.minute == 0)
    ).all():
        raise RuntimeError("DCRM-1 decision boundary changed")
    if not (checked["last_feature_time"] == checked["decision_time"] - pd.Timedelta(minutes=5)).all():
        raise RuntimeError("DCRM-1 feature cutoff changed")
    if not (checked["entry_time"] == checked["decision_time"] + pd.Timedelta(minutes=5)).all():
        raise RuntimeError("DCRM-1 entry delay changed")
    if not (checked["exit_time"] == checked["entry_time"] + pd.Timedelta(days=7)).all():
        raise RuntimeError("DCRM-1 hold changed")
    numeric = [column for column in CLOCK_COLUMNS if column not in {"policy_id", *times, "long_symbol", "short_symbol"}]
    values = checked[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("DCRM-1 non-finite support feature")
    if not checked["gross_scale"].isin([0.25, 1.0]).all():
        raise RuntimeError("DCRM-1 gross scale changed")
    gross = checked["long_weight"] + checked["short_weight_abs"]
    if not np.allclose(gross, checked["gross_scale"], atol=1e-12):
        raise RuntimeError("DCRM-1 executed gross mismatch")
    base_gross = checked["base_long_weight"] + checked["base_short_weight_abs"]
    if not np.allclose(base_gross, 1.0, atol=1e-12):
        raise RuntimeError("DCRM-1 base gross mismatch")
    long_beta_dollar = checked["base_long_weight"] * checked["long_beta"]
    short_beta_dollar = checked["base_short_weight_abs"] * checked["short_beta"]
    if not np.allclose(long_beta_dollar, short_beta_dollar, atol=1e-12):
        raise RuntimeError("DCRM-1 beta neutralization changed")
    if (checked["long_symbol"] == checked["short_symbol"]).any():
        raise RuntimeError("DCRM-1 selected the same symbol twice")
    if not (checked["long_score"] >= checked["short_score"]).all():
        raise RuntimeError("DCRM-1 score direction changed")


def _concentration(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"events": 0, "unique_ordered_pairs": 0, "maximum_pair_share": None, "maximum_month_share": None}
    pair = frame["long_symbol"].astype(str) + "__" + frame["short_symbol"].astype(str)
    month = pd.to_datetime(frame["decision_time"]).dt.to_period("M").astype(str)
    return {
        "events": int(len(frame)),
        "unique_ordered_pairs": int(pair.nunique()),
        "maximum_pair_share": float(pair.value_counts(normalize=True).max()),
        "maximum_month_share": float(month.value_counts(normalize=True).max()),
    }


def support_stats(clock: pd.DataFrame) -> dict[str, Any]:
    checked = clock.copy()
    checked["decision_time"] = pd.to_datetime(checked["decision_time"])
    checked["year"] = checked["decision_time"].dt.year.astype(str)
    checked["half"] = checked["decision_time"].map(
        lambda value: f"{value.year}H{1 if value.month <= 6 else 2}"
    )
    checked["pair"] = checked["long_symbol"] + "__" + checked["short_symbol"]
    checked["month"] = checked["decision_time"].dt.to_period("M").astype(str)
    year_counts = checked["year"].value_counts().sort_index().astype(int).to_dict()
    half_counts = checked["half"].value_counts().sort_index().astype(int).to_dict()
    pair_counts = checked["pair"].value_counts()
    month_counts = checked["month"].value_counts()
    scale_counts = {
        str(scale): int(count)
        for scale, count in checked["gross_scale"].value_counts().sort_index().items()
    }
    scale_year_counts: dict[str, dict[str, int]] = {}
    scale_half_counts: dict[str, dict[str, int]] = {}
    scale_concentration: dict[str, dict[str, Any]] = {}
    for scale, group in checked.groupby("gross_scale", sort=True):
        key = str(scale)
        scale_year_counts[key] = group["year"].value_counts().sort_index().astype(int).to_dict()
        scale_half_counts[key] = group["half"].value_counts().sort_index().astype(int).to_dict()
        scale_concentration[key] = _concentration(group)
    gates = protocol()["support_gate"]
    stats: dict[str, Any] = {
        "events": int(len(checked)),
        "year_counts": year_counts,
        "half_counts": half_counts,
        "gross_scale_counts": scale_counts,
        "gross_scale_year_counts": scale_year_counts,
        "gross_scale_half_counts": scale_half_counts,
        "gross_scale_concentration": scale_concentration,
        "unique_ordered_pairs": int(pair_counts.size),
        "maximum_ordered_pair_share": float(pair_counts.max() / len(checked)),
        "maximum_month_share": float(month_counts.max() / len(checked)),
        "long_symbols": sorted(checked["long_symbol"].unique().tolist()),
        "short_symbols": sorted(checked["short_symbol"].unique().tolist()),
        "top_ordered_pairs": {str(key): int(value) for key, value in pair_counts.head(10).items()},
    }
    stats["passes_support"] = bool(
        stats["events"] >= gates["events_2023_2024_at_least"]
        and min(year_counts.values()) >= gates["events_each_year_at_least"]
        and min(half_counts.values()) >= gates["events_each_half_at_least"]
        and stats["unique_ordered_pairs"] >= gates["unique_ordered_pairs_at_least"]
        and stats["maximum_ordered_pair_share"] <= gates["maximum_ordered_pair_share_at_most"]
        and stats["maximum_month_share"] <= gates["maximum_month_share_at_most"]
        and set(stats["long_symbols"]) == set(SYMBOLS)
        and set(stats["short_symbols"]) == set(SYMBOLS)
    )
    return stats


def occupied_clock(clock: pd.DataFrame, *, frequency: str = "5min") -> set[pd.Timestamp]:
    occupied: set[pd.Timestamp] = set()
    for row in clock.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        occupied.update(pd.date_range(entry, exit_time - pd.Timedelta(frequency), freq=frequency))
    return occupied


def clock_overlap(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    candidate_entries = set(pd.to_datetime(candidate["entry_time"]))
    reference_entries = set(pd.to_datetime(reference["entry_time"]))
    entry_union = candidate_entries | reference_entries
    candidate_position = occupied_clock(candidate)
    reference_position = occupied_clock(reference)
    position_union = candidate_position | reference_position
    return {
        "post_entry_returns_or_pnl_read": False,
        "candidate_entries": len(candidate_entries),
        "reference_entries": len(reference_entries),
        "exact_entry_jaccard": len(candidate_entries & reference_entries) / len(entry_union) if entry_union else 0.0,
        "position_time_jaccard_5m": len(candidate_position & reference_position) / len(position_union) if position_union else 0.0,
    }


def _markdown(result: dict[str, Any]) -> str:
    stats = result["support"]
    lore = result["outcome_blind_clock_overlap"]["LORE_2023_2024"]
    return f"""# DCRM-1 outcome-blind support — 2026-07-17

- Post-entry returns/PnL calculated: **no**
- Accepted weekly states: `{stats['events']}`; years `{stats['year_counts']}`; halves `{stats['half_counts']}`
- Gross buckets: `{stats['gross_scale_counts']}`
- Unique ordered pairs: `{stats['unique_ordered_pairs']}`
- Maximum pair share: `{stats['maximum_ordered_pair_share']:.2%}`
- Maximum month share: `{stats['maximum_month_share']:.2%}`
- Long symbols: `{stats['long_symbols']}`
- Short symbols: `{stats['short_symbols']}`
- LORE exact-entry Jaccard: `{lore['exact_entry_jaccard']:.4f}`
- LORE 5m position-time Jaccard: `{lore['position_time_jaccard_5m']:.4f}`
- Support gate: **{'PASS' if stats['passes_support'] else 'REJECT'}**
- Clock SHA-256: `{result['clock_sha256']}`

The builder read only timestamps and closes through 2024. The weekly clock,
leave-one-out factors, shifted betas, pair identities, and gross scales contain
no entry/exit price, post-entry return, PnL, equity, 2025, or 2026 outcome.
The 0.25 and 1.0 buckets are reported separately so nominal event count does
not hide reduced exposure. LORE overlap also uses clocks only, not returns.
"""


def run(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    clock_path: str | Path = DEFAULT_CLOCK,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    docs_path: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    preregistration = _verify_preregistration()
    close, source_records = load_close_panel(source_dir)
    clock = build_clock(close)
    assert_clock_contract(clock)
    stats = support_stats(clock)
    output_clock = Path(clock_path)
    deterministic_csv_gz(clock, output_clock)
    reread = pd.read_csv(output_clock)
    assert_clock_contract(reread)
    lore_path = resolve(LORE_CLOCK)
    lore_clock = pd.read_csv(lore_path, usecols=["entry_time", "exit_time"])
    overlap = clock_overlap(clock, lore_clock)
    result: dict[str, Any] = {
        "protocol_version": "dcrm_v1_support_2026-07-17",
        "preregistration_protocol_hash": preregistration["protocol_hash"],
        "post_entry_returns_or_pnl_calculated": False,
        "market_columns_read": ["date", "close"],
        "2023_selection_outcomes_opened": False,
        "2024_test_outcomes_opened": False,
        "2025_eval_outcomes_opened": False,
        "2026_holdout_outcomes_opened": False,
        "clock_path": str(output_clock),
        "clock_sha256": sha256_file(output_clock),
        "source_records": source_records,
        "support": stats,
        "outcome_blind_clock_overlap": {
            "LORE_2023_2024": {
                **overlap,
                "reference_path": str(lore_path),
                "reference_sha256": sha256_file(lore_path),
            },
            "LORC_2025": {
                "calendar_overlap": False,
                "reason": "DCRM support qualification ends before 2025",
                "post_entry_returns_or_pnl_read": False,
            },
            "active_live_sleeves": {
                "exact_historical_clocks_available": False,
                "deferred_until_standalone_pass": True,
                "post_entry_returns_or_pnl_read": False,
            },
        },
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output_manifest = Path(manifest_path)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    output_docs = Path(docs_path)
    output_docs.parent.mkdir(parents=True, exist_ok=True)
    output_docs.write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--clock", default=str(DEFAULT_CLOCK))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir, args.clock, args.manifest, args.docs), indent=2))


if __name__ == "__main__":
    main()
