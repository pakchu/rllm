"""Validate Top-10 state-model survivors under a shared strict protocol."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.portfolio_opt_new_alpha_pool import _alpha_active
from training.search_bidirectional_state_alpha import Config, sim
from training.search_bocpd_state_gated_alpha import _map_output, _model_output, _state_from_mapped
from training.search_gaussian_hmm_regime_alpha import SPLITS, hourly_features
from training.search_kalman_state_gated_alpha import kalman_hourly_state, map_hourly_state
from training.search_semimarkov_duration_alpha import (
    WINDOWS,
    _bootstrap_mean_trade_return,
    _trade_returns,
    duration_key,
    map_hourly_key,
    observable_state,
)


YEAR_WINDOWS = {
    "2020": ("2020-01-01", "2021-01-01"),
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
    "2026": ("2026-01-01", "2026-06-02"),
}


def _load_result(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _ranked_live_rows(result: dict) -> list[tuple[int, dict]]:
    live_hashes = {row["signal_hash"] for row in result.get("live_grade", [])}
    return [
        (rank, row)
        for rank, row in enumerate(result["selected"][:10], start=1)
        if row["signal_hash"] in live_hashes
    ]


def _reconstruct_kalman(
    row: dict,
    market: pd.DataFrame,
    dates: pd.Series,
    setup: np.ndarray,
    hourly: pd.DataFrame,
) -> np.ndarray:
    train_hour = np.asarray(
        (hourly.index >= SPLITS["train"][0]) & (hourly.index < SPLITS["train"][1]),
        dtype=bool,
    )
    state_frame, _ = kalman_hourly_state(
        hourly,
        train_hour,
        q_level=row["q_level"],
        q_slope=row["q_slope"],
        r_obs=row["r_obs"],
        low_quantile=row["state_quantiles"][0],
        high_quantile=row["state_quantiles"][1],
    )
    state = map_hourly_state(dates, state_frame)
    return setup & np.isin(state, row["allowed_states"])


def _reconstruct_bocpd(
    row: dict,
    dates: pd.Series,
    setup: np.ndarray,
    hourly_feature: pd.DataFrame,
) -> np.ndarray:
    train_hour = np.asarray(
        (hourly_feature.index >= SPLITS["train"][0])
        & (hourly_feature.index < SPLITS["train"][1]),
        dtype=bool,
    )
    secondary_index = None if row["model_name"] == "return" else 1
    output, _ = _model_output(
        hourly_feature,
        train_hour,
        columns=tuple(row["model"]["columns"]),
        secondary_index=secondary_index,
        hazard_lambda=int(row["model"]["hazard_lambda_hours"]),
    )
    state = _state_from_mapped(_map_output(dates, output), row["state_thresholds"])
    return setup & np.isin(state, row["allowed_states"])


def _reconstruct_semimarkov(
    row: dict,
    dates: pd.Series,
    setup: np.ndarray,
    hourly_feature: pd.DataFrame,
) -> np.ndarray:
    fit_hour = np.asarray(
        (hourly_feature.index >= WINDOWS["fit2020_2022"][0])
        & (hourly_feature.index < WINDOWS["fit2020_2022"][1]),
        dtype=bool,
    )
    state, _ = observable_state(hourly_feature, fit_hour, *row["trend_quantiles"])
    key, _ = duration_key(
        state,
        tuple(row["duration_cutpoints_hours"]),
        timestamps=hourly_feature.index,
    )
    mapped_key = map_hourly_key(dates, hourly_feature.index, key)
    return setup & np.isin(mapped_key, row["allowed_keys"])


def _entry_positions(
    active: np.ndarray,
    dates: pd.Series,
    start: str = "2024-01-01",
    end: str = "2026-06-02",
) -> set[int]:
    mask = _split_mask(dates, start, end)
    positions = np.arange(143, len(active) - 578, 12, dtype=np.int64)
    positions = positions[mask[positions] & active[positions]]
    entries: set[int] = set()
    next_allowed = 0
    for pos in positions:
        if pos < next_allowed or not mask[int(pos) + 577]:
            continue
        entries.add(int(pos))
        next_allowed = int(pos) + 578
    return entries


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def run(args: argparse.Namespace) -> dict:
    source_results = {
        "kalman": _load_result(args.kalman_result),
        "bocpd": _load_result(args.bocpd_result),
        "semimarkov": _load_result(args.semimarkov_result),
    }
    base_config = source_results["kalman"]["config"]
    config_values = {
        name: base_config[name]
        for name in Config.__dataclass_fields__
        if name in base_config
    }
    config_values.update(
        input_csv=args.input_csv,
        output=args.output,
        funding_csv=args.funding_csv,
        premium_csv=args.premium_csv,
        exclude_from=args.exclude_from,
    )
    cfg = Config(**config_values)
    for name, bounds in WINDOWS.items():
        state_sim.W[name] = bounds
    for year, bounds in YEAR_WINDOWS.items():
        state_sim.W[f"year_{year}"] = bounds

    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    setup = _alpha_active(features, "long_minimal_funding_premium")
    hourly, hourly_feature = hourly_features(market)
    zero = np.zeros(len(market), dtype=bool)

    candidates: list[dict] = []
    signals: dict[str, np.ndarray] = {}
    for family, result in source_results.items():
        for rank, row in _ranked_live_rows(result):
            candidate_id = f"{family}_rank{rank}"
            if family == "kalman":
                active = _reconstruct_kalman(row, market, dates, setup, hourly)
            elif family == "bocpd":
                active = _reconstruct_bocpd(row, dates, setup, hourly_feature)
            else:
                active = _reconstruct_semimarkov(row, dates, setup, hourly_feature)
            signals[candidate_id] = active
            yearly = {
                year: sim(market, dates, active, zero, cfg, 576, 12, 10.0, 10.0, f"year_{year}")
                for year in YEAR_WINDOWS
            }
            stress = {}
            for bps in (6, 8, 10, 15):
                stressed = replace(cfg, fee_rate=max(0.0, bps / 10000 - cfg.slippage_rate))
                stress[str(bps)] = {
                    split: sim(market, dates, active, zero, stressed, 576, 12, 10.0, 10.0, split)
                    for split in ("test2024", "eval2025", "ytd2026")
                }
            bootstrap = {
                split: _bootstrap_mean_trade_return(_trade_returns(market, dates, active, cfg, split))
                for split in ("test2024", "eval2025", "ytd2026")
            }
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "family": family,
                    "pre_evaluation_rank": rank,
                    "source_signal_hash": row["signal_hash"],
                    "source_parameters": {
                        key: row[key]
                        for key in (
                            "q_level",
                            "q_slope",
                            "r_obs",
                            "state_quantiles",
                            "allowed_states",
                            "model_name",
                            "model",
                            "duration_cutpoints_hours",
                            "allowed_keys",
                        )
                        if key in row
                    },
                    "source_metrics": {
                        split: row[split]
                        for split in ("test2024", "eval2025", "ytd2026")
                    },
                    "yearly": yearly,
                    "cost_stress_bps_per_side": stress,
                    "trade_return_bootstrap": bootstrap,
                    "oos_entry_count": len(_entry_positions(active, dates)),
                }
            )

    entry_sets = {name: _entry_positions(signal, dates) for name, signal in signals.items()}
    overlap = {
        f"{left}|{right}": {
            "jaccard": _jaccard(entry_sets[left], entry_sets[right]),
            "intersection": len(entry_sets[left] & entry_sets[right]),
            "union": len(entry_sets[left] | entry_sets[right]),
        }
        for left, right in itertools.combinations(sorted(entry_sets), 2)
    }
    canonical = {}
    for family in source_results:
        family_rows = [row for row in candidates if row["family"] == family]
        if family_rows:
            canonical[family] = min(family_rows, key=lambda row: row["pre_evaluation_rank"])[
                "candidate_id"
            ]

    output = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": (
            "qualifying members of each pre-evaluation Top-10 family; representative is the "
            "highest pre-evaluation rank among OOS passers; full-window CAGR; strict intratrade "
            "MDD; 6/8/10/15bp cost stress; trade bootstrap; no OOS ranking within passers"
        ),
        "candidate_count": len(candidates),
        "canonical_by_family": canonical,
        "candidates": candidates,
        "oos_entry_overlap": overlap,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--funding-csv", default="")
    parser.add_argument("--premium-csv", default="")
    parser.add_argument("--exclude-from", default="2026-06-02")
    parser.add_argument(
        "--kalman-result",
        default="results/kalman_state_gated_alpha_scan_2026-07-13.json",
    )
    parser.add_argument(
        "--bocpd-result",
        default="results/bocpd_state_gated_alpha_scan_2026-07-13.json",
    )
    parser.add_argument(
        "--semimarkov-result",
        default="results/semimarkov_duration_alpha_scan_2026-07-13.json",
    )
    args = parser.parse_args()
    output = run(args)
    print(
        json.dumps(
            {
                "candidate_count": output["candidate_count"],
                "canonical_by_family": output["canonical_by_family"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
