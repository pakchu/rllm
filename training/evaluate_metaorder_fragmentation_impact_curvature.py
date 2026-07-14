"""One-shot pre-2024 return evaluation for the frozen MFIC protocol.

The signal definition lives in the preregistration module and is not exposed as
an evaluator parameter.  This stage opens only 2020-2023 outcomes; 2024 onward
remains sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.preregister_metaorder_fragmentation_impact_curvature import (
    CANDIDATES,
    Config as SignalConfig,
    compute_mfic,
    load_causal_frame,
    nonoverlapping_schedule,
)
from training.strict_bar_backtest import _trade_stats


PREREGISTRATION_COMMIT = "81ea71c"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "51e99dbdc5ba13e6b4ac15e3915ec5b30e36dff89c1e5b31a5f3f7f272f01a59"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/metaorder-fragmentation-impact-curvature-preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "eb9ee988263fcd287e3a83b7404efcdc74557f8f8747b30683099e23c62a6373"
)
PREREGISTRATION_RESULT = Path(
    "results/metaorder_fragmentation_impact_curvature_support_2026-07-14.json"
)
PREREGISTRATION_RESULT_SHA256 = (
    "03bc5b2f67f974efa04715511920701c0db875b8bb4251f2e4c734a591aa80c8"
)

WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}


@dataclass(frozen=True)
class EvaluationConfig:
    output: str = "results/metaorder_fragmentation_impact_curvature_selection_2026-07-14.json"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_714


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_preregistration() -> dict[str, Any]:
    if _sha256(PREREGISTRATION_SOURCE) != PREREGISTRATION_SOURCE_SHA256:
        raise ValueError("MFIC preregistration source changed after freeze")
    if _sha256(PREREGISTRATION_DOCUMENT) != PREREGISTRATION_DOCUMENT_SHA256:
        raise ValueError("MFIC preregistration document changed after freeze")
    if _sha256(PREREGISTRATION_RESULT) != PREREGISTRATION_RESULT_SHA256:
        raise ValueError("MFIC preregistration result hash changed after freeze")
    result = json.loads(PREREGISTRATION_RESULT.read_text())
    if result.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("MFIC preregistration did not preserve unopened outcomes")
    if result.get("all_candidates_pass_support") is not True:
        raise ValueError("MFIC candidates did not pass frozen support gates")
    if result.get("config") != asdict(SignalConfig()):
        raise ValueError("MFIC signal config differs from the frozen support artifact")
    frozen_candidates = [item["candidate"] for item in result.get("candidates", [])]
    if frozen_candidates != [asdict(candidate) for candidate in CANDIDATES]:
        raise ValueError("MFIC candidate set differs from the frozen support artifact")
    return result


def weekly_cluster_sign_flip(
    trade_returns: list[float],
    entry_dates: list[str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    if permutations < 1:
        raise ValueError("cluster permutations must be positive")
    if len(trade_returns) != len(entry_dates):
        raise ValueError("trade returns and entry dates must have equal length")
    if not trade_returns:
        return {
            "p_value_one_sided": 1.0,
            "observed_mean_return": 0.0,
            "cluster_count": 0,
            "permutations": int(permutations),
            "seed": int(seed),
        }

    dates = pd.to_datetime(pd.Series(entry_dates), utc=True, errors="raise")
    monday = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.floor("D")
    values = np.asarray(trade_returns, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("trade returns must be finite")
    cluster_sums = (
        pd.DataFrame({"week": monday, "return": values})
        .groupby("week", sort=True, observed=True)["return"]
        .sum()
        .to_numpy(float)
    )
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    batch_size = 4_096
    while completed < permutations:
        batch = min(batch_size, permutations - completed)
        signs = rng.integers(
            0,
            2,
            size=(batch, len(cluster_sums)),
            dtype=np.int8,
        ).astype(float)
        signs = signs * 2.0 - 1.0
        permuted = signs.dot(cluster_sums) / len(values)
        exceedances += int(np.count_nonzero(permuted >= observed))
        completed += batch
    return {
        "p_value_one_sided": float((1 + exceedances) / (permutations + 1)),
        "observed_mean_return": observed,
        "cluster_count": int(len(cluster_sums)),
        "permutations": int(permutations),
        "seed": int(seed),
    }


def simulate_schedule(
    frame: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("simulation start must be before end")
    if cfg.leverage <= 0.0:
        raise ValueError("leverage must be positive")
    per_side_cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    if not 0.0 <= per_side_cost < 1.0:
        raise ValueError("per-side execution cost is invalid")

    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    dates = frame["date"]
    equity = 1.0
    peak = 1.0
    strict_mdd = 0.0
    previous_exit = -1
    trade_returns: list[float] = []
    entry_dates: list[str] = []
    sides: list[int] = []
    branches: list[str] = []

    for row in schedule.itertuples(index=False):
        signal_position = int(row.signal_position)
        entry_position = int(row.entry_position)
        exit_position = int(row.exit_position)
        side = int(row.side)
        if side not in (-1, 1):
            raise ValueError("scheduled side must be long or short")
        if not signal_position < entry_position < exit_position:
            raise ValueError("scheduled positions are not strictly ordered")
        if entry_position < previous_exit:
            raise ValueError("scheduled trades overlap")
        if exit_position >= len(frame):
            raise ValueError("scheduled exit exceeds market frame")
        if not (
            start_timestamp <= dates.iloc[signal_position] < end_timestamp
            and start_timestamp <= dates.iloc[entry_position] < end_timestamp
            and start_timestamp <= dates.iloc[exit_position] < end_timestamp
        ):
            raise ValueError("scheduled trade crosses the simulation split")
        entry_price = float(opens[entry_position])
        exit_price = float(opens[exit_position])
        if entry_price <= 0.0 or exit_price <= 0.0:
            raise ValueError("scheduled trade has non-positive open price")
        held_high = float(np.max(highs[entry_position:exit_position]))
        held_low = float(np.min(lows[entry_position:exit_position]))

        entry_equity = equity
        equity *= 1.0 - per_side_cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        favorable_price = held_high if side > 0 else held_low
        adverse_price = held_low if side > 0 else held_high
        favorable_equity = max(
            0.0,
            equity
            * (1.0 + cfg.leverage * side * (favorable_price / entry_price - 1.0)),
        )
        intratrade_peak = max(peak, favorable_equity)
        adverse_equity = max(
            0.0,
            equity
            * (1.0 + cfg.leverage * side * (adverse_price / entry_price - 1.0)),
        )
        strict_mdd = max(strict_mdd, 1.0 - adverse_equity / intratrade_peak)
        peak = max(peak, intratrade_peak)

        raw_return = side * (exit_price / entry_price - 1.0)
        equity *= max(0.0, 1.0 + cfg.leverage * raw_return)
        equity *= 1.0 - per_side_cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        trade_returns.append(equity / entry_equity - 1.0)
        entry_dates.append(str(dates.iloc[entry_position]))
        sides.append(side)
        branches.append(str(row.branch))
        previous_exit = exit_position

    years = (end_timestamp - start_timestamp).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    strict_mdd_pct = strict_mdd * 100.0
    trade_stats = _trade_stats(trade_returns)
    cluster = weekly_cluster_sign_flip(
        trade_returns,
        entry_dates,
        permutations=cfg.cluster_permutations,
        seed=cfg.cluster_seed,
    )
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_mdd_pct),
        "cagr_to_strict_mdd": (
            float(cagr / strict_mdd_pct) if strict_mdd_pct > 1e-12 else 0.0
        ),
        "trade_count": int(len(trade_returns)),
        "long_count": int(sum(side > 0 for side in sides)),
        "short_count": int(sum(side < 0 for side in sides)),
        "continuation_count": int(sum(branch == "continuation" for branch in branches)),
        "fade_count": int(sum(branch == "fade" for branch in branches)),
        "wall_clock_years": float(years),
        "trade_statistics": trade_stats,
        "weekly_cluster_sign_flip": cluster,
    }


def _qualification(candidate: dict[str, Any]) -> dict[str, Any]:
    train = candidate["windows"]["train"]
    select = candidate["windows"]["select2023"]
    h1 = candidate["windows"]["select2023_h1"]
    h2 = candidate["windows"]["select2023_h2"]
    failures: list[str] = []
    for name, metrics in (("train", train), ("select2023", select)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if metrics["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}: strict MDD above 15%")
        if metrics["weekly_cluster_sign_flip"]["p_value_one_sided"] >= 0.10:
            failures.append(f"{name}: weekly-cluster p-value not below 0.10")
    for name, metrics in (("select2023_h1", h1), ("select2023_h2", h2)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < 40:
            failures.append(f"{name}: fewer than 40 trades")
    if select["trade_count"] < 100:
        failures.append("select2023: fewer than 100 trades")
    return {"qualifies": not failures, "failures": failures}


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [item for item in candidates if item["qualification"]["qualifies"]]
    if not qualified:
        return {
            "selected_candidate": None,
            "rejected": True,
            "reason": "no frozen MFIC candidate passed every selection gate",
        }
    selected = sorted(
        qualified,
        key=lambda item: (
            -min(
                item["windows"]["train"]["cagr_to_strict_mdd"],
                item["windows"]["select2023"]["cagr_to_strict_mdd"],
            ),
            item["windows"]["select2023"]["strict_mdd_pct"],
            item["candidate"]["name"],
        ),
    )[0]
    return {
        "selected_candidate": selected["candidate"]["name"],
        "rejected": False,
        "reason": "passed frozen train/select gates",
    }


def run_evaluation(cfg: EvaluationConfig) -> dict[str, Any]:
    preregistration = _verify_preregistration()
    signal_cfg = SignalConfig()
    frame, source = load_causal_frame(signal_cfg)
    candidates: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        signal = compute_mfic(frame, candidate, signal_cfg)
        windows: dict[str, Any] = {}
        for name, (start, end) in WINDOWS.items():
            schedule = nonoverlapping_schedule(
                signal,
                frame,
                start=start,
                end=end,
            )
            windows[name] = simulate_schedule(
                frame,
                schedule,
                start=start,
                end=end,
                cfg=cfg,
            )
        item = {"candidate": asdict(candidate), "windows": windows}
        item["qualification"] = _qualification(item)
        candidates.append(item)

    result = {
        "protocol": {
            "name": "MFIC frozen pre-2024 selection evaluation",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
            "preregistration_result_sha256": PREREGISTRATION_RESULT_SHA256,
            "outcomes_opened": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "signal_parameters_mutable": False,
            "entry": "next 5m open",
            "exit": "scheduled future 5m open",
            "strict_mdd": "complete held path, favorable extreme first then adverse extreme",
            "cagr": "full wall-clock split including idle cash",
        },
        "evaluation_config": asdict(cfg),
        "signal_config": preregistration["config"],
        "source": source,
        "candidates": candidates,
        "selection": _select_candidate(candidates),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=EvaluationConfig.output)
    args = parser.parse_args()
    cfg = EvaluationConfig(output=args.output)
    result = run_evaluation(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "candidates": [
                    {
                        "name": item["candidate"]["name"],
                        "qualification": item["qualification"],
                        "windows": {
                            name: {
                                key: value[key]
                                for key in (
                                    "absolute_return_pct",
                                    "cagr_pct",
                                    "strict_mdd_pct",
                                    "cagr_to_strict_mdd",
                                    "trade_count",
                                )
                            }
                            for name, value in item["windows"].items()
                        },
                    }
                    for item in result["candidates"]
                ],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
