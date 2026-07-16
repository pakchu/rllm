"""One-shot strict pre-2024 evaluator for frozen LVRT-R0.

This module cannot load execution prices until its own source has been frozen
by ``freeze_liquidity_vacuum_replenishment_evaluator``.  It replays the frozen
primary clock, evaluates preregistered controls, and never opens 2024+ data.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_liquidity_vacuum_replenishment_support as support
from training import preregister_liquidity_vacuum_replenishment as prereg


SUPPORT_COMMIT = "70746f8fc6673913df0b272f3b623f321f5fa220"
SUPPORT_SOURCE_SHA256 = (
    "edb277fb8c1fbd25261c82e58e79350c01d16db060174c7136aa0f18f7485d88"
)
SUPPORT_DOCUMENT = Path(
    "docs/liquidity-vacuum-replenishment-r0-support-freeze-2026-07-17.md"
)
SUPPORT_DOCUMENT_SHA256 = (
    "30a0646fdc70e937c42837d3bcfaff5618d8d28568276fd0ad02e29b23279005"
)
SUPPORT_RESULT = Path(support.DEFAULT_OUTPUT)
SUPPORT_RESULT_SHA256 = (
    "bbce868ab2ca861bb1e56d49d4be228d20fe7e63f4dbaf66ff7b0eb1f8a3fbc6"
)
EVENT_CLOCK = Path(support.DEFAULT_CLOCK)
EVENT_CLOCK_SHA256 = (
    "ed9dd6391df2118ac09d147a4e57c3cb3f6e105a13f6c0d973ee424cfedd54d2"
)
FUNDING_DATA = Path("results/binance_um_btcusdt_realized_funding_2020_2023.csv")
FUNDING_DATA_SHA256 = (
    "c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7"
)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_realized_funding_2020_2023_manifest.json"
)
FUNDING_MANIFEST_SHA256 = (
    "c70280e46bcbc2410cc59c2bcc93780c40997dbc5d0edb82d82127b59593250c"
)
EVALUATION_SOURCE = Path("training/evaluate_liquidity_vacuum_replenishment.py")
EVALUATION_FREEZE = Path(
    "results/liquidity_vacuum_replenishment_evaluator_freeze_2026-07-17.json"
)
DEFAULT_OUTPUT = "results/liquidity_vacuum_replenishment_selection_2026-07-17.json"
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
POLICY_NAMES = (
    "primary",
    "direction_flip",
    "no_reversal_confirmation",
    "one_bar_extra_delay",
    "one_day_shifted_setup",
    "sign_permuted_confirmation",
)
REJECTION_PLACEBOS = (
    "one_day_shifted_setup",
    "sign_permuted_confirmation",
)


@dataclass(frozen=True)
class EvaluationConfig:
    output: str = DEFAULT_OUTPUT
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0008
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_717
    minimum_mean_gross_underlying_bp: float = 12.0


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clock_sha256(schedule: pd.DataFrame) -> str:
    content = schedule[list(support.CLOCK_COLUMNS)].to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("LVRT-R0 evaluator freeze is missing")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if _canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("LVRT-R0 evaluator freeze manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("LVRT-R0 evaluator was not frozen before outcomes")
    if payload.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("LVRT-R0 evaluator freeze source path changed")
    if payload.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("LVRT-R0 evaluator differs from its pre-outcome freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("LVRT-R0 evaluator freeze support commit changed")
    if payload.get("opened_windows") != []:
        raise ValueError("LVRT-R0 evaluator freeze already opened a window")
    if payload.get("returns_or_prices_loaded_during_freeze") is not False:
        raise ValueError("LVRT-R0 evaluator freeze loaded outcomes")
    if payload.get("sealed_windows") != [*WINDOWS, "2024", "2025", "2026_ytd"]:
        raise ValueError("LVRT-R0 evaluator sealed windows changed")
    if payload.get("mutable_parameters") != []:
        raise ValueError("LVRT-R0 evaluator freeze permits mutable parameters")
    if len(str(payload.get("evaluation_source_commit", ""))) != 40:
        raise ValueError("LVRT-R0 evaluator freeze commit is invalid")
    expected_policies = set(POLICY_NAMES)
    if set(payload.get("policy_clock_sha256", {})) != expected_policies:
        raise ValueError("LVRT-R0 evaluator freeze clock hash set changed")
    if set(payload.get("policy_clock_rows", {})) != expected_policies:
        raise ValueError("LVRT-R0 evaluator freeze clock row set changed")
    if payload["policy_clock_sha256"]["primary"] != EVENT_CLOCK_SHA256:
        raise ValueError("LVRT-R0 evaluator freeze primary clock changed")
    return payload


def _canonical_clock(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    missing = set(support.CLOCK_COLUMNS).difference(schedule.columns)
    if missing:
        raise ValueError(f"LVRT-R0 clock lacks columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for row in schedule[list(support.CLOCK_COLUMNS)].itertuples(index=False):
        rows.append(
            {
                "setup_position": int(row.setup_position),
                "signal_position": int(row.signal_position),
                "entry_position": int(row.entry_position),
                "exit_position": int(row.exit_position),
                "setup_date": str(row.setup_date),
                "signal_date": str(row.signal_date),
                "entry_date": str(row.entry_date),
                "exit_date": str(row.exit_date),
                "side": int(row.side),
                "branch": str(row.branch),
                "hold_bars": int(row.hold_bars),
            }
        )
    return rows


def _verify_policy_clock_hashes(
    schedules: dict[str, pd.DataFrame],
    *,
    expected_hashes: dict[str, str],
    expected_rows: dict[str, int],
) -> None:
    if set(schedules) != set(POLICY_NAMES):
        raise ValueError("LVRT-R0 replay policy set changed")
    for name, schedule in schedules.items():
        if _clock_sha256(schedule) != expected_hashes.get(name):
            raise ValueError(f"LVRT-R0 frozen control clock hash changed: {name}")
        if len(schedule) != expected_rows.get(name):
            raise ValueError(f"LVRT-R0 frozen control clock rows changed: {name}")


def verify_support_and_replay(
    *,
    expected_clock_hashes: dict[str, str] | None = None,
    expected_clock_rows: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    frozen_files = (
        (Path(support.__file__), SUPPORT_SOURCE_SHA256),
        (SUPPORT_DOCUMENT, SUPPORT_DOCUMENT_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
        (EVENT_CLOCK, EVENT_CLOCK_SHA256),
    )
    for path, expected in frozen_files:
        if _sha256(path) != expected:
            raise ValueError(f"frozen LVRT-R0 support dependency changed: {path}")
    frozen_result = json.loads(SUPPORT_RESULT.read_text())
    if frozen_result.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("LVRT-R0 support artifact opened outcomes")
    if frozen_result.get("support_decision") != "pass":
        raise ValueError("LVRT-R0 support gate did not pass")
    if frozen_result.get("support", {}).get("passes_support") is not True:
        raise ValueError("LVRT-R0 support row is not passing")
    if frozen_result.get("policy") != asdict(prereg.Policy()):
        raise ValueError("LVRT-R0 frozen support policy changed")

    frame, source = support.load_support_frame(prereg.Policy())
    signals, _ = support.classify_signals(frame, prereg.Policy())
    schedules = {
        name: support.nonoverlapping_schedule(signals[name], frame)
        for name in POLICY_NAMES
    }
    frozen_clock = pd.read_csv(EVENT_CLOCK)
    if _canonical_clock(schedules["primary"]) != _canonical_clock(frozen_clock):
        raise ValueError("LVRT-R0 primary event-clock replay differs from freeze")
    replay_bytes = schedules["primary"].to_csv(index=False, lineterminator="\n").encode(
        "utf-8"
    )
    if hashlib.sha256(replay_bytes).hexdigest() != EVENT_CLOCK_SHA256:
        raise ValueError("LVRT-R0 replayed clock hash differs from freeze")
    if source != frozen_result.get("source"):
        raise ValueError("LVRT-R0 support source replay differs from freeze")
    for name in POLICY_NAMES[1:]:
        frozen_count = frozen_result["controls"][name]["nonoverlap_count"]
        if len(schedules[name]) != frozen_count:
            raise ValueError(f"LVRT-R0 control clock changed: {name}")
    if (expected_clock_hashes is None) != (expected_clock_rows is None):
        raise ValueError("LVRT-R0 expected clock hashes and rows must be paired")
    if expected_clock_hashes is not None and expected_clock_rows is not None:
        _verify_policy_clock_hashes(
            schedules,
            expected_hashes=expected_clock_hashes,
            expected_rows=expected_clock_rows,
        )
    return frame, schedules, frozen_result


def load_execution_market(signal_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = json.loads(SUPPORT_RESULT.read_text())["source"]
    market_path = prereg.MARKET_PATH
    if _sha256(market_path) != source["market_sha256"]:
        raise ValueError("LVRT-R0 execution market differs from frozen hash")
    market = pd.read_csv(
        market_path,
        compression="gzip",
        usecols=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    if not market["date"].equals(signal_frame["date"]):
        raise ValueError("LVRT-R0 execution market is not aligned to signal frame")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("LVRT-R0 market has invalid prices")
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    closes = market["close"].to_numpy(float)
    if (
        (highs < np.maximum(opens, closes)).any()
        or (lows > np.minimum(opens, closes)).any()
        or (highs < lows).any()
    ):
        raise ValueError("LVRT-R0 market violates OHLC invariants")
    return market, {
        "market_sha256": _sha256(market_path),
        "market_rows": int(len(market)),
        "columns_loaded": ["date", "open", "high", "low", "close"],
        "first_date": str(market["date"].iloc[0]),
        "last_date": str(market["date"].iloc[-1]),
    }


def load_realized_funding() -> tuple[pd.DataFrame, dict[str, Any]]:
    if _sha256(FUNDING_MANIFEST) != FUNDING_MANIFEST_SHA256:
        raise ValueError("LVRT-R0 funding manifest differs from frozen hash")
    if _sha256(FUNDING_DATA) != FUNDING_DATA_SHA256:
        raise ValueError("LVRT-R0 funding data differs from frozen hash")
    manifest = json.loads(FUNDING_MANIFEST.read_text())
    if manifest.get("protocol", {}).get("luri_outcomes_opened") is not False:
        raise ValueError("LVRT-R0 funding source lacks unopened-source provenance")
    funding = pd.read_csv(
        FUNDING_DATA,
        usecols=["funding_time_ms", "funding_time_utc", "symbol", "funding_rate"],
        dtype={"symbol": str, "funding_rate": str},
    )
    if len(funding) != manifest["data"]["rows"]:
        raise ValueError("LVRT-R0 funding row count differs from manifest")
    funding["funding_time_ms"] = pd.to_numeric(
        funding["funding_time_ms"], errors="raise"
    ).astype(np.int64)
    utc = pd.to_datetime(funding["funding_time_utc"], utc=True, errors="raise").dt.tz_convert(None)
    epoch = pd.to_datetime(
        funding["funding_time_ms"], unit="ms", utc=True, errors="raise"
    ).dt.tz_convert(None)
    if not utc.equals(epoch):
        raise ValueError("LVRT-R0 funding timestamps disagree")
    if funding["funding_time_ms"].duplicated().any() or not funding[
        "funding_time_ms"
    ].is_monotonic_increasing:
        raise ValueError("LVRT-R0 funding timestamps are invalid")
    if not funding["symbol"].eq("BTCUSDT").all():
        raise ValueError("LVRT-R0 funding contains another symbol")
    rates = pd.to_numeric(funding["funding_rate"], errors="raise").to_numpy(float)
    if not np.isfinite(rates).all() or utc.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("LVRT-R0 funding values or interval are invalid")
    normalized = pd.DataFrame(
        {
            "funding_time_ms": funding["funding_time_ms"].to_numpy(np.int64),
            "funding_time": utc,
            "funding_rate": rates,
        }
    )
    return normalized, {
        "funding_manifest_sha256": _sha256(FUNDING_MANIFEST),
        "funding_data_sha256": _sha256(FUNDING_DATA),
        "funding_rows": int(len(normalized)),
        "columns_loaded": ["funding_time_ms", "funding_time_utc", "symbol", "funding_rate"],
        "first_funding_time": str(normalized["funding_time"].iloc[0]),
        "last_funding_time": str(normalized["funding_time"].iloc[-1]),
    }


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
        raise ValueError("trade returns and dates must have equal length")
    if not trade_returns:
        return {
            "p_value_one_sided": 1.0,
            "observed_mean_return": 0.0,
            "cluster_count": 0,
            "permutations": int(permutations),
            "seed": int(seed),
        }
    values = np.asarray(trade_returns, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("trade returns must be finite")
    dates = pd.to_datetime(pd.Series(entry_dates), utc=True, errors="raise")
    monday = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.floor("D")
    clusters = (
        pd.DataFrame({"week": monday, "return": values})
        .groupby("week", sort=True, observed=True)["return"]
        .sum()
        .to_numpy(float)
    )
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    while completed < permutations:
        batch = min(4_096, permutations - completed)
        signs = rng.integers(0, 2, size=(batch, len(clusters)), dtype=np.int8)
        signs = signs.astype(float) * 2.0 - 1.0
        permuted = signs.dot(clusters) / len(values)
        exceedances += int(np.count_nonzero(permuted >= observed))
        completed += batch
    return {
        "p_value_one_sided": float((1 + exceedances) / (permutations + 1)),
        "observed_mean_return": observed,
        "cluster_count": int(len(clusters)),
        "permutations": int(permutations),
        "seed": int(seed),
    }


def _trade_statistics(values: list[float]) -> dict[str, Any]:
    count = len(values)
    if not count:
        return {
            "n_trades": 0,
            "mean_trade_ret_pct": 0.0,
            "std_trade_ret_pct": 0.0,
            "t_stat_like": 0.0,
            "ci95_mean_trade_ret_pct": [0.0, 0.0],
        }
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    se = std / math.sqrt(count) if count else 0.0
    t_stat = mean / se if se > 0.0 else 0.0
    return {
        "n_trades": count,
        "mean_trade_ret_pct": mean * 100.0,
        "std_trade_ret_pct": std * 100.0,
        "t_stat_like": t_stat,
        "ci95_mean_trade_ret_pct": [
            (mean - 1.96 * se) * 100.0,
            (mean + 1.96 * se) * 100.0,
        ],
    }


def _slice_schedule(schedule: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    setup = pd.to_datetime(schedule["setup_date"], errors="raise")
    signal = pd.to_datetime(schedule["signal_date"], errors="raise")
    entry = pd.to_datetime(schedule["entry_date"], errors="raise")
    exit_ = pd.to_datetime(schedule["exit_date"], errors="raise")
    inside = (
        setup.ge(start_timestamp)
        & signal.ge(start_timestamp)
        & entry.ge(start_timestamp)
        & exit_.ge(start_timestamp)
        & setup.lt(end_timestamp)
        & signal.lt(end_timestamp)
        & entry.lt(end_timestamp)
        & exit_.lt(end_timestamp)
    )
    return schedule.loc[inside].reset_index(drop=True)


def simulate_schedule(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cost_notional_per_side: float,
    cfg: EvaluationConfig,
    compute_cluster: bool,
) -> dict[str, Any]:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("simulation start must precede end")
    if cfg.leverage <= 0.0:
        raise ValueError("leverage must be positive")
    per_side_cost = cost_notional_per_side * cfg.leverage
    if not 0.0 <= per_side_cost < 1.0:
        raise ValueError("per-side cost is invalid")
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    dates = market["date"]
    funding_times = funding["funding_time_ms"].to_numpy(np.int64)
    funding_rates = funding["funding_rate"].to_numpy(float)
    equity = 1.0
    peak = 1.0
    strict_mdd = 0.0
    previous_exit = -1
    trade_returns: list[float] = []
    gross_returns: list[float] = []
    entry_dates: list[str] = []
    sides: list[int] = []
    settlement_count = 0
    trades_with_funding = 0
    aggregate_funding_factor = 1.0

    for row in schedule.itertuples(index=False):
        setup_position = int(row.setup_position)
        signal_position = int(row.signal_position)
        entry_position = int(row.entry_position)
        exit_position = int(row.exit_position)
        side = int(row.side)
        if side not in (-1, 1):
            raise ValueError("LVRT-R0 side must be long or short")
        if not setup_position <= signal_position < entry_position < exit_position:
            raise ValueError("LVRT-R0 scheduled positions are invalid")
        if entry_position != signal_position + 1:
            raise ValueError("LVRT-R0 entry is not next-open")
        if exit_position != entry_position + prereg.Policy().hold_bars:
            raise ValueError("LVRT-R0 exit differs from frozen hold")
        if int(row.hold_bars) != prereg.Policy().hold_bars:
            raise ValueError("LVRT-R0 hold_bars differs from freeze")
        if entry_position < previous_exit:
            raise ValueError("LVRT-R0 schedules overlap")
        if exit_position >= len(market):
            raise ValueError("LVRT-R0 exit exceeds market frame")
        if not (
            start_timestamp <= dates.iloc[setup_position] < end_timestamp
            and start_timestamp <= dates.iloc[signal_position] < end_timestamp
            and start_timestamp <= dates.iloc[entry_position] < end_timestamp
            and start_timestamp <= dates.iloc[exit_position] < end_timestamp
        ):
            raise ValueError("LVRT-R0 trade crosses simulation split")

        entry_price = float(opens[entry_position])
        exit_price = float(opens[exit_position])
        held_high = float(np.max(highs[entry_position:exit_position]))
        held_low = float(np.min(lows[entry_position:exit_position]))
        if min(entry_price, exit_price, held_high, held_low) <= 0.0:
            raise ValueError("LVRT-R0 scheduled trade has invalid price")

        entry_ms = int(pd.Timestamp(dates.iloc[entry_position]).value // 1_000_000)
        exit_ms = int(pd.Timestamp(dates.iloc[exit_position]).value // 1_000_000)
        # Half-open funding interval prevents a settlement at a shared
        # exit/re-entry timestamp from being charged to both trades.
        funding_left = int(np.searchsorted(funding_times, entry_ms, side="left"))
        funding_right = int(np.searchsorted(funding_times, exit_ms, side="left"))
        rates = funding_rates[funding_left:funding_right]
        factors = 1.0 - cfg.leverage * side * rates
        if not np.isfinite(factors).all() or (factors <= 0.0).any():
            raise ValueError("LVRT-R0 scheduled funding factor is invalid")
        funding_factor = float(np.prod(factors, dtype=float))
        credit_factor = float(np.prod(np.maximum(factors, 1.0), dtype=float))

        entry_equity = equity
        # Refresh the global/pre-entry HWM before charging a new entry cost.
        peak = max(peak, equity)
        equity *= 1.0 - per_side_cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        post_entry_equity = equity

        favorable_price = held_high if side > 0 else held_low
        adverse_price = held_low if side > 0 else held_high
        favorable_factor = max(
            0.0,
            1.0 + cfg.leverage * side * (favorable_price / entry_price - 1.0),
        )
        adverse_factor = max(
            0.0,
            1.0 + cfg.leverage * side * (adverse_price / entry_price - 1.0),
        )
        favorable_equity = post_entry_equity * credit_factor * favorable_factor
        intratrade_peak = max(peak, favorable_equity)
        # A position could be liquidated at the adverse extreme; apply the
        # same frozen exit cost to that hypothetical liquidation path.
        adverse_liquidation_equity = (
            post_entry_equity
            * funding_factor
            * adverse_factor
            * (1.0 - per_side_cost)
        )
        strict_mdd = max(
            strict_mdd,
            1.0 - max(0.0, adverse_liquidation_equity) / intratrade_peak,
        )
        peak = intratrade_peak

        gross_return = side * (exit_price / entry_price - 1.0)
        exit_price_factor = max(0.0, 1.0 + cfg.leverage * gross_return)
        equity = (
            post_entry_equity
            * funding_factor
            * exit_price_factor
            * (1.0 - per_side_cost)
        )
        strict_mdd = max(strict_mdd, 1.0 - max(0.0, equity) / peak)
        peak = max(peak, equity)

        trade_returns.append(equity / entry_equity - 1.0)
        gross_returns.append(gross_return)
        entry_dates.append(str(dates.iloc[entry_position]))
        sides.append(side)
        settlement_count += int(len(rates))
        trades_with_funding += int(len(rates) > 0)
        aggregate_funding_factor *= funding_factor
        previous_exit = exit_position

    years = (end_timestamp - start_timestamp).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    strict_mdd_pct = strict_mdd * 100.0
    cluster = (
        weekly_cluster_sign_flip(
            trade_returns,
            entry_dates,
            permutations=cfg.cluster_permutations,
            seed=cfg.cluster_seed,
        )
        if compute_cluster
        else None
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
        "wall_clock_years": float(years),
        "mean_gross_underlying_move_bp": (
            float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0
        ),
        "funding_settlement_count": int(settlement_count),
        "trades_with_funding": int(trades_with_funding),
        "aggregate_funding_factor": float(aggregate_funding_factor),
        "execution_cost_notional_per_side_bp": float(cost_notional_per_side * 10_000.0),
        "trade_statistics": _trade_statistics(trade_returns),
        "weekly_cluster_sign_flip": cluster,
    }


def _evaluate_policy_windows(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for name, (start, end) in WINDOWS.items():
        sliced = _slice_schedule(schedule, start=start, end=end)
        cluster = name in {"train", "select2023"}
        windows[name] = {
            "base": simulate_schedule(
                market,
                funding,
                sliced,
                start=start,
                end=end,
                cost_notional_per_side=cfg.base_cost_notional_per_side,
                cfg=cfg,
                compute_cluster=cluster,
            ),
            "stress_8bp": simulate_schedule(
                market,
                funding,
                sliced,
                start=start,
                end=end,
                cost_notional_per_side=cfg.stress_cost_notional_per_side,
                cfg=cfg,
                compute_cluster=False,
            ),
        }
    return windows


def _policy_gate_failures(windows: dict[str, Any], cfg: EvaluationConfig) -> list[str]:
    failures: list[str] = []
    for name in ("train", "select2023"):
        base = windows[name]["base"]
        stress = windows[name]["stress_8bp"]
        if base["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if base["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if base["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}: strict MDD above 15%")
        cluster = base["weekly_cluster_sign_flip"]
        if cluster is None or cluster["p_value_one_sided"] > 0.10:
            failures.append(f"{name}: weekly-cluster p-value above 0.10")
        if base["mean_gross_underlying_move_bp"] <= cfg.minimum_mean_gross_underlying_bp:
            failures.append(f"{name}: mean gross move not above 12 bp")
        if stress["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: 8bp stress non-positive")
    if windows["train"]["base"]["trade_count"] < 120:
        failures.append("train: fewer than 120 trades")
    if windows["select2023"]["base"]["trade_count"] < 80:
        failures.append("select2023: fewer than 80 trades")
    for name in ("select2023_h1", "select2023_h2"):
        base = windows[name]["base"]
        if base["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if base["trade_count"] < 20:
            failures.append(f"{name}: fewer than 20 trades")
    return failures


def qualification(
    policy_windows: dict[str, dict[str, Any]], cfg: EvaluationConfig
) -> dict[str, Any]:
    primary_failures = _policy_gate_failures(policy_windows["primary"], cfg)
    placebo_failures = {
        name: _policy_gate_failures(policy_windows[name], cfg)
        for name in REJECTION_PLACEBOS
    }
    passing_placebos = [name for name, failures in placebo_failures.items() if not failures]
    failures = list(primary_failures)
    for name in passing_placebos:
        failures.append(f"placebo independently passed all primary gates: {name}")
    return {
        "qualifies": not failures,
        "failures": failures,
        "primary_gate_failures": primary_failures,
        "rejection_placebo_gate_failures": placebo_failures,
        "passing_rejection_placebos": passing_placebos,
    }


def run_evaluation(cfg: EvaluationConfig | None = None) -> dict[str, Any]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    if frozen_cfg != EvaluationConfig(output=frozen_cfg.output):
        raise ValueError("LVRT-R0 evaluation parameters are frozen")
    evaluator_freeze = verify_evaluation_freeze()
    signal_frame, schedules, support_result = verify_support_and_replay(
        expected_clock_hashes=evaluator_freeze["policy_clock_sha256"],
        expected_clock_rows=evaluator_freeze["policy_clock_rows"],
    )
    market, market_source = load_execution_market(signal_frame)
    funding, funding_source = load_realized_funding()
    windows = {
        name: _evaluate_policy_windows(market, funding, schedule, frozen_cfg)
        for name, schedule in schedules.items()
    }
    verdict = qualification(windows, frozen_cfg)
    return {
        "protocol": {
            "name": "LVRT-R0 frozen pre-2024 selection evaluation",
            "support_commit": SUPPORT_COMMIT,
            "evaluation_source_commit": evaluator_freeze["evaluation_source_commit"],
            "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
            "evaluation_freeze_sha256": _sha256(EVALUATION_FREEZE),
            "outcomes_opened": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["2024", "2025", "2026_ytd"],
            "parameters_mutable": False,
            "primary_clock_replayed_from_freeze": True,
            "control_clocks_reserved_before_outcomes": True,
            "funding_interval": "entry_time <= funding_time < exit_time",
            "entry": "next 5m open after completed confirmation",
            "exit": "scheduled open after 12 held 5m bars",
            "strict_mdd": (
                "global/pre-entry HWM; favorable then adverse OHLC; funding credits "
                "raise HWM, all funding affects adverse; hypothetical liquidation cost"
            ),
            "cagr": "full wall-clock split including idle cash",
        },
        "evaluation_config": asdict(frozen_cfg),
        "source": {
            "support": support_result["source"],
            "execution": market_source,
            "funding": funding_source,
        },
        "global_clock_counts": {
            name: int(len(schedule)) for name, schedule in schedules.items()
        },
        "windows": windows,
        "qualification": verdict,
        "selection": {
            "selected_alpha": "LVRT-R0" if verdict["qualifies"] else None,
            "rejected": not verdict["qualifies"],
            "reason": (
                "passed every frozen pre-2024 gate"
                if verdict["qualifies"]
                else "failed at least one frozen pre-2024 gate"
            ),
        },
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "mean_gross_underlying_move_bp",
            "trade_count",
            "funding_settlement_count",
        )
    }


def main() -> None:
    result = run_evaluation()
    output = Path(DEFAULT_OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "qualification": result["qualification"],
                "primary": {
                    name: {
                        cost: _headline(metrics)
                        for cost, metrics in result["windows"]["primary"][name].items()
                    }
                    for name in WINDOWS
                },
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
