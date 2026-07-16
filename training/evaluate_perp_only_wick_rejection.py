"""One-shot strict pre-2024 evaluator for frozen POWR-12.

This module cannot run execution simulation or load funding until its own source
has been frozen by ``freeze_perp_only_wick_rejection_evaluator``.  Clock replay
does load the same completed-bar OHLC used by the frozen POWR signal, but it does
not calculate any post-signal return.  The evaluator never opens 2024+ data.
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

from training import build_perp_only_wick_rejection_support as support
from training import preregister_perp_only_wick_rejection as prereg


SUPPORT_COMMIT = "4202671697a3a31117d7e2caabbfa536295d5837"
SUPPORT_SOURCE_SHA256 = (
    "4adf1812ea75170d5360f3ab40ecec8403682401659451515b0b3c83f0e8f583"
)
SUPPORT_DOCUMENT = Path(
    "docs/perp-only-wick-rejection-support-freeze-2026-07-17.md"
)
SUPPORT_DOCUMENT_SHA256 = (
    "52c9ec857240c202f22afbd0f3cb67b4d5fbbf3972e6ae2fcda4edd2e422239a"
)
SUPPORT_RESULT = Path(support.DEFAULT_OUTPUT)
SUPPORT_RESULT_SHA256 = (
    "6e753d4dbd525f5c6e45882d7df6b1f5fe6e614727f939f7853c7c3c857d347d"
)
EVENT_CLOCK = Path(support.DEFAULT_CLOCK)
EVENT_CLOCK_SHA256 = (
    "7ecd567bf182fd7f92a8a1583b8f82c409ea5530d2e0eef25174880d52502619"
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
EVALUATION_SOURCE = Path("training/evaluate_perp_only_wick_rejection.py")
EVALUATION_FREEZE = Path(
    "results/perp_only_wick_rejection_evaluator_freeze_2026-07-17.json"
)
DEFAULT_OUTPUT = "results/perp_only_wick_rejection_selection_2026-07-17.json"
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
POLICY_NAMES = support.POLICY_NAMES
MECHANISM_REJECTION_CONTROLS = (
    "spot_only_wick",
    "common_wick",
    "basis_free_perp_wick",
    "stale_spot_1h",
    "stale_spot_1d",
)
DELAYED_ENTRY_CONTROL = "one_bar_delayed_entry"


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
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal_result(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": _canonical_hash(core)}


def validate_result_hash(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if _canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("POWR-12 result manifest hash mismatch")


def _clock_sha256(schedule: pd.DataFrame) -> str:
    content = schedule[list(support.CLOCK_COLUMNS)].to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("POWR-12 evaluator freeze is missing")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if _canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("POWR-12 evaluator freeze manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("POWR-12 evaluator was not frozen before outcomes")
    if payload.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("POWR-12 evaluator freeze source path changed")
    if payload.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("POWR-12 evaluator differs from its pre-outcome freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("POWR-12 evaluator freeze support commit changed")
    if payload.get("opened_windows") != []:
        raise ValueError("POWR-12 evaluator freeze already opened a window")
    if payload.get("signal_feature_ohlc_loaded_during_freeze") is not True:
        raise ValueError("POWR-12 evaluator freeze source disclosure changed")
    if payload.get("post_signal_returns_computed_during_freeze") is not False:
        raise ValueError("POWR-12 evaluator freeze computed post-signal returns")
    if payload.get("funding_loaded_during_freeze") is not False:
        raise ValueError("POWR-12 evaluator freeze loaded funding")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise ValueError("POWR-12 evaluator freeze ran execution simulation")
    if payload.get("sealed_windows") != [*WINDOWS, "2024", "2025", "2026_ytd"]:
        raise ValueError("POWR-12 evaluator sealed windows changed")
    if payload.get("mutable_parameters") != []:
        raise ValueError("POWR-12 evaluator freeze permits mutable parameters")
    if len(str(payload.get("evaluation_source_commit", ""))) != 40:
        raise ValueError("POWR-12 evaluator freeze commit is invalid")
    expected_policies = set(POLICY_NAMES)
    if set(payload.get("policy_clock_sha256", {})) != expected_policies:
        raise ValueError("POWR-12 evaluator freeze clock hash set changed")
    if set(payload.get("policy_clock_rows", {})) != expected_policies:
        raise ValueError("POWR-12 evaluator freeze clock row set changed")
    if payload["policy_clock_sha256"]["primary"] != EVENT_CLOCK_SHA256:
        raise ValueError("POWR-12 evaluator freeze primary clock changed")
    return payload


def _canonical_clock(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    missing = set(support.CLOCK_COLUMNS).difference(schedule.columns)
    if missing:
        raise ValueError(f"POWR-12 clock lacks columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for row in schedule[list(support.CLOCK_COLUMNS)].itertuples(index=False):
        rows.append(
            {
                "signal_position": int(row.signal_position),
                "entry_position": int(row.entry_position),
                "exit_position": int(row.exit_position),
                "signal_date": str(row.signal_date),
                "entry_date": str(row.entry_date),
                "exit_date": str(row.exit_date),
                "side": int(row.side),
                "branch": str(row.branch),
                "entry_delay_bars": int(row.entry_delay_bars),
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
        raise ValueError("POWR-12 replay policy set changed")
    for name, schedule in schedules.items():
        if _clock_sha256(schedule) != expected_hashes.get(name):
            raise ValueError(f"POWR-12 frozen control clock hash changed: {name}")
        if len(schedule) != expected_rows.get(name):
            raise ValueError(f"POWR-12 frozen control clock rows changed: {name}")


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
            raise ValueError(f"frozen POWR-12 support dependency changed: {path}")
    frozen_result = json.loads(SUPPORT_RESULT.read_text())
    if frozen_result.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("POWR-12 support artifact opened outcomes")
    if frozen_result.get("support_decision") != "pass":
        raise ValueError("POWR-12 support gate did not pass")
    if frozen_result.get("support", {}).get("passes_support") is not True:
        raise ValueError("POWR-12 support row is not passing")
    if frozen_result.get("policy") != asdict(prereg.Policy()):
        raise ValueError("POWR-12 frozen support policy changed")

    frame, source = support.load_support_frame(prereg.Policy())
    signals, _ = support.classify_signals(frame, prereg.Policy())
    schedules = {
        name: support.nonoverlapping_schedule(signals[name], frame)
        for name in POLICY_NAMES
    }
    frozen_clock = pd.read_csv(EVENT_CLOCK)
    if _canonical_clock(schedules["primary"]) != _canonical_clock(frozen_clock):
        raise ValueError("POWR-12 primary event-clock replay differs from freeze")
    replay_bytes = schedules["primary"].to_csv(index=False, lineterminator="\n").encode(
        "utf-8"
    )
    if hashlib.sha256(replay_bytes).hexdigest() != EVENT_CLOCK_SHA256:
        raise ValueError("POWR-12 replayed clock hash differs from freeze")
    if source != frozen_result.get("source"):
        raise ValueError("POWR-12 support source replay differs from freeze")
    for name in POLICY_NAMES[1:]:
        frozen_count = frozen_result["controls"][name]["nonoverlap_count"]
        if len(schedules[name]) != frozen_count:
            raise ValueError(f"POWR-12 control clock changed: {name}")
    if (expected_clock_hashes is None) != (expected_clock_rows is None):
        raise ValueError("POWR-12 expected clock hashes and rows must be paired")
    if expected_clock_hashes is not None and expected_clock_rows is not None:
        _verify_policy_clock_hashes(
            schedules,
            expected_hashes=expected_clock_hashes,
            expected_rows=expected_clock_rows,
        )
    return frame, schedules, frozen_result


def load_execution_market(signal_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    verify_evaluation_freeze()
    source = json.loads(SUPPORT_RESULT.read_text())["source"]
    if source.get("perp_sha256") != _sha256(prereg.PERP_SOURCE):
        raise ValueError("POWR-12 frozen execution source changed")
    if not signal_frame["perp_complete"].astype(bool).all():
        raise ValueError("POWR-12 execution market contains an incomplete Perp bar")
    market = signal_frame[
        ["date", "perp_open", "perp_high", "perp_low", "perp_close"]
    ].rename(
        columns={
            "perp_open": "open",
            "perp_high": "high",
            "perp_low": "low",
            "perp_close": "close",
        }
    )
    market = market.reset_index(drop=True)
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("POWR-12 market has invalid prices")
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    closes = market["close"].to_numpy(float)
    if (
        (highs < np.maximum(opens, closes)).any()
        or (lows > np.minimum(opens, closes)).any()
        or (highs < lows).any()
    ):
        raise ValueError("POWR-12 market violates OHLC invariants")
    return market, {
        "perp_1m_source_sha256": source["perp_sha256"],
        "aggregation": "frozen causal 1m-to-5m OHLC from support replay",
        "market_rows": int(len(market)),
        "columns_loaded": ["date", "open", "high", "low", "close"],
        "first_date": str(market["date"].iloc[0]),
        "last_date": str(market["date"].iloc[-1]),
    }


def load_realized_funding() -> tuple[pd.DataFrame, dict[str, Any]]:
    verify_evaluation_freeze()
    if _sha256(FUNDING_MANIFEST) != FUNDING_MANIFEST_SHA256:
        raise ValueError("POWR-12 funding manifest differs from frozen hash")
    if _sha256(FUNDING_DATA) != FUNDING_DATA_SHA256:
        raise ValueError("POWR-12 funding data differs from frozen hash")
    manifest = json.loads(FUNDING_MANIFEST.read_text())
    if manifest.get("protocol", {}).get("luri_outcomes_opened") is not False:
        raise ValueError("POWR-12 funding source lacks unopened-source provenance")
    if manifest.get("protocol", {}).get("stage") != "pre_outcome_funding_source_freeze":
        raise ValueError("POWR-12 funding source stage changed")
    if manifest.get("data", {}).get("sha256") != FUNDING_DATA_SHA256:
        raise ValueError("POWR-12 funding manifest data hash differs")
    funding = pd.read_csv(
        FUNDING_DATA,
        usecols=[
            "funding_time_ms",
            "funding_time_utc",
            "symbol",
            "funding_rate",
        ],
        dtype={"symbol": str, "funding_rate": str},
    )
    if len(funding) != manifest["data"]["rows"]:
        raise ValueError("POWR-12 funding row count differs from manifest")
    funding["funding_time_ms"] = pd.to_numeric(
        funding["funding_time_ms"], errors="raise"
    ).astype(np.int64)
    utc = pd.to_datetime(funding["funding_time_utc"], utc=True, errors="raise").dt.tz_convert(None)
    epoch = pd.to_datetime(
        funding["funding_time_ms"], unit="ms", utc=True, errors="raise"
    ).dt.tz_convert(None)
    if not utc.equals(epoch):
        raise ValueError("POWR-12 funding timestamps disagree")
    if funding["funding_time_ms"].duplicated().any() or not funding[
        "funding_time_ms"
    ].is_monotonic_increasing:
        raise ValueError("POWR-12 funding timestamps are invalid")
    if not funding["symbol"].eq("BTCUSDT").all():
        raise ValueError("POWR-12 funding contains another symbol")
    rates = pd.to_numeric(funding["funding_rate"], errors="raise").to_numpy(float)
    if not np.isfinite(rates).all() or utc.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("POWR-12 funding values or interval are invalid")
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
        "columns_loaded": [
            "funding_time_ms",
            "funding_time_utc",
            "symbol",
            "funding_rate",
        ],
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
    signal = pd.to_datetime(schedule["signal_date"], errors="raise")
    entry = pd.to_datetime(schedule["entry_date"], errors="raise")
    exit_ = pd.to_datetime(schedule["exit_date"], errors="raise")
    inside = (
        signal.ge(start_timestamp)
        & entry.ge(start_timestamp)
        & exit_.ge(start_timestamp)
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
    verify_evaluation_freeze()
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
        signal_position = int(row.signal_position)
        entry_position = int(row.entry_position)
        exit_position = int(row.exit_position)
        entry_delay_bars = int(row.entry_delay_bars)
        side = int(row.side)
        if side not in (-1, 1):
            raise ValueError("POWR-12 side must be long or short")
        if not 0 <= signal_position < entry_position < exit_position:
            raise ValueError("POWR-12 scheduled positions are invalid")
        if entry_delay_bars not in {
            prereg.Policy().entry_delay_bars,
            prereg.Policy().entry_delay_bars + 1,
        }:
            raise ValueError("POWR-12 entry delay differs from frozen controls")
        if entry_position != signal_position + entry_delay_bars:
            raise ValueError("POWR-12 entry position differs from frozen delay")
        if exit_position != entry_position + prereg.Policy().hold_bars:
            raise ValueError("POWR-12 exit differs from frozen hold")
        if int(row.hold_bars) != prereg.Policy().hold_bars:
            raise ValueError("POWR-12 hold_bars differs from freeze")
        if entry_position < previous_exit:
            raise ValueError("POWR-12 schedules overlap")
        if exit_position >= len(market):
            raise ValueError("POWR-12 exit exceeds market frame")
        for label, position in (
            ("signal", signal_position),
            ("entry", entry_position),
            ("exit", exit_position),
        ):
            if pd.Timestamp(getattr(row, f"{label}_date")) != dates.iloc[position]:
                raise ValueError(f"POWR-12 {label} timestamp differs from market")
        if not (
            start_timestamp <= dates.iloc[signal_position] < end_timestamp
            and start_timestamp <= dates.iloc[entry_position] < end_timestamp
            and start_timestamp <= dates.iloc[exit_position] < end_timestamp
        ):
            raise ValueError("POWR-12 trade crosses simulation split")

        entry_price = float(opens[entry_position])
        exit_price = float(opens[exit_position])
        held_high = float(np.max(highs[entry_position:exit_position]))
        held_low = float(np.min(lows[entry_position:exit_position]))
        if min(entry_price, exit_price, held_high, held_low) <= 0.0:
            raise ValueError("POWR-12 scheduled trade has invalid price")

        entry_ms = int(pd.Timestamp(dates.iloc[entry_position]).value // 1_000_000)
        exit_ms = int(pd.Timestamp(dates.iloc[exit_position]).value // 1_000_000)
        # Half-open funding interval prevents a settlement at a shared
        # exit/re-entry timestamp from being charged to both trades.
        funding_left = int(np.searchsorted(funding_times, entry_ms, side="left"))
        funding_right = int(np.searchsorted(funding_times, exit_ms, side="left"))
        rates = funding_rates[funding_left:funding_right]
        # The frozen source guarantees realized settlement rates but its
        # historical mark-price column is mostly absent.  Apply each realized
        # rate to the fixed entry notional, as preregistered, rather than
        # silently imputing a settlement mark from future price data.
        funding_contributions = -cfg.leverage * side * rates
        funding_return = float(np.sum(funding_contributions, dtype=float))
        funding_factor = 1.0 + funding_return
        funding_credit = float(
            np.sum(np.maximum(funding_contributions, 0.0), dtype=float)
        )
        if not np.isfinite(funding_factor) or funding_factor <= 0.0:
            raise ValueError("POWR-12 scheduled funding factor is invalid")

        entry_equity = equity
        # Refresh the global/pre-entry HWM before charging a new entry cost.
        peak = max(peak, equity)
        equity = entry_equity * (1.0 - per_side_cost)
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)

        favorable_price = held_high if side > 0 else held_low
        adverse_price = held_low if side > 0 else held_high
        favorable_equity = entry_equity * max(
            0.0,
            1.0
            - per_side_cost
            + cfg.leverage * side * (favorable_price / entry_price - 1.0)
            + funding_credit,
        )
        intratrade_peak = max(peak, favorable_equity)
        # Fixed quantity implies that both funding and liquidation commission
        # scale with the settlement/adverse mark relative to entry price.
        adverse_liquidation_factor = max(
            0.0,
            1.0
            - per_side_cost
            + cfg.leverage * side * (adverse_price / entry_price - 1.0)
            + funding_return
            - per_side_cost * (adverse_price / entry_price),
        )
        adverse_liquidation_equity = entry_equity * adverse_liquidation_factor
        strict_mdd = max(
            strict_mdd,
            1.0 - max(0.0, adverse_liquidation_equity) / intratrade_peak,
        )
        peak = intratrade_peak

        gross_return = side * (exit_price / entry_price - 1.0)
        exit_equity_factor = max(
            0.0,
            1.0
            - per_side_cost
            + cfg.leverage * gross_return
            + funding_return
            - per_side_cost * (exit_price / entry_price),
        )
        equity = entry_equity * exit_equity_factor
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


def _performance_gate_failures(
    windows: dict[str, Any], cfg: EvaluationConfig
) -> list[str]:
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
        # The frozen preregistration document says the move must be strictly
        # greater than 12 bp; equality therefore fails closed.
        if base["mean_gross_underlying_move_bp"] <= cfg.minimum_mean_gross_underlying_bp:
            failures.append(f"{name}: mean gross move not above 12 bp")
        if stress["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: 8bp stress non-positive")
    for name in ("select2023_h1", "select2023_h2"):
        base = windows[name]["base"]
        if base["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
    return failures


def qualification(
    policy_windows: dict[str, dict[str, Any]], cfg: EvaluationConfig
) -> dict[str, Any]:
    primary_performance_failures = _performance_gate_failures(
        policy_windows["primary"], cfg
    )
    delayed_entry_failures = [
        f"{name}: one-bar-delayed entry non-positive absolute return"
        for name in ("train", "select2023")
        if policy_windows[DELAYED_ENTRY_CONTROL][name]["base"][
            "absolute_return_pct"
        ]
        <= 0.0
    ]
    mechanism_control_failures = {
        name: _performance_gate_failures(policy_windows[name], cfg)
        for name in MECHANISM_REJECTION_CONTROLS
    }
    passing_controls = [
        name for name, failures in mechanism_control_failures.items() if not failures
    ]
    failures = [*primary_performance_failures, *delayed_entry_failures]
    for name in passing_controls:
        failures.append(
            f"mechanism control independently passed all primary gates: {name}"
        )
    return {
        "qualifies": not failures,
        "scope": "pre-orthogonality performance and mechanism gates",
        "final_promotion_allowed": False,
        "failures": failures,
        "primary_performance_gate_failures": primary_performance_failures,
        "delayed_entry_gate_failures": delayed_entry_failures,
        "mechanism_control_gate_failures": mechanism_control_failures,
        "passing_mechanism_controls": passing_controls,
        "direction_flip_is_diagnostic_only": True,
    }


def _selection_decision(verdict: dict[str, Any]) -> dict[str, Any]:
    if verdict["qualifies"]:
        return {
            "selected_alpha": None,
            "performance_candidate": "POWR-12",
            "rejected": False,
            "status": "pending_preregistered_orthogonality_and_portfolio_gates",
            "orthogonality_evaluated": False,
            "promotion_ready": False,
            "reason": (
                "pre-2024 performance passed; final selection remains forbidden "
                "until frozen orthogonality and marginal-portfolio gates pass"
            ),
        }
    return {
        "selected_alpha": None,
        "performance_candidate": None,
        "rejected": True,
        "status": "rejected_before_orthogonality",
        "orthogonality_evaluated": False,
        "promotion_ready": False,
        "reason": "failed at least one frozen pre-orthogonality gate",
    }


def run_evaluation(cfg: EvaluationConfig | None = None) -> dict[str, Any]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    if frozen_cfg != EvaluationConfig():
        raise ValueError("POWR-12 evaluation parameters are frozen")
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
    core = {
        "protocol": {
            "name": "POWR-12 frozen pre-2024 selection evaluation",
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
            "funding_notional_model": (
                "realized funding rate applied to fixed entry notional; missing "
                "historical settlement marks are never imputed"
            ),
            "entry": (
                "signal+3 open (t+15m) after completed joint latency bucket; "
                "delayed control enters at signal+4"
            ),
            "exit": "scheduled open after 12 held 5m bars",
            "strict_mdd": (
                "global/pre-entry HWM; favorable then adverse OHLC; funding credits "
                "raise HWM, all funding affects adverse; hypothetical liquidation cost"
            ),
            "cagr": "full wall-clock split including idle cash",
            "direction_flip": "diagnostic-only; never repairs or rejects primary",
            "orthogonality_after_performance": (
                "not evaluated in this stage; final selection is fail-closed until "
                "entry/position/PnL overlap and marginal portfolio gates pass"
            ),
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
        "selection": _selection_decision(verdict),
    }
    return _seal_result(core)


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
    output = Path(DEFAULT_OUTPUT)
    if output.exists():
        raise RuntimeError("POWR-12 outcome result already exists")
    result = run_evaluation()
    validate_result_hash(result)
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
