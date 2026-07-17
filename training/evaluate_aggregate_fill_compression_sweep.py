"""Strict staged evaluator for frozen AFCS-144.

Stage 1 physically stops parsing execution OHLC and funding settlement marks
before 2023.  Stage 2 is unavailable unless the immutable stage-1 gate passes.
The evaluator uses a fixed-quantity linear USD-M ledger, exact funding rates
with the frozen settlement-mark proxy, full-clock CAGR, and global
favorable-before-adverse held-path strict MDD.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_aggregate_fill_compression_sweep_support as support
from training import preregister_aggregate_fill_compression_sweep as prereg


SUPPORT_COMMIT = "15bc72ac7aa2eba3ad2f2f9f9f2112058a3432e1"
SUPPORT_SOURCE_SHA256 = (
    "e9f13d16efa837c7cf467de18a766bd6e40c7d4752b19fe374964d5a88bdea1c"
)
SUPPORT_DOCUMENT = Path(
    "docs/aggregate-fill-compression-sweep-support-freeze-2026-07-17.md"
)
SUPPORT_DOCUMENT_SHA256 = (
    "dfdc96e4ee0250e68dc279147d820a796ac48b767ed54f0598a5452ab23aa89d"
)
SUPPORT_RESULT = Path(support.DEFAULT_OUTPUT)
SUPPORT_RESULT_SHA256 = (
    "48bfc85e7dc8fe18cd0d961928097f2741e4c8272ee89eb305474b28526fa9ab"
)
EVENT_CLOCK = Path(support.DEFAULT_CLOCK)
EVENT_CLOCK_SHA256 = (
    "bf1611554604c1930ba2212e674ea434f7c9793377b3f33ef531b3b4e0381688"
)
FUNDING_DATA = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
FUNDING_DATA_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)
EVALUATION_SOURCE = Path("training/evaluate_aggregate_fill_compression_sweep.py")
EVALUATION_FREEZE = Path(
    "results/aggregate_fill_compression_sweep_evaluator_freeze_2026-07-17.json"
)
STAGE1_OUTPUT = Path(
    "results/aggregate_fill_compression_sweep_stage1_2020_2022_2026-07-17.json"
)
STAGE2_OUTPUT = Path(
    "results/aggregate_fill_compression_sweep_stage2_2023_2026-07-17.json"
)
STAGE1_START = pd.Timestamp("2020-01-01")
STAGE1_END = pd.Timestamp("2023-01-01")
STAGE2_START = pd.Timestamp("2023-01-01")
STAGE2_END = pd.Timestamp("2024-01-01")
COMPONENT_CONTROLS = ("no_compression", "no_coherence", "no_aligned_response")
REJECTION_PLACEBOS = ("one_day_shifted_clock", "random_side")


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_717
    minimum_mean_gross_underlying_bp: float = 20.0


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": _canonical_hash(core)}


def _clock_sha256(schedule: pd.DataFrame) -> str:
    content = schedule[list(support.CLOCK_COLUMNS)].to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _canonical_clock(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in schedule[list(support.CLOCK_COLUMNS)].itertuples(index=False):
        rows.append(
            {
                "origin_position": int(row.origin_position),
                "signal_position": int(row.signal_position),
                "entry_position": int(row.entry_position),
                "exit_position": int(row.exit_position),
                "origin_date": str(row.origin_date),
                "signal_date": str(row.signal_date),
                "entry_date": str(row.entry_date),
                "exit_date": str(row.exit_date),
                "side": int(row.side),
                "branch": str(row.branch),
                "delay_bars": int(row.delay_bars),
                "hold_bars": int(row.hold_bars),
            }
        )
    return rows


def verify_support_and_replay() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    for path, expected in (
        (Path(support.__file__), SUPPORT_SOURCE_SHA256),
        (SUPPORT_DOCUMENT, SUPPORT_DOCUMENT_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
        (EVENT_CLOCK, EVENT_CLOCK_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen AFCS-144 support dependency changed: {path}")
    frozen = json.loads(SUPPORT_RESULT.read_text())
    if frozen.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("AFCS-144 support artifact opened outcomes")
    if frozen.get("support_decision") != "pass":
        raise ValueError("AFCS-144 support did not pass")
    if frozen.get("policy") != asdict(prereg.Policy()):
        raise ValueError("AFCS-144 support policy changed")

    frame, source = support.load_support_frame(prereg.Policy())
    signals, _ = support.classify_signals(frame, prereg.Policy())
    schedules = {
        name: support.nonoverlapping_schedule(signals[name], frame)
        for name in support.POLICY_NAMES
    }
    frozen_clock = pd.read_csv(EVENT_CLOCK)
    if _canonical_clock(schedules["primary"]) != _canonical_clock(frozen_clock):
        raise ValueError("AFCS-144 primary clock replay differs from freeze")
    if _clock_sha256(schedules["primary"]) != EVENT_CLOCK_SHA256:
        raise ValueError("AFCS-144 primary clock hash differs from freeze")
    if source != frozen.get("source"):
        raise ValueError("AFCS-144 support source replay differs from freeze")
    for name, schedule in schedules.items():
        if name == "primary":
            continue
        expected = frozen["controls"][name]
        if len(schedule) != expected["nonoverlap_count"]:
            raise ValueError(f"AFCS-144 control row count changed: {name}")
        if _clock_sha256(schedule) != expected["clock_sha256"]:
            raise ValueError(f"AFCS-144 control clock changed: {name}")
    return frame, schedules, frozen


def build_freeze_manifest(evaluation_source_commit: str) -> dict[str, Any]:
    if len(evaluation_source_commit) != 40:
        raise ValueError("AFCS-144 evaluator commit must be a full Git hash")
    _, schedules, _ = verify_support_and_replay()
    core = {
        "candidate_id": "AFCS-144",
        "support_commit": SUPPORT_COMMIT,
        "evaluation_source": str(EVALUATION_SOURCE),
        "evaluation_source_commit": evaluation_source_commit,
        "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
        "evaluation_config": asdict(EvaluationConfig()),
        "control_names": list(support.POLICY_NAMES),
        "control_schedules": {
            name: {"rows": int(len(schedule)), "clock_sha256": _clock_sha256(schedule)}
            for name, schedule in schedules.items()
        },
        "market_sha256": prereg.build_manifest()["source_contract"]["market_sha256"],
        "funding_data_sha256": FUNDING_DATA_SHA256,
        "funding_manifest_sha256": FUNDING_MANIFEST_SHA256,
        "opened_windows": [],
        "sealed_windows": ["stage1_2020_2022", "stage2_2023", "2024", "2025", "2026_ytd"],
        "mutable_parameters": [],
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_settlement_marks_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "stage1_physical_boundary": "stop before parsing any 2023 OHLC or funding value",
    }
    return _seal(core)


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("AFCS-144 evaluator was not frozen before outcomes")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if _canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("AFCS-144 evaluator freeze manifest hash mismatch")
    if payload.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("AFCS-144 evaluator source differs from freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("AFCS-144 support commit differs from freeze")
    if payload.get("evaluation_config") != asdict(EvaluationConfig()):
        raise ValueError("AFCS-144 evaluation configuration drifted")
    if payload.get("opened_windows") != [] or payload.get("mutable_parameters") != []:
        raise ValueError("AFCS-144 evaluator freeze is not sealed")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("AFCS-144 freeze parsed execution OHLC")
    if payload.get("funding_settlement_marks_loaded_during_freeze") != 0:
        raise ValueError("AFCS-144 freeze loaded funding marks")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise ValueError("AFCS-144 freeze simulated outcomes")
    _, schedules, _ = verify_support_and_replay()
    for name, schedule in schedules.items():
        expected = payload["control_schedules"][name]
        if expected["rows"] != len(schedule) or expected["clock_sha256"] != _clock_sha256(schedule):
            raise ValueError(f"AFCS-144 frozen clock drifted: {name}")
    return payload


def _parse_market_before(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    rows: list[tuple[Any, ...]] = []
    boundary_seen = False
    wanted = ("date", "open", "high", "low", "close")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        if positions["date"] != 0:
            raise ValueError("AFCS-144 market date must be the first physical column")
        cutoff_date = cutoff.strftime("%Y-%m-%d")
        for line in handle:
            date_text = line.split(",", 1)[0]
            if date_text[:10] >= cutoff_date:
                boundary_seen = True
                break
            fields = line.rstrip("\r\n").split(",")
            rows.append(
                (
                    date_text,
                    float(fields[positions["open"]]),
                    float(fields[positions["high"]]),
                    float(fields[positions["low"]]),
                    float(fields[positions["close"]]),
                )
            )
    if cutoff < pd.Timestamp("2024-01-01") and not boundary_seen:
        raise ValueError("AFCS-144 market source did not reach the sealed boundary")
    frame = pd.DataFrame(rows, columns=wanted)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def _parse_funding_before(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    rows: list[tuple[Any, ...]] = []
    boundary_seen = False
    wanted = (
        "funding_time_ms",
        "funding_time_utc",
        "funding_rate",
        "settlement_mark_price",
    )
    cutoff_ms = int(cutoff.timestamp() * 1_000)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        if positions["funding_time_ms"] != 0:
            raise ValueError("AFCS-144 funding milliseconds must be first")
        for line in handle:
            first = line.split(",", 1)[0]
            if int(first) >= cutoff_ms:
                boundary_seen = True
                break
            fields = line.rstrip("\r\n").split(",")
            rows.append(
                (
                    int(first),
                    fields[positions["funding_time_utc"]],
                    float(fields[positions["funding_rate"]]),
                    float(fields[positions["settlement_mark_price"]]),
                )
            )
    if cutoff < pd.Timestamp("2024-01-01") and not boundary_seen:
        raise ValueError("AFCS-144 funding source did not reach the sealed boundary")
    frame = pd.DataFrame(rows, columns=wanted)
    frame["funding_time"] = pd.to_datetime(
        frame["funding_time_utc"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    return frame


def load_execution_inputs(
    cutoff: pd.Timestamp,
    signal_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    verify_evaluation_freeze()
    source = prereg.build_manifest()["source_contract"]
    if _sha256(prereg.MARKET_PATH) != source["market_sha256"]:
        raise ValueError("AFCS-144 execution market hash changed")
    if _sha256(FUNDING_DATA) != FUNDING_DATA_SHA256:
        raise ValueError("AFCS-144 funding marks hash changed")
    if _sha256(FUNDING_MANIFEST) != FUNDING_MANIFEST_SHA256:
        raise ValueError("AFCS-144 funding manifest hash changed")
    funding_manifest = json.loads(FUNDING_MANIFEST.read_text())
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("AFCS-144 funding mark source opened outcomes")
    market = _parse_market_before(Path(prereg.MARKET_PATH), cutoff)
    funding = _parse_funding_before(FUNDING_DATA, cutoff)
    expected_dates = signal_frame.loc[signal_frame["date"].lt(cutoff), "date"].reset_index(drop=True)
    if not market["date"].equals(expected_dates):
        raise ValueError("AFCS-144 execution prefix is not aligned to the signal clock")
    values = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("AFCS-144 execution market has invalid prices")
    opening, high, low, close = values.T
    if (
        (high < np.maximum(opening, close)).any()
        or (low > np.minimum(opening, close)).any()
        or (high < low).any()
    ):
        raise ValueError("AFCS-144 market violates OHLC invariants")
    funding_values = funding[["funding_rate", "settlement_mark_price"]].to_numpy(float)
    if not np.isfinite(funding_values).all() or (funding["settlement_mark_price"] <= 0.0).any():
        raise ValueError("AFCS-144 funding source has invalid values")
    if len(funding) and funding["funding_time"].max() >= cutoff:
        raise ValueError("AFCS-144 funding parser crossed cutoff")
    return market, funding, {
        "cutoff": cutoff.isoformat(),
        "market_rows_parsed": int(len(market)),
        "funding_rows_parsed": int(len(funding)),
        "physical_parse_boundary": f"stopped before parsing execution values at {cutoff.isoformat()}",
        "last_market_time": str(market["date"].iloc[-1]),
        "last_funding_time": str(funding["funding_time"].iloc[-1]) if len(funding) else None,
    }


def linear_pnl(*, side: int, quantity: float, entry: float, mark: float) -> float:
    if side not in {-1, 1} or min(quantity, entry, mark) <= 0.0:
        raise ValueError("invalid AFCS-144 linear-ledger input")
    return float(side) * quantity * (mark - entry)


def weekly_cluster_signflip(
    trade_returns: list[float],
    entry_dates: list[str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    if len(trade_returns) != len(entry_dates):
        raise ValueError("AFCS-144 trade returns and dates differ in length")
    if not trade_returns:
        return {"p_value_one_sided": 1.0, "cluster_count": 0, "method": "empty"}
    values = np.asarray(trade_returns, dtype=float)
    dates = pd.to_datetime(pd.Series(entry_dates), errors="raise")
    monday = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.floor("D")
    weekly = pd.DataFrame({"week": monday, "return": values}).groupby("week")["return"].sum()
    cluster_values = weekly.to_numpy(float)
    observed = float(values.mean())
    if len(cluster_values) <= 20:
        exceed = sum(
            float(np.dot(np.asarray(signs), cluster_values) / len(values)) >= observed - 1e-15
            for signs in product((-1.0, 1.0), repeat=len(cluster_values))
        )
        total = 2 ** len(cluster_values)
        p_value = exceed / total
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        exceed = 0
        remaining = permutations
        while remaining:
            batch = min(4_096, remaining)
            signs = rng.choice((-1.0, 1.0), size=(batch, len(cluster_values)))
            exceed += int((signs.dot(cluster_values) / len(values) >= observed - 1e-15).sum())
            remaining -= batch
        p_value = (1 + exceed) / (permutations + 1)
        method = "monte_carlo"
    return {
        "p_value_one_sided": float(p_value),
        "cluster_count": int(len(cluster_values)),
        "method": method,
        "observed_mean_return": observed,
    }


def _window_schedule(
    schedule: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    origin = pd.to_datetime(schedule["origin_date"], errors="raise")
    signal = pd.to_datetime(schedule["signal_date"], errors="raise")
    entry = pd.to_datetime(schedule["entry_date"], errors="raise")
    exit_ = pd.to_datetime(schedule["exit_date"], errors="raise")
    inside = (
        origin.ge(start)
        & signal.ge(start)
        & entry.ge(start)
        & exit_.ge(start)
        & origin.lt(end)
        & signal.lt(end)
        & entry.lt(end)
        & exit_.lt(end)
    )
    return schedule.loc[inside].reset_index(drop=True)


def simulate_schedule(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    cost_rate: float,
    cfg: EvaluationConfig = EvaluationConfig(),
    compute_cluster: bool = True,
) -> dict[str, Any]:
    if cfg.leverage <= 0.0 or not 0.0 <= cost_rate < 0.1 or period_end <= period_start:
        raise ValueError("invalid AFCS-144 simulation configuration")
    policy = prereg.Policy()
    equity = 1.0
    high_water = 1.0
    strict_mdd = 0.0
    previous_exit = -1
    total_cost = 0.0
    total_funding = 0.0
    total_price_pnl = 0.0
    trade_returns: list[float] = []
    gross_returns: list[float] = []
    entry_dates: list[str] = []
    sides: list[int] = []
    funding_events = 0
    details: list[dict[str, Any]] = []
    funding_buckets: dict[pd.Timestamp, list[tuple[float, float]]] = {}
    for row in funding.itertuples(index=False):
        bucket = pd.Timestamp(row.funding_time).floor("5min")
        funding_buckets.setdefault(bucket, []).append(
            (float(row.funding_rate), float(row.settlement_mark_price))
        )

    for event in schedule.itertuples(index=False):
        origin_position = int(event.origin_position)
        signal_position = int(event.signal_position)
        entry_position = int(event.entry_position)
        exit_position = int(event.exit_position)
        side = int(event.side)
        if side not in {-1, 1}:
            raise ValueError("AFCS-144 side must be long or short")
        if entry_position != signal_position + policy.execution_delay_bars:
            raise ValueError("AFCS-144 entry differs from frozen t+2 delay")
        if exit_position != entry_position + policy.hold_bars:
            raise ValueError("AFCS-144 exit differs from frozen hold")
        if int(event.delay_bars) != policy.execution_delay_bars or int(event.hold_bars) != policy.hold_bars:
            raise ValueError("AFCS-144 schedule metadata differs from freeze")
        if not origin_position <= signal_position < entry_position < exit_position:
            raise ValueError("AFCS-144 scheduled positions are invalid")
        if entry_position < previous_exit:
            raise ValueError("AFCS-144 schedules overlap")
        if exit_position >= len(market):
            raise ValueError("AFCS-144 schedule exceeds parsed market prefix")
        dates = market["date"]
        if not (
            period_start <= dates.iloc[origin_position] < period_end
            and period_start <= dates.iloc[signal_position] < period_end
            and period_start <= dates.iloc[entry_position] < period_end
            and period_start <= dates.iloc[exit_position] < period_end
        ):
            raise ValueError("AFCS-144 trade crosses simulation split")
        if str(dates.iloc[entry_position]) != str(event.entry_date):
            raise ValueError("AFCS-144 entry timestamp differs from frozen clock")

        entry_price = float(market.loc[entry_position, "open"])
        exit_price = float(market.loc[exit_position, "open"])
        pre_equity = equity
        quantity = cfg.leverage * pre_equity / entry_price
        entry_fee = cost_rate * quantity * entry_price
        total_cost += entry_fee
        cumulative_funding = 0.0
        equity_after_entry = pre_equity - entry_fee
        strict_mdd = max(strict_mdd, 1.0 - max(0.0, equity_after_entry) / high_water)
        high_water = max(high_water, pre_equity)

        for position in range(entry_position, exit_position):
            bar = market.iloc[position]
            timestamp = pd.Timestamp(bar["date"])
            for rate, settlement_mark in funding_buckets.get(timestamp, []):
                funding_cash = -side * quantity * rate * settlement_mark
                cumulative_funding += funding_cash
                funding_events += 1
                settlement_equity = (
                    pre_equity
                    - entry_fee
                    + cumulative_funding
                    + linear_pnl(
                        side=side,
                        quantity=quantity,
                        entry=entry_price,
                        mark=settlement_mark,
                    )
                )
                high_water = max(high_water, settlement_equity)
                liquidation = settlement_equity - cost_rate * quantity * settlement_mark
                strict_mdd = max(
                    strict_mdd,
                    1.0 - max(0.0, liquidation) / max(high_water, 1e-15),
                )

            favorable_mark = float(bar["high"] if side > 0 else bar["low"])
            favorable_equity = (
                pre_equity
                - entry_fee
                + cumulative_funding
                + linear_pnl(
                    side=side,
                    quantity=quantity,
                    entry=entry_price,
                    mark=favorable_mark,
                )
            )
            high_water = max(high_water, favorable_equity)
            adverse_mark = float(bar["low"] if side > 0 else bar["high"])
            adverse_equity = (
                pre_equity
                - entry_fee
                + cumulative_funding
                + linear_pnl(
                    side=side,
                    quantity=quantity,
                    entry=entry_price,
                    mark=adverse_mark,
                )
                - cost_rate * quantity * adverse_mark
            )
            strict_mdd = max(
                strict_mdd,
                1.0 - max(0.0, adverse_equity) / max(high_water, 1e-15),
            )

        price_pnl = linear_pnl(
            side=side,
            quantity=quantity,
            entry=entry_price,
            mark=exit_price,
        )
        exit_fee = cost_rate * quantity * exit_price
        total_cost += exit_fee
        total_funding += cumulative_funding
        total_price_pnl += price_pnl
        equity = pre_equity - entry_fee + price_pnl + cumulative_funding - exit_fee
        strict_mdd = max(
            strict_mdd,
            1.0 - max(0.0, equity) / max(high_water, 1e-15),
        )
        high_water = max(high_water, equity)
        net_return = equity / pre_equity - 1.0
        gross_return = side * (exit_price / entry_price - 1.0)
        trade_returns.append(net_return)
        gross_returns.append(gross_return)
        entry_dates.append(str(event.entry_date))
        sides.append(side)
        details.append(
            {
                "entry_date": str(event.entry_date),
                "exit_date": str(event.exit_date),
                "side": side,
                "branch": str(event.branch),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity_btc": quantity,
                "price_pnl": price_pnl,
                "funding_cash": cumulative_funding,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "net_return": net_return,
                "gross_underlying_return": gross_return,
            }
        )
        previous_exit = exit_position

    years = (period_end - period_start).total_seconds() / (365.25 * 86_400.0)
    absolute_return = equity - 1.0
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0.0 else -1.0
    ratio = cagr / strict_mdd if strict_mdd > 1e-15 else (math.inf if cagr > 0 else 0.0)
    cluster = (
        weekly_cluster_signflip(
            trade_returns,
            entry_dates,
            permutations=cfg.cluster_permutations,
            seed=cfg.cluster_seed,
        )
        if compute_cluster
        else None
    )
    return {
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "calendar_years": years,
        "absolute_return_pct": absolute_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "strict_mdd_pct": strict_mdd * 100.0,
        "cagr_to_strict_mdd": ratio,
        "trade_count": int(len(trade_returns)),
        "long_count": int(sum(side > 0 for side in sides)),
        "short_count": int(sum(side < 0 for side in sides)),
        "mean_gross_underlying_move_bp": (
            float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0
        ),
        "transaction_cost_pct_initial": total_cost * 100.0,
        "price_pnl_pct_initial": total_price_pnl * 100.0,
        "funding_cash_pct_initial": total_funding * 100.0,
        "funding_settlement_count": int(funding_events),
        "weekly_cluster_signflip": cluster,
        "trade_details": details,
    }


def _stage_gate(metrics: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    cfg = EvaluationConfig()
    return {
        "absolute_return_positive": metrics["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": metrics["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": metrics["strict_mdd_pct"] <= 15.0,
        "weekly_cluster_p_at_most_0p10": metrics["weekly_cluster_signflip"]["p_value_one_sided"] <= 0.10,
        "mean_gross_underlying_move_above_20bp": metrics["mean_gross_underlying_move_bp"] > cfg.minimum_mean_gross_underlying_bp,
        "stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
    }


def _passes_stage_gate(
    metrics: dict[str, Any],
    stress: dict[str, Any],
    *,
    minimum_trades: int,
) -> bool:
    return (
        all(_stage_gate(metrics, stress).values())
        and metrics["trade_count"] >= minimum_trades
    )


def evaluate_stage1() -> dict[str, Any]:
    freeze = verify_evaluation_freeze()
    signal_frame, schedules, _ = verify_support_and_replay()
    market, funding, source = load_execution_inputs(STAGE1_END, signal_frame)
    stage_schedules = {
        name: _window_schedule(schedule, STAGE1_START, STAGE1_END)
        for name, schedule in schedules.items()
    }
    cfg = EvaluationConfig()
    base = simulate_schedule(
        market,
        funding,
        stage_schedules["primary"],
        period_start=STAGE1_START,
        period_end=STAGE1_END,
        cost_rate=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    stress = simulate_schedule(
        market,
        funding,
        stage_schedules["primary"],
        period_start=STAGE1_START,
        period_end=STAGE1_END,
        cost_rate=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    controls = {
        name: simulate_schedule(
            market,
            funding,
            schedule,
            period_start=STAGE1_START,
            period_end=STAGE1_END,
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        for name, schedule in stage_schedules.items()
        if name != "primary"
    }
    annual: dict[str, Any] = {}
    half_year: dict[str, Any] = {}
    for year in range(2020, 2023):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year + 1}-01-01")
        annual[str(year)] = simulate_schedule(
            market,
            funding,
            _window_schedule(stage_schedules["primary"], start, end),
            period_start=start,
            period_end=end,
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
            compute_cluster=False,
        )
        for half, (half_start, half_end) in {
            "h1": (start, pd.Timestamp(f"{year}-07-01")),
            "h2": (pd.Timestamp(f"{year}-07-01"), end),
        }.items():
            half_year[f"{year}_{half}"] = simulate_schedule(
                market,
                funding,
                _window_schedule(stage_schedules["primary"], half_start, half_end),
                period_start=half_start,
                period_end=half_end,
                cost_rate=cfg.base_cost_notional_per_side,
                cfg=cfg,
                compute_cluster=False,
            )

    gate = _stage_gate(base, stress)
    gate["trade_count_at_least_250"] = base["trade_count"] >= 250
    primary_ratio = base["cagr_to_strict_mdd"]
    gate["beats_each_component_removal_ratio"] = all(
        primary_ratio > controls[name]["cagr_to_strict_mdd"]
        for name in COMPONENT_CONTROLS
    )
    passing_placebos = [
        name
        for name in REJECTION_PLACEBOS
        if _passes_stage_gate(
            controls[name],
            simulate_schedule(
                market,
                funding,
                stage_schedules[name],
                period_start=STAGE1_START,
                period_end=STAGE1_END,
                cost_rate=cfg.stress_cost_notional_per_side,
                cfg=cfg,
            ),
            minimum_trades=250,
        )
    ]
    gate["no_passing_rejection_placebo"] = not passing_placebos
    qualifies = all(gate.values())
    core = {
        "candidate_id": "AFCS-144",
        "stage": "stage1_2020_2022",
        "evaluation_freeze_sha256": _sha256(EVALUATION_FREEZE),
        "evaluation_source_commit": freeze["evaluation_source_commit"],
        "source": source,
        "accounting": {
            "fixed_quantity_linear_usdm": True,
            "funding_interval": "entry_time <= funding_time < exit_time",
            "funding_mark": "frozen official 8h mark-price kline open proxy",
            "strict_mdd": (
                "global/pre-entry HWM; each held bar favorable then adverse; "
                "funding at exact settlement bucket; hypothetical liquidation cost"
            ),
            "cagr_clock": "2020-01-01 through 2023-01-01 including idle cash",
        },
        "base": base,
        "stress_10bp": stress,
        "annual": annual,
        "half_year": half_year,
        "controls": controls,
        "gate": gate,
        "passing_rejection_placebos": passing_placebos,
        "stage1_qualifies": qualifies,
        "next_action": "open_2023" if qualifies else "reject_keep_2023_and_2024plus_sealed",
        "sealed_after_run": ["2023", "2024", "2025", "2026_ytd"] if not qualifies else ["2024", "2025", "2026_ytd"],
    }
    return _seal(core)


def evaluate_stage2() -> dict[str, Any]:
    if not STAGE1_OUTPUT.is_file():
        raise ValueError("AFCS-144 stage 1 result is missing")
    stage1 = json.loads(STAGE1_OUTPUT.read_text())
    if stage1.get("stage1_qualifies") is not True:
        raise ValueError("AFCS-144 stage 1 failed; 2023 must remain sealed")
    freeze = verify_evaluation_freeze()
    signal_frame, schedules, _ = verify_support_and_replay()
    market, funding, source = load_execution_inputs(STAGE2_END, signal_frame)
    cfg = EvaluationConfig()
    stage_schedules = {
        name: _window_schedule(schedule, STAGE2_START, STAGE2_END)
        for name, schedule in schedules.items()
    }
    selected = stage_schedules["primary"]
    base = simulate_schedule(
        market,
        funding,
        selected,
        period_start=STAGE2_START,
        period_end=STAGE2_END,
        cost_rate=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    stress = simulate_schedule(
        market,
        funding,
        selected,
        period_start=STAGE2_START,
        period_end=STAGE2_END,
        cost_rate=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    controls = {
        name: simulate_schedule(
            market,
            funding,
            schedule,
            period_start=STAGE2_START,
            period_end=STAGE2_END,
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        for name, schedule in stage_schedules.items()
        if name != "primary"
    }
    halves = {}
    for name, (start, end) in {
        "2023_h1": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-01")),
        "2023_h2": (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-01-01")),
    }.items():
        halves[name] = simulate_schedule(
            market,
            funding,
            _window_schedule(selected, start, end),
            period_start=start,
            period_end=end,
            cost_rate=cfg.base_cost_notional_per_side,
            cfg=cfg,
            compute_cluster=False,
        )
    gate = _stage_gate(base, stress)
    gate["trade_count_at_least_60"] = base["trade_count"] >= 60
    gate["both_halves_positive"] = all(
        value["absolute_return_pct"] > 0.0 and value["trade_count"] >= 25
        for value in halves.values()
    )
    primary_ratio = base["cagr_to_strict_mdd"]
    gate["beats_each_component_removal_ratio"] = all(
        primary_ratio > controls[name]["cagr_to_strict_mdd"]
        for name in COMPONENT_CONTROLS
    )
    passing_placebos = [
        name
        for name in REJECTION_PLACEBOS
        if _passes_stage_gate(
            controls[name],
            simulate_schedule(
                market,
                funding,
                stage_schedules[name],
                period_start=STAGE2_START,
                period_end=STAGE2_END,
                cost_rate=cfg.stress_cost_notional_per_side,
                cfg=cfg,
            ),
            minimum_trades=60,
        )
    ]
    gate["no_passing_rejection_placebo"] = not passing_placebos
    qualifies = all(gate.values())
    return _seal(
        {
            "candidate_id": "AFCS-144",
            "stage": "stage2_2023",
            "evaluation_source_commit": freeze["evaluation_source_commit"],
            "source": source,
            "base": base,
            "stress_10bp": stress,
            "halves": halves,
            "controls": controls,
            "gate": gate,
            "passing_rejection_placebos": passing_placebos,
            "stage2_qualifies": qualifies,
            "next_action": "open_2024" if qualifies else "reject_keep_2024plus_sealed",
            "sealed_after_run": ["2024", "2025", "2026_ytd"],
        }
    )


def _write_once(path: Path, payload: dict[str, Any]) -> str:
    content = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite frozen AFCS-144 result: {path}")
        return "verified_existing"
    with path.open("xb") as handle:
        handle.write(content)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    args = parser.parse_args()
    if args.stage == "stage1":
        payload = evaluate_stage1()
        output = STAGE1_OUTPUT
    else:
        payload = evaluate_stage2()
        output = STAGE2_OUTPUT
    status = _write_once(output, payload)
    primary = payload["base"]
    print(
        json.dumps(
            {
                "status": status,
                "stage": payload["stage"],
                "absolute_return_pct": primary["absolute_return_pct"],
                "cagr_pct": primary["cagr_pct"],
                "strict_mdd_pct": primary["strict_mdd_pct"],
                "cagr_to_strict_mdd": primary["cagr_to_strict_mdd"],
                "trade_count": primary["trade_count"],
                "qualifies": payload.get("stage1_qualifies", payload.get("stage2_qualifies")),
                "next_action": payload["next_action"],
                "output": str(output),
                "manifest_hash": payload["manifest_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
