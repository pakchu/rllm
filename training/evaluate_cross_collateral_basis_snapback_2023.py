"""Run the frozen CCBS-12 2023 development ledger after evaluator commit."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import training.build_cross_collateral_basis_snapback_support as support_builder
from training.build_cross_collateral_basis_snapback_support import (
    PREREGISTRATION_HASH,
    SOURCE_MANIFEST,
    SOURCE_PANEL,
    SIGNAL_COLUMNS,
    build_signal_features,
    file_sha256,
)
from training.preregister_cross_collateral_basis_snapback import (
    SOURCE_MANIFEST_CONTENT_HASH,
    SOURCE_MANIFEST_FILE_SHA256,
    SOURCE_PANEL_FILE_SHA256,
    canonical_hash,
)


SUPPORT = "results/cross_collateral_basis_snapback_support_2026-07-17.json"
SUPPORT_CONTENT_HASH = "29a002968e784604512e83407facb0d53a0a3c6536d1038af4f9d44adf51d4f1"
OUTPUT = "results/cross_collateral_basis_snapback_2023_evaluation_2026-07-17.json"
DOCS_OUTPUT = "docs/cross-collateral-basis-snapback-2023-evaluation-2026-07-17.md"

FULL_COLUMNS = (
    "open_time",
    "available_time",
    "um_open",
    "um_high",
    "um_low",
    "um_close",
    "um_ohlc_valid",
    "cm_open",
    "cm_high",
    "cm_low",
    "cm_close",
    "cm_ohlc_valid",
    "source_complete",
    "delivery_time",
    "contract_segment",
)


@dataclass(frozen=True)
class Config:
    source_manifest: str = SOURCE_MANIFEST
    source_panel: str = SOURCE_PANEL
    support: str = SUPPORT
    output: str = OUTPUT
    docs_output: str = DOCS_OUTPUT
    period_start: str = "2023-01-01"
    period_end: str = "2024-01-01"
    lookback_bars: int = 4_032
    minimum_prior_bars: int = 3_226
    normalization_z: float = 0.5
    maximum_hold_bars: int = 144
    base_cost_rate: float = 0.0006
    stress_cost_rate: float = 0.0010
    cm_contract_multiplier_usd: float = 100.0


@dataclass(frozen=True)
class ExitDecision:
    exit_time: pd.Timestamp
    reason: str
    trigger_open_time: pd.Timestamp | None


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_inputs(cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(cfg.source_manifest)
    if file_sha256(cfg.source_manifest) != SOURCE_MANIFEST_FILE_SHA256:
        raise ValueError("source manifest file hash drifted")
    if manifest.get("manifest_hash") != SOURCE_MANIFEST_CONTENT_HASH:
        raise ValueError("source manifest content hash drifted")
    if file_sha256(cfg.source_panel) != SOURCE_PANEL_FILE_SHA256:
        raise ValueError("source panel hash drifted")

    support = _load_json(cfg.support)
    body = {
        key: value
        for key, value in support.items()
        if key not in {"as_of", "content_hash"}
    }
    if canonical_hash(body) != SUPPORT_CONTENT_HASH:
        raise ValueError("support artifact body hash drifted")
    if support.get("content_hash") != SUPPORT_CONTENT_HASH:
        raise ValueError("support artifact content hash drifted")
    if file_sha256(support_builder.__file__) != support.get("support_builder_sha256"):
        raise ValueError("support builder implementation hash drifted")
    if support.get("protocol_hash") != PREREGISTRATION_HASH:
        raise ValueError("support is not bound to the frozen preregistration")
    if support.get("disposition") != "PASS_SUPPORT_OPEN_2023_PNL":
        raise ValueError("support artifact did not authorize 2023 PnL")
    if float(support.get("selected_threshold")) != 2.0:
        raise ValueError("support-selected threshold drifted")
    return manifest, support


def load_full_frame(path: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=list(FULL_COLUMNS),
        parse_dates=["open_time", "available_time", "delivery_time"],
    )
    if tuple(frame.columns) != FULL_COLUMNS:
        raise ValueError("full evaluation columns drifted")
    if frame["open_time"].duplicated().any() or not frame["open_time"].is_monotonic_increasing:
        raise ValueError("evaluation clock is duplicated or unsorted")
    return frame


def _signal_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(SIGNAL_COLUMNS)].copy()


def select_exit(
    event: dict[str, Any],
    features: pd.DataFrame,
    *,
    normalization_z: float,
    maximum_hold_bars: int,
) -> ExitDecision:
    entry_time = pd.Timestamp(event["entry_time"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    else:
        entry_time = entry_time.tz_convert("UTC")
    hard_exit = entry_time + pd.Timedelta(minutes=5 * maximum_hold_bars)
    latest_trigger_open = hard_exit - pd.Timedelta(minutes=10)
    possible = features.loc[
        features["open_time"].between(entry_time, latest_trigger_open, inclusive="both")
        & features["zscore"].abs().le(float(normalization_z))
    ]
    if possible.empty:
        return ExitDecision(hard_exit, "time", None)
    trigger = pd.Timestamp(possible.iloc[0]["open_time"])
    return ExitDecision(trigger + pd.Timedelta(minutes=10), "normalization", trigger)


def um_pnl_usd(
    *,
    side: int,
    quantity_btc: float,
    entry_price: float,
    mark_price: float,
) -> float:
    if side not in {-1, 1} or min(quantity_btc, entry_price, mark_price) <= 0.0:
        raise ValueError("invalid USD-M ledger input")
    return float(side) * quantity_btc * (mark_price - entry_price)


def cm_inverse_pnl_usd(
    *,
    side: int,
    contracts: float,
    multiplier_usd: float,
    entry_price: float,
    mark_price: float,
) -> float:
    if side not in {-1, 1} or min(contracts, multiplier_usd, entry_price, mark_price) <= 0.0:
        raise ValueError("invalid COIN-M ledger input")
    coin_pnl = (
        float(side)
        * contracts
        * multiplier_usd
        * (1.0 / entry_price - 1.0 / mark_price)
    )
    return coin_pnl * mark_price


def monthly_signflip_pvalue(trade_returns: pd.DataFrame) -> tuple[float, int, dict[str, float]]:
    if trade_returns.empty:
        return 1.0, 0, {}
    months = pd.to_datetime(trade_returns["entry_time"], utc=True).dt.strftime("%Y-%m")
    monthly = trade_returns.assign(month=months).groupby("month")["net_return"].sum()
    values = monthly.to_numpy(float)
    if not len(values):
        return 1.0, 0, {}
    observed = float(values.sum())
    exceed = 0
    permutations = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        if float(np.dot(np.asarray(signs), values)) >= observed - 1e-15:
            exceed += 1
    return (
        exceed / permutations,
        len(values),
        {str(index): float(value) for index, value in monthly.items()},
    )


def _compound(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.prod(1.0 + array) - 1.0) if len(array) else 0.0


def _mark_equity(
    *,
    pre_entry_equity: float,
    entry_fee: float,
    um_side: int,
    cm_side: int,
    um_quantity: float,
    cm_contracts: float,
    cm_multiplier: float,
    um_entry: float,
    cm_entry: float,
    um_mark: float,
    cm_mark: float,
) -> float:
    return (
        pre_entry_equity
        - entry_fee
        + um_pnl_usd(
            side=um_side,
            quantity_btc=um_quantity,
            entry_price=um_entry,
            mark_price=um_mark,
        )
        + cm_inverse_pnl_usd(
            side=cm_side,
            contracts=cm_contracts,
            multiplier_usd=cm_multiplier,
            entry_price=cm_entry,
            mark_price=cm_mark,
        )
    )


def run_ledger(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    events: list[dict[str, Any]],
    *,
    cost_rate: float,
    normalization_z: float,
    maximum_hold_bars: int,
    cm_multiplier: float,
) -> dict[str, Any]:
    if cost_rate < 0.0 or maximum_hold_bars < 1:
        raise ValueError("invalid ledger cost or hold")
    indexed = frame.set_index("open_time", drop=False)
    equity = 1.0
    high_water_mark = 1.0
    maximum_drawdown = 0.0
    transaction_cost = 0.0
    pre_cost_pnl = 0.0
    trades: list[dict[str, Any]] = []
    previous_maximum_exit: pd.Timestamp | None = None

    for event in sorted(events, key=lambda row: str(row["entry_time"])):
        entry_time = pd.Timestamp(event["entry_time"])
        entry_time = (
            entry_time.tz_localize("UTC")
            if entry_time.tzinfo is None
            else entry_time.tz_convert("UTC")
        )
        reserved_exit = pd.Timestamp(event["maximum_exit_time"])
        reserved_exit = (
            reserved_exit.tz_localize("UTC")
            if reserved_exit.tzinfo is None
            else reserved_exit.tz_convert("UTC")
        )
        if previous_maximum_exit is not None and entry_time < previous_maximum_exit:
            raise ValueError("support events violate full reservation")
        previous_maximum_exit = reserved_exit
        decision = select_exit(
            event,
            features,
            normalization_z=normalization_z,
            maximum_hold_bars=maximum_hold_bars,
        )
        if decision.exit_time > reserved_exit:
            raise ValueError("dynamic exit exceeds frozen reservation")
        if entry_time not in indexed.index or decision.exit_time not in indexed.index:
            raise ValueError("entry or exit open is unavailable")

        entry_row = indexed.loc[entry_time]
        exit_row = indexed.loc[decision.exit_time]
        segment = str(event["contract_segment"])
        if str(entry_row["contract_segment"]) != segment or str(exit_row["contract_segment"]) != segment:
            raise ValueError("trade crossed a delivery segment")
        if decision.exit_time >= pd.Timestamp(event["delivery_time"]):
            raise ValueError("trade touched or crossed delivery")
        held = frame.loc[
            frame["open_time"].between(entry_time, decision.exit_time, inclusive="left")
        ]
        if held.empty or not held["contract_segment"].astype(str).eq(segment).all():
            raise ValueError("held path is empty or crossed a delivery segment")
        clean_columns = ["source_complete", "um_ohlc_valid", "cm_ohlc_valid"]
        if not held[clean_columns].astype(bool).all(axis=None):
            raise ValueError("held path contains an incomplete source row")
        if not bool(exit_row[clean_columns].astype(bool).all()):
            raise ValueError("exit row is incomplete")

        z_signal = float(event["zscore"])
        if not np.isfinite(z_signal) or z_signal == 0.0:
            raise ValueError("event has no frozen spread sign")
        um_side = -1 if z_signal > 0.0 else 1
        cm_side = -um_side
        um_entry = float(entry_row["um_open"])
        cm_entry = float(entry_row["cm_open"])
        pre_equity = equity
        face = 0.5 * pre_equity
        um_quantity = face / um_entry
        cm_contracts = face / cm_multiplier
        entry_fee = (
            cost_rate * abs(um_quantity) * um_entry
            + cost_rate * abs(cm_contracts) * cm_multiplier
        )
        transaction_cost += entry_fee
        equity_after_entry = pre_equity - entry_fee
        maximum_drawdown = max(
            maximum_drawdown,
            1.0 - equity_after_entry / max(high_water_mark, 1e-15),
        )

        for _, bar in held.iterrows():
            um_favorable = float(bar["um_high"] if um_side > 0 else bar["um_low"])
            cm_favorable = float(bar["cm_high"] if cm_side > 0 else bar["cm_low"])
            favorable_equity = _mark_equity(
                pre_entry_equity=pre_equity,
                entry_fee=entry_fee,
                um_side=um_side,
                cm_side=cm_side,
                um_quantity=um_quantity,
                cm_contracts=cm_contracts,
                cm_multiplier=cm_multiplier,
                um_entry=um_entry,
                cm_entry=cm_entry,
                um_mark=um_favorable,
                cm_mark=cm_favorable,
            )
            high_water_mark = max(high_water_mark, favorable_equity)

            um_adverse = float(bar["um_low"] if um_side > 0 else bar["um_high"])
            cm_adverse = float(bar["cm_low"] if cm_side > 0 else bar["cm_high"])
            hypothetical_exit_fee = (
                cost_rate * abs(um_quantity) * um_adverse
                + cost_rate * abs(cm_contracts) * cm_multiplier
            )
            adverse_equity = _mark_equity(
                pre_entry_equity=pre_equity,
                entry_fee=entry_fee,
                um_side=um_side,
                cm_side=cm_side,
                um_quantity=um_quantity,
                cm_contracts=cm_contracts,
                cm_multiplier=cm_multiplier,
                um_entry=um_entry,
                cm_entry=cm_entry,
                um_mark=um_adverse,
                cm_mark=cm_adverse,
            ) - hypothetical_exit_fee
            maximum_drawdown = max(
                maximum_drawdown,
                1.0 - adverse_equity / max(high_water_mark, 1e-15),
            )

        um_exit = float(exit_row["um_open"])
        cm_exit = float(exit_row["cm_open"])
        gross_pnl = (
            um_pnl_usd(
                side=um_side,
                quantity_btc=um_quantity,
                entry_price=um_entry,
                mark_price=um_exit,
            )
            + cm_inverse_pnl_usd(
                side=cm_side,
                contracts=cm_contracts,
                multiplier_usd=cm_multiplier,
                entry_price=cm_entry,
                mark_price=cm_exit,
            )
        )
        exit_fee = (
            cost_rate * abs(um_quantity) * um_exit
            + cost_rate * abs(cm_contracts) * cm_multiplier
        )
        transaction_cost += exit_fee
        pre_cost_pnl += gross_pnl
        equity = pre_equity - entry_fee + gross_pnl - exit_fee
        maximum_drawdown = max(
            maximum_drawdown,
            1.0 - equity / max(high_water_mark, 1e-15),
        )
        high_water_mark = max(high_water_mark, equity)
        entry_wedge = float(np.log(um_entry / cm_entry))
        exit_wedge = float(np.log(um_exit / cm_exit))
        trades.append(
            {
                "signal_time": str(event["open_time"]),
                "entry_time": entry_time.isoformat(),
                "exit_time": decision.exit_time.isoformat(),
                "exit_reason": decision.reason,
                "trigger_open_time": (
                    decision.trigger_open_time.isoformat()
                    if decision.trigger_open_time is not None
                    else None
                ),
                "contract_segment": segment,
                "zscore": z_signal,
                "rich_leg": "um" if z_signal > 0.0 else "cm",
                "um_side": um_side,
                "cm_side": cm_side,
                "pre_entry_equity": pre_equity,
                "face_per_leg": face,
                "entry_fee": entry_fee,
                "gross_derivative_pnl": gross_pnl,
                "exit_fee": exit_fee,
                "net_pnl": equity - pre_equity,
                "net_return": equity / pre_equity - 1.0,
                "post_exit_equity": equity,
                "signed_wedge_convergence": -float(np.sign(z_signal))
                * (exit_wedge - entry_wedge),
            }
        )

    return {
        "ending_equity": equity,
        "strict_mdd": maximum_drawdown,
        "pre_cost_pnl": pre_cost_pnl,
        "transaction_cost": transaction_cost,
        "trades": trades,
    }


def summarize_ledger(
    ledger: dict[str, Any],
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> dict[str, Any]:
    years = (period_end - period_start).total_seconds() / (365.25 * 86_400.0)
    ending_equity = float(ledger["ending_equity"])
    absolute_return = ending_equity - 1.0
    cagr = ending_equity ** (1.0 / years) - 1.0 if ending_equity > 0.0 else -1.0
    strict_mdd = float(ledger["strict_mdd"])
    trades = pd.DataFrame(ledger["trades"])
    if trades.empty:
        trades = pd.DataFrame(columns=["entry_time", "net_return", "rich_leg", "signed_wedge_convergence"])
    entry_time = pd.to_datetime(trades["entry_time"], utc=True)
    h1 = trades.loc[entry_time.dt.month <= 6, "net_return"]
    h2 = trades.loc[entry_time.dt.month > 6, "net_return"]
    um_rich = trades.loc[trades["rich_leg"].eq("um"), "net_return"]
    cm_rich = trades.loc[trades["rich_leg"].eq("cm"), "net_return"]
    pvalue, active_months, monthly = monthly_signflip_pvalue(trades)
    convergence = trades["signed_wedge_convergence"].to_numpy(float)
    return {
        "absolute_return_pct": absolute_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "strict_mdd_pct": strict_mdd * 100.0,
        "cagr_to_strict_mdd": cagr / max(strict_mdd, 1e-12),
        "trades": int(len(trades)),
        "h1_absolute_return_pct": _compound(h1) * 100.0,
        "h2_absolute_return_pct": _compound(h2) * 100.0,
        "um_rich_absolute_return_pct": _compound(um_rich) * 100.0,
        "cm_rich_absolute_return_pct": _compound(cm_rich) * 100.0,
        "pre_cost_pnl": float(ledger["pre_cost_pnl"]),
        "transaction_cost": float(ledger["transaction_cost"]),
        "pre_cost_pnl_to_cost": float(ledger["pre_cost_pnl"])
        / max(float(ledger["transaction_cost"]), 1e-15),
        "median_signed_wedge_convergence": (
            float(np.median(convergence)) if len(convergence) else 0.0
        ),
        "signed_wedge_convergence_hit_rate": (
            float((convergence > 0.0).mean()) if len(convergence) else 0.0
        ),
        "monthly_signflip_pvalue": pvalue,
        "active_entry_months": active_months,
        "monthly_net_return_sums": monthly,
    }


def gate_results(base: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    return {
        "absolute_return_positive": base["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": base["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_at_most_15pct": base["strict_mdd_pct"] <= 15.0,
        "trades_at_least_50": base["trades"] >= 50,
        "h1_absolute_return_positive": base["h1_absolute_return_pct"] > 0.0,
        "h2_absolute_return_positive": base["h2_absolute_return_pct"] > 0.0,
        "um_rich_branch_positive": base["um_rich_absolute_return_pct"] > 0.0,
        "cm_rich_branch_positive": base["cm_rich_absolute_return_pct"] > 0.0,
        "ten_bp_stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "pre_cost_pnl_exceeds_transaction_cost": base["pre_cost_pnl"]
        > base["transaction_cost"],
        "median_signed_wedge_convergence_positive": base[
            "median_signed_wedge_convergence"
        ]
        > 0.0,
        "signed_wedge_convergence_hit_rate_at_least_55pct": base[
            "signed_wedge_convergence_hit_rate"
        ]
        >= 0.55,
        "active_entry_months_at_least_6": base["active_entry_months"] >= 6,
        "monthly_signflip_pvalue_at_most_10pct": base["monthly_signflip_pvalue"] <= 0.10,
    }


def _development_events(support: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    start = pd.Timestamp(cfg.period_start, tz="UTC")
    end = pd.Timestamp(cfg.period_end, tz="UTC")
    return [
        event
        for event in support["selected_events_2021_2023"]
        if start <= pd.Timestamp(event["entry_time"]) < end
    ]


def markdown(report: dict[str, Any]) -> str:
    base = report["base_cost"]
    stress = report["stress_cost"]
    failures = [name for name, passed in report["gates"].items() if not passed]
    return f"""# CCBS-12 2023 development evaluation — 2026-07-17

2023 is outcome-blind development, not pristine OOS; 2024 remains sealed.
This result uses the committed CCBS evaluator, fractional derivative quantities,
6 bp base / 10 bp stress costs, full-calendar CAGR, and global favorable-before-
adverse strict MDD. COIN-M BTC collateral remains outside this research ledger.

| Cost | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 6 bp | {base['absolute_return_pct']:.4f}% | {base['cagr_pct']:.4f}% | {base['strict_mdd_pct']:.4f}% | {base['cagr_to_strict_mdd']:.4f} | {base['trades']} |
| 10 bp | {stress['absolute_return_pct']:.4f}% | {stress['cagr_pct']:.4f}% | {stress['strict_mdd_pct']:.4f}% | {stress['cagr_to_strict_mdd']:.4f} | {stress['trades']} |

Disposition: **{report['disposition']}**.

Failed gates: `{failures}`.

2024 may be opened only after every development and subsequent PnL-
orthogonality gate passes. Even a pass is not live-ready until the BTC-
collateral ledger and forward shadow are complete.

Report content hash: `{report['content_hash']}`
"""


def run(cfg: Config) -> dict[str, Any]:
    manifest, support = verify_inputs(cfg)
    frame = load_full_frame(cfg.source_panel)
    features = build_signal_features(
        _signal_view(frame),
        lookback_bars=cfg.lookback_bars,
        minimum_prior_bars=cfg.minimum_prior_bars,
    )
    events = _development_events(support, cfg)
    if len(events) != 58:
        raise ValueError("frozen 2023 event count drifted")
    base_ledger = run_ledger(
        frame,
        features,
        events,
        cost_rate=cfg.base_cost_rate,
        normalization_z=cfg.normalization_z,
        maximum_hold_bars=cfg.maximum_hold_bars,
        cm_multiplier=cfg.cm_contract_multiplier_usd,
    )
    stress_ledger = run_ledger(
        frame,
        features,
        events,
        cost_rate=cfg.stress_cost_rate,
        normalization_z=cfg.normalization_z,
        maximum_hold_bars=cfg.maximum_hold_bars,
        cm_multiplier=cfg.cm_contract_multiplier_usd,
    )
    start = pd.Timestamp(cfg.period_start, tz="UTC")
    end = pd.Timestamp(cfg.period_end, tz="UTC")
    base = summarize_ledger(base_ledger, period_start=start, period_end=end)
    stress = summarize_ledger(stress_ledger, period_start=start, period_end=end)
    gates = gate_results(base, stress)
    passed = all(gates.values())
    body = {
        "mode": "2023_outcome_blind_development_not_pristine_oos",
        "config": asdict(cfg),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_hash": PREREGISTRATION_HASH,
        "support_content_hash": SUPPORT_CONTENT_HASH,
        "source_manifest_content_hash": manifest["manifest_hash"],
        "source_panel_sha256": file_sha256(cfg.source_panel),
        "base_cost": base,
        "stress_cost": stress,
        "gates": gates,
        "gate_passed": passed,
        "base_cost_trades": base_ledger["trades"],
        "stress_cost_trades": stress_ledger["trades"],
        "disposition": (
            "PASS_DEVELOPMENT_PENDING_PNL_ORTHOGONALITY"
            if passed
            else "REJECT_2023_KEEP_2024_SEALED"
        ),
    }
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        **body,
        "content_hash": canonical_hash(body),
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.docs_output).write_text(markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--docs-output", default=DOCS_OUTPUT)
    args = parser.parse_args()
    report = run(Config(output=args.output, docs_output=args.docs_output))
    print(
        json.dumps(
            {
                "base_cost": report["base_cost"],
                "stress_cost": report["stress_cost"],
                "gates": report["gates"],
                "disposition": report["disposition"],
                "content_hash": report["content_hash"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
