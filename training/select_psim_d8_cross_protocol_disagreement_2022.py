#!/usr/bin/env python3
"""Select the frozen PSIM-D8-CDP1 top1 on 2022 only."""

from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from training import preregister_psim_d8_cross_protocol_disagreement_persistence as prereg
from training import run_psim_d8_cross_protocol_disagreement_source_support as source


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "psim_d8_cross_protocol_disagreement_2022_selection_v1r1"
SOURCE_RESULT = source.DEFAULT_OUTPUT
SOURCE_RESULT_SHA256 = (
    "43a6815201d20ed083ee224c0874405370fb3a061455af7cacc32f7c79c16d6f"
)
SOURCE_RESULT_HASH = (
    "3ac80ced7ea528dc53826e398fc4f7621664f79dd744f79de34f20ea2888c0d7"
)
RUNNER_PATH = Path(
    "training/select_psim_d8_cross_protocol_disagreement_2022.py"
)
ATTEMPT_PATH = Path(
    "results/psim_d8_cross_protocol_disagreement_2022_selection_"
    "attempt_r1_2026-07-29.json"
)
FAILED_ATTEMPT_PATH = Path(
    "results/psim_d8_cross_protocol_disagreement_2022_selection_"
    "attempt_2026-07-29.json"
)
FAILED_ATTEMPT_HASH = (
    "2fbb33f3b2d8bacf54e7b50dc6ccfa388d898bc9ae3ff5747c374fef0132036c"
)
RESULT_PATH = Path(
    "results/psim_d8_cross_protocol_disagreement_2022_selection_"
    "result_2026-07-29.json"
)
MONTHLY_MARKET_DIR = Path(
    "data/binance_um_kline_reference_btc_2020_2023/monthly"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with (REPO_ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    target = REPO_ROOT / path
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"write-once artifact drift: {path}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")


def prepare_attempt() -> dict[str, Any]:
    support = _read_json(SOURCE_RESULT)
    if (
        _sha256_file(SOURCE_RESULT) != SOURCE_RESULT_SHA256
        or support.get("result_hash") != SOURCE_RESULT_HASH
        or support.get("decision") != "pass"
        or support.get("2022_market_or_funding_authorized") is not True
        or support.get("2023_market_or_funding_authorized") is not False
    ):
        raise RuntimeError("CDP1 source-support authority drift")
    runner_sha = _sha256_file(RUNNER_PATH)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "2022_selection",
        "runner": {
            "path": RUNNER_PATH.as_posix(),
            "sha256": runner_sha,
        },
        "source_support": {
            "path": SOURCE_RESULT.as_posix(),
            "sha256": SOURCE_RESULT_SHA256,
            "result_hash": SOURCE_RESULT_HASH,
            "eligible_candidate_ids": support["eligible_candidate_ids"],
        },
        "prior_operational_failure": {
            "attempt_path": FAILED_ATTEMPT_PATH.as_posix(),
            "attempt_hash": FAILED_ATTEMPT_HASH,
            "reason": (
                "funding_time_utc has exchange millisecond jitter; exact "
                "settlement mark clock is mark_open_time_utc"
            ),
            "candidate_metrics_computed": 0,
            "thresholds_or_candidates_changed": False,
        },
        "access_boundary": {
            "2022_market_rows_parsed": 105_120,
            "2022_funding_rows_parsed": 1_095,
            "2022_economic_metrics_computed": 0,
            "2023_market_rows_parsed": 0,
            "2023_funding_rows_parsed": 0,
            "2023_economic_metrics_computed": 0,
        },
        "selection_contract": prereg.build_preregistration()[
            "evaluation_contract"
        ],
    }
    payload = {**core, "attempt_hash": prereg.canonical_hash(core)}
    _write_once(ATTEMPT_PATH, payload)
    return payload


def trade_net_return(
    *,
    side: int,
    entry_price: float,
    exit_price: float,
    funding_cashflows: Sequence[float],
    cost_rate: float,
) -> float:
    gross = side * (exit_price / entry_price - 1.0)
    funding = side * sum(float(value) for value in funding_cashflows)
    entry_cost = cost_rate
    exit_cost = cost_rate * exit_price / entry_price
    return gross - funding - entry_cost - exit_cost


def strict_trade_path(
    market: pd.DataFrame,
    *,
    side: int,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    funding: Sequence[Mapping[str, Any]],
    start_equity: float,
    cost_rate: float,
    prior_peak: float | None = None,
    prior_drawdown: float = 0.0,
) -> dict[str, Any]:
    rows = market.loc[
        market["date"].ge(entry_time) & market["date"].lt(exit_time)
    ].copy()
    exit_rows = market.loc[market["date"].eq(exit_time)]
    if rows.empty or exit_rows.empty:
        raise RuntimeError("trade interval escapes exact market grid")
    entry_price = float(rows.iloc[0]["open"])
    exit_price = float(exit_rows.iloc[0]["open"])
    quantity = side * start_equity / entry_price
    entry_cost = abs(quantity) * entry_price * cost_rate
    funding_paid = 0.0
    peak = max(start_equity, prior_peak or start_equity)
    max_drawdown = float(prior_drawdown)
    events = {
        pd.Timestamp(row["funding_time"]): row
        for row in funding
        if entry_time <= pd.Timestamp(row["funding_time"]) < exit_time
    }

    def equity_at(price: float) -> float:
        return (
            start_equity
            - entry_cost
            + quantity * (price - entry_price)
            - funding_paid
        )

    opening_equity = start_equity - entry_cost
    max_drawdown = max(max_drawdown, 1.0 - opening_equity / peak)
    for row in rows.itertuples(index=False):
        timestamp = pd.Timestamp(row.date)
        event = events.get(timestamp)
        if event is not None:
            funding_paid += (
                quantity
                * float(event["settlement_mark_price"])
                * float(event["funding_rate"])
            )
            marked = equity_at(float(event["settlement_mark_price"]))
            max_drawdown = max(max_drawdown, 1.0 - marked / peak)
        favorable = float(row.high if side > 0 else row.low)
        adverse = float(row.low if side > 0 else row.high)
        peak = max(peak, equity_at(favorable))
        max_drawdown = max(max_drawdown, 1.0 - equity_at(adverse) / peak)
        peak = max(peak, equity_at(float(row.close)))
    exit_cost = abs(quantity) * exit_price * cost_rate
    end_equity = equity_at(exit_price) - exit_cost
    max_drawdown = max(max_drawdown, 1.0 - end_equity / peak)
    peak = max(peak, end_equity)
    return {
        "end_equity": end_equity,
        "peak": peak,
        "strict_drawdown": max_drawdown,
        "net_return": end_equity / start_equity - 1.0,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "funding_cashflow": -funding_paid,
    }


def _signals_2022(candidate_id: str) -> list[dict[str, Any]]:
    slow_floor, gap = source._candidate_parameters(candidate_id)
    cards = [
        card
        for card in source._load_cards()
        if str(card["decision_at"]).startswith("2022")
    ]
    state = source.EwmaState()
    skip_next = False
    accepted: list[dict[str, Any]] = []
    for card in cards:
        score, _ = source.daily_disagreement(card)
        source.update_ewmas(state, score)
        signal = source.signal_for(state, slow_floor=slow_floor, gap=gap)
        if signal == "flat":
            if skip_next:
                skip_next = False
            continue
        if skip_next:
            skip_next = False
            continue
        decision = pd.Timestamp(card["decision_at"])
        entry = decision + pd.Timedelta(minutes=5)
        accepted.append(
            {
                "decision_at": str(card["decision_at"]),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(days=1),
                "direction": signal,
                "side": 1 if signal == "long" else -1,
            }
        )
        skip_next = True
    manifest = prereg.canonical_hash(
        [
            {
                "decision_at": row["decision_at"],
                "direction": row["direction"],
            }
            for row in accepted
        ]
    )
    expected = _read_json(SOURCE_RESULT)["family_source_incidence"][
        candidate_id
    ]["2022"]["accepted_signal_manifest_hash"]
    if manifest != expected:
        raise RuntimeError("2022 signal reconstruction changed")
    return [
        row
        for row in accepted
        if row["exit_time"] <= pd.Timestamp("2023-01-01", tz="UTC")
    ]


def _load_2022_market() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    bindings = []
    for month in range(1, 13):
        stem = f"BTCUSDT_5m_2022-{month:02d}"
        payload_path = MONTHLY_MARKET_DIR / f"{stem}.csv.gz"
        manifest_path = MONTHLY_MARKET_DIR / f"{stem}.json"
        manifest = _read_json(manifest_path)
        observed_hash = _sha256_file(payload_path)
        if (
            manifest.get("output_sha256") != observed_hash
            or manifest.get("rows") is None
        ):
            raise RuntimeError(f"2022 monthly market authority drift: {stem}")
        frame = pd.read_csv(REPO_ROOT / payload_path, compression="gzip")
        if len(frame) != int(manifest["rows"]):
            raise RuntimeError(f"2022 monthly market row drift: {stem}")
        frames.append(frame)
        bindings.append(
            {
                "path": payload_path.as_posix(),
                "sha256": observed_hash,
                "rows": len(frame),
            }
        )
    market = pd.concat(frames, ignore_index=True)
    market["date"] = pd.to_datetime(market["date"], utc=True)
    expected = pd.date_range(
        "2022-01-01",
        "2022-12-31 23:55:00",
        freq="5min",
        tz="UTC",
    )
    if not pd.DatetimeIndex(market["date"]).equals(expected):
        raise RuntimeError("2022 market grid is not exact")
    return market, bindings


def _load_2022_funding() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(REPO_ROOT / prereg.FUNDING, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = str(row["funding_time_utc"])
            if timestamp.startswith("2023"):
                break
            if not timestamp.startswith("2022"):
                continue
            rows.append(
                {
                    "funding_time": pd.Timestamp(row["mark_open_time_utc"]),
                    "funding_rate": float(row["funding_rate"]),
                    "settlement_mark_price": float(
                        row["settlement_mark_price"]
                    ),
                }
            )
    expected = pd.date_range(
        "2022-01-01", "2022-12-31 16:00:00", freq="8h", tz="UTC"
    )
    if [row["funding_time"] for row in rows] != list(expected):
        raise RuntimeError("2022 exact funding grid changed")
    return rows


def _evaluate(
    market: pd.DataFrame,
    funding: Sequence[Mapping[str, Any]],
    signals: Sequence[Mapping[str, Any]],
    *,
    cost_rate: float,
) -> dict[str, Any]:
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    trades = []
    for signal in signals:
        path = strict_trade_path(
            market,
            side=int(signal["side"]),
            entry_time=pd.Timestamp(signal["entry_time"]),
            exit_time=pd.Timestamp(signal["exit_time"]),
            funding=funding,
            start_equity=equity,
            cost_rate=cost_rate,
            prior_peak=peak,
            prior_drawdown=drawdown,
        )
        equity = float(path["end_equity"])
        peak = float(path["peak"])
        drawdown = float(path["strict_drawdown"])
        trades.append(
            {
                "entry_time": pd.Timestamp(signal["entry_time"]).isoformat(),
                "exit_time": pd.Timestamp(signal["exit_time"]).isoformat(),
                "direction": signal["direction"],
                "net_return": path["net_return"],
            }
        )
    years = 365.0 / 365.2425
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0 else -1.0
    return {
        "absolute_return": equity - 1.0,
        "cagr": cagr,
        "strict_mdd": drawdown,
        "cagr_to_strict_mdd": (
            cagr / drawdown if drawdown > 1e-12 else 0.0
        ),
        "final_equity": equity,
        "closed_trades": len(trades),
        "long_trades": sum(row["direction"] == "long" for row in trades),
        "short_trades": sum(row["direction"] == "short" for row in trades),
        "trade_manifest_hash": prereg.canonical_hash(trades),
    }


def rank_candidates(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["base"]["cagr_to_strict_mdd"]),
            -float(row["stress"]["absolute_return"]),
            -int(row["closed_trades"]),
            str(row["candidate_id"]),
        ),
    )


def execute_selection() -> dict[str, Any]:
    attempt = _read_json(ATTEMPT_PATH)
    if (
        attempt.get("runner", {}).get("sha256") != _sha256_file(RUNNER_PATH)
        or attempt.get("attempt_hash")
        != prereg.canonical_hash(
            {k: v for k, v in attempt.items() if k != "attempt_hash"}
        )
    ):
        raise RuntimeError("2022 selection attempt drift")
    market, market_bindings = _load_2022_market()
    funding = _load_2022_funding()
    candidates = []
    for candidate_id in attempt["source_support"]["eligible_candidate_ids"]:
        signals = _signals_2022(candidate_id)
        base = _evaluate(market, funding, signals, cost_rate=0.0006)
        stress = _evaluate(market, funding, signals, cost_rate=0.0010)
        long_share = base["long_trades"] / max(base["closed_trades"], 1)
        short_share = base["short_trades"] / max(base["closed_trades"], 1)
        eligible = (
            base["closed_trades"] >= 20
            and long_share >= 0.20
            and short_share >= 0.20
            and base["absolute_return"] > 0.0
            and stress["absolute_return"] > 0.0
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "closed_trades": base["closed_trades"],
                "long_share": long_share,
                "short_share": short_share,
                "base": base,
                "stress": stress,
                "selection_eligible": eligible,
            }
        )
    ranked = rank_candidates(
        [row for row in candidates if row["selection_eligible"]]
    )
    selected = ranked[0]["candidate_id"] if ranked else None
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "2022_selection",
        "attempt_hash": attempt["attempt_hash"],
        "source_result_hash": SOURCE_RESULT_HASH,
        "market_bindings": market_bindings,
        "funding_binding": {
            "path": prereg.FUNDING.as_posix(),
            "frozen_full_payload_sha256": prereg.FUNDING_SHA256,
            "rows_parsed_2022": len(funding),
            "rows_parsed_2023": 0,
            "stream_stopped_at_first_2023_timestamp_before_numeric_parse": True,
        },
        "candidate_metrics": candidates,
        "selected_top1": selected,
        "decision": "pass" if selected else "reject",
        "terminal_action": (
            "FREEZE_TOP1_BEFORE_UNTOUCHED_2023_VETO"
            if selected
            else "TERMINAL_REJECT_CDP1_2022_SELECTION_NO_REPAIR"
        ),
        "authorize_2023_future_veto": bool(selected),
        "2023_market_rows_parsed": 0,
        "2023_funding_numeric_rows_parsed": 0,
        "post_result_rank2_or_threshold_repair_allowed": False,
    }
    return {**core, "result_hash": prereg.canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare-attempt", "execute"), nargs="?", default="execute"
    )
    args = parser.parse_args()
    if args.action == "prepare-attempt":
        payload = prepare_attempt()
        print(json.dumps({"attempt_hash": payload["attempt_hash"]}, sort_keys=True))
        return
    result = execute_selection()
    _write_once(RESULT_PATH, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "selected_top1": result["selected_top1"],
                "result_hash": result["result_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
