"""Pure signed-net planning for a Binance account that remains in Hedge Mode.

The research ledger owns virtual positions per strategy sleeve.  Binance owns
only their signed sum: a positive sum is represented by the LONG hedge side, a
negative sum by the SHORT hedge side, and the opposite physical side must be
flat.  This module performs no network or filesystem IO so the risk and
recovery contracts can be tested independently from the live runner.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from decimal import ROUND_DOWN, Decimal
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

SIGNED_NET_HEDGE_MODE = "signed_net_hedge_v1"
SINGLE_SIDE_HEDGE_MODEL = "single_side_hedge_net"


def _decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:  # pragma: no cover - Decimal gives varied errors
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError(f"invalid {label}: {value!r}")
    return result


def _utc(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"invalid timestamp: {value!r}")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def digest(value: Any) -> str:
    body = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SignedNetPolicy:
    mode: str
    net_cap_after_fees: Decimal
    fee_rate: Decimal
    weights: Mapping[str, Decimal]


@dataclass(frozen=True)
class HedgeLeg:
    side: Literal["BUY", "SELL"]
    position_side: Literal["LONG", "SHORT"]
    quantity: Decimal
    reduce_position: bool


@dataclass(frozen=True)
class SignedNetPlan:
    plan_id: str
    policy_hash: str
    base_revision: int
    base_quantity: Decimal
    target_quantity: Decimal
    unrounded_target_quantity: Decimal
    proposed_dust_quantity: Decimal
    reference_price: Decimal
    equity_before: Decimal
    estimated_fee: Decimal
    resize_scale: Decimal
    quantity_step: Decimal
    execution_time: str
    legs: tuple[HedgeLeg, ...]
    proposed_open_sleeves: dict[str, dict[str, Any]]
    proposed_processed_signals: dict[str, str]
    proposed_processed_signal_ids: dict[str, list[str]]
    opened_sleeves: tuple[str, ...]
    closed_sleeves: tuple[str, ...]
    blocked_entries: tuple[tuple[str, str], ...]


def parse_policy(portfolio: Mapping[str, Any]) -> SignedNetPolicy | None:
    """Parse an explicit opt-in policy; legacy portfolios remain unchanged."""

    raw = portfolio.get("position_aggregation")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RuntimeError("position_aggregation must be an object")
    mode = str(raw.get("mode", "")).strip()
    if mode != SIGNED_NET_HEDGE_MODE:
        raise RuntimeError(f"unsupported position_aggregation.mode={mode or 'missing'}")
    invariants = {
        "cross_sleeve_overlap_allowed": True,
        "same_sleeve_overlap": "source_policy",
        "offset_opposite_sides_before_costs": True,
        "physical_position_model": SINGLE_SIDE_HEDGE_MODEL,
    }
    for key, expected in invariants.items():
        if raw.get(key) != expected:
            raise RuntimeError(
                f"position_aggregation.{key} must be {expected!r}, got {raw.get(key)!r}"
            )
    cap = _decimal(raw.get("net_cap_after_fees"), "net_cap_after_fees", positive=True)
    fee = _decimal(raw.get("fee_rate"), "fee_rate")
    if fee < 0 or cap * fee >= 1:
        raise RuntimeError("position_aggregation fee/cap combination is invalid")

    sleeves = portfolio.get("base_sleeves")
    if not isinstance(sleeves, list) or not sleeves:
        raise RuntimeError("signed-net portfolio requires non-empty base_sleeves")
    weights: dict[str, Decimal] = {}
    for sleeve in sleeves:
        name = str(sleeve.get("name", "")).strip()
        if not name or name in weights:
            raise RuntimeError(f"invalid or duplicate signed-net sleeve name: {name!r}")
        weight = _decimal(sleeve.get("weight"), f"weight[{name}]")
        if weight < 0:
            raise RuntimeError(f"signed-net sleeve weight must be non-negative: {name}")
        if weight > 0:
            weights[name] = weight
    if not weights:
        raise RuntimeError("signed-net portfolio weights sum to zero")
    return SignedNetPolicy(
        mode=mode,
        net_cap_after_fees=cap,
        fee_rate=fee,
        weights=weights,
    )


def policy_digest(policy: SignedNetPolicy) -> str:
    return digest(
        {
            "mode": policy.mode,
            "net_cap_after_fees": policy.net_cap_after_fees,
            "fee_rate": policy.fee_rate,
            "weights": dict(policy.weights),
        }
    )


def side_sign(side: Any) -> Decimal:
    value = str(side).upper()
    if value == "LONG":
        return Decimal("1")
    if value == "SHORT":
        return Decimal("-1")
    raise ValueError(f"invalid effective sleeve side: {side!r}")


def round_toward_zero(value: Decimal, step: Decimal) -> Decimal:
    step = _decimal(step, "quantity_step", positive=True)
    value = _decimal(value, "quantity")
    sign = Decimal("1") if value >= 0 else Decimal("-1")
    units = (abs(value) / step).to_integral_value(rounding=ROUND_DOWN)
    return sign * units * step


def signed_state_quantity(state: Mapping[str, Any]) -> Decimal:
    total = Decimal("0")
    sleeves = state.get("open_sleeves", {})
    if not isinstance(sleeves, Mapping):
        raise RuntimeError("open_sleeves must be an object")
    for name, sleeve in sleeves.items():
        if not isinstance(sleeve, Mapping):
            raise RuntimeError(f"invalid virtual sleeve state: {name}")
        quantity = _decimal(sleeve.get("quantity", "0"), f"quantity[{name}]")
        if quantity < 0:
            raise RuntimeError(f"virtual sleeve quantity must be non-negative: {name}")
        total += side_sign(sleeve.get("side")) * quantity
    total += _decimal(state.get("signed_net_dust_quantity", "0"), "signed_net_dust_quantity")
    return total


def migrate_policy_subset(
    state: Mapping[str, Any],
    *,
    old_policy: SignedNetPolicy,
    new_policy: SignedNetPolicy,
    retired_sleeves: Sequence[str] | set[str],
    physical_quantity: Decimal,
    quantity_step: Decimal,
    migrated_at: Any,
    reason: str,
) -> dict[str, Any]:
    """Rebind an active ledger after removing only inactive policy sleeves.

    This deliberately supports only a strict, weight-preserving subset.  It
    never closes, opens, or resizes exposure and therefore requires the
    virtual ledger (including dust) to already match the exchange position.
    """

    if old_policy.mode != new_policy.mode:
        raise RuntimeError("signed-net policy migration cannot change aggregation mode")
    if (
        old_policy.net_cap_after_fees != new_policy.net_cap_after_fees
        or old_policy.fee_rate != new_policy.fee_rate
    ):
        raise RuntimeError("signed-net policy migration cannot change risk terms")

    result = copy.deepcopy(dict(state))
    if result.get("position_aggregation_mode") != old_policy.mode:
        raise RuntimeError("state aggregation mode does not match the old signed-net policy")
    if result.get("pending_signed_net_transition") is not None:
        raise RuntimeError("pending signed-net transition blocks policy migration")

    old_hash = policy_digest(old_policy)
    new_hash = policy_digest(new_policy)
    if result.get("signed_net_policy_hash") != old_hash:
        raise RuntimeError("state policy hash does not match the old signed-net policy")

    old_names = set(old_policy.weights)
    new_names = set(new_policy.weights)
    if not new_names < old_names:
        raise RuntimeError("new signed-net policy must be a strict subset of the old policy")
    removed = old_names - new_names
    expected_removed = {str(name).strip() for name in retired_sleeves if str(name).strip()}
    if removed != expected_removed:
        raise RuntimeError(
            "retired signed-net sleeves do not exactly match the policy subset: "
            f"removed={sorted(removed)} requested={sorted(expected_removed)}"
        )
    for name in sorted(new_names):
        if old_policy.weights[name] != new_policy.weights[name]:
            raise RuntimeError(f"retained signed-net weight changed for sleeve={name}")

    open_sleeves = result.get("open_sleeves", {})
    if not isinstance(open_sleeves, Mapping):
        raise RuntimeError("open_sleeves must be an object")
    retired_open = sorted(set(open_sleeves) & removed)
    if retired_open:
        raise RuntimeError(
            "retired virtual sleeves are still open; close them under the old policy: "
            + ",".join(retired_open)
        )
    unknown_open = sorted(set(open_sleeves) - new_names)
    if unknown_open:
        raise RuntimeError(
            "open virtual sleeves are absent from the new signed-net policy: "
            + ",".join(unknown_open)
        )

    step = _decimal(quantity_step, "quantity_step", positive=True)
    physical = _decimal(physical_quantity, "physical_quantity")
    tolerance = max(step / Decimal("10"), Decimal("0.0000000001"))
    if abs(physical - round_toward_zero(physical, step)) > tolerance:
        raise RuntimeError("physical quantity is not aligned with the exchange lot step")
    virtual = signed_state_quantity(result)
    if abs(virtual - physical) > tolerance:
        raise RuntimeError(
            "physical/virtual signed quantity mismatch; policy migration refused: "
            f"state={virtual} physical={physical}"
        )

    migration_time = str(_utc(migrated_at))
    migration_reason = str(reason).strip()
    if not migration_reason:
        raise RuntimeError("signed-net policy migration reason is required")
    raw_history = result.get("signed_net_policy_migration_history", [])
    if not isinstance(raw_history, list):
        raise RuntimeError("signed_net_policy_migration_history must be a list")
    history = copy.deepcopy(raw_history)
    history.append(
        {
            "from_policy_hash": old_hash,
            "to_policy_hash": new_hash,
            "retired_sleeves": sorted(removed),
            "reason": migration_reason,
            "migrated_at": migration_time,
            "physical_quantity": str(physical),
            "open_sleeves": sorted(open_sleeves),
        }
    )
    result["signed_net_policy_migration_history"] = history[-100:]
    result["signed_net_policy_hash"] = new_hash
    result["signed_net_revision"] = int(result.get("signed_net_revision", 0) or 0) + 1
    result["updated_at"] = migration_time
    return result


def physical_signed_quantity(
    positions: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    tolerance: Decimal = Decimal("0.00000001"),
) -> Decimal:
    """Return one signed physical quantity and reject ambiguous ownership."""

    long_quantity = Decimal("0")
    short_quantity = Decimal("0")
    one_way_quantity = Decimal("0")
    for position in positions:
        if str(position.get("symbol", symbol)).upper() != str(symbol).upper():
            continue
        amount = _decimal(position.get("positionAmt", "0"), "positionAmt")
        if abs(amount) <= tolerance:
            continue
        position_side = str(position.get("positionSide", "BOTH")).upper()
        if position_side == "LONG":
            long_quantity += abs(amount)
        elif position_side == "SHORT":
            short_quantity += abs(amount)
        elif position_side == "BOTH":
            one_way_quantity += amount
        else:
            raise RuntimeError(f"unknown Binance positionSide={position_side!r}")
    if abs(one_way_quantity) > tolerance:
        raise RuntimeError("signed-net hedge executor requires Hedge Mode, not a BOTH position")
    if long_quantity > tolerance and short_quantity > tolerance:
        raise RuntimeError(
            "both LONG and SHORT physical hedge sides are open; automatic attribution is unsafe"
        )
    if long_quantity > tolerance:
        return long_quantity
    if short_quantity > tolerance:
        return -short_quantity
    return Decimal("0")


def build_hedge_legs(
    current: Decimal,
    target: Decimal,
    *,
    quantity_step: Decimal,
) -> tuple[HedgeLeg, ...]:
    """Translate a signed delta into safe Hedge Mode legs.

    A sign flip always closes the existing physical side before opening the
    opposite one.  Binance Hedge Mode uses the explicit position side rather
    than the reduceOnly flag; ``reduce_position`` records intent for auditing.
    """

    step = _decimal(quantity_step, "quantity_step", positive=True)
    current = round_toward_zero(current, step)
    target = round_toward_zero(target, step)
    if current == target:
        return ()

    legs: list[HedgeLeg] = []

    def append(quantity: Decimal, *, side: str, position_side: str, reduce: bool) -> None:
        aligned = round_toward_zero(abs(quantity), step)
        if aligned <= 0:
            return
        legs.append(
            HedgeLeg(
                side=side,  # type: ignore[arg-type]
                position_side=position_side,  # type: ignore[arg-type]
                quantity=aligned,
                reduce_position=reduce,
            )
        )

    if current > 0 and target < 0:
        append(current, side="SELL", position_side="LONG", reduce=True)
        append(abs(target), side="SELL", position_side="SHORT", reduce=False)
    elif current < 0 and target > 0:
        append(abs(current), side="BUY", position_side="SHORT", reduce=True)
        append(target, side="BUY", position_side="LONG", reduce=False)
    elif current >= 0 and target >= 0:
        if target > current:
            append(target - current, side="BUY", position_side="LONG", reduce=False)
        else:
            append(current - target, side="SELL", position_side="LONG", reduce=True)
    else:
        current_abs = abs(current)
        target_abs = abs(target)
        if target_abs > current_abs:
            append(target_abs - current_abs, side="SELL", position_side="SHORT", reduce=False)
        else:
            append(current_abs - target_abs, side="BUY", position_side="SHORT", reduce=True)
    return tuple(legs)


def _append_processed(
    processed: dict[str, str],
    history: dict[str, list[str]],
    *,
    name: str,
    signal_id: str,
) -> None:
    processed[name] = signal_id
    values = history.setdefault(name, [])
    if signal_id not in values:
        values.append(signal_id)
    if len(values) > 2_000:
        del values[:-2_000]


def _plan_payload(plan: SignedNetPlan, *, include_id: bool) -> dict[str, Any]:
    payload = {
        "policy_hash": plan.policy_hash,
        "base_revision": plan.base_revision,
        "base_quantity": plan.base_quantity,
        "target_quantity": plan.target_quantity,
        "unrounded_target_quantity": plan.unrounded_target_quantity,
        "proposed_dust_quantity": plan.proposed_dust_quantity,
        "reference_price": plan.reference_price,
        "equity_before": plan.equity_before,
        "estimated_fee": plan.estimated_fee,
        "resize_scale": plan.resize_scale,
        "quantity_step": plan.quantity_step,
        "execution_time": plan.execution_time,
        "legs": [leg.__dict__ for leg in plan.legs],
        "proposed_open_sleeves": plan.proposed_open_sleeves,
        "proposed_processed_signals": plan.proposed_processed_signals,
        "proposed_processed_signal_ids": plan.proposed_processed_signal_ids,
        "opened_sleeves": plan.opened_sleeves,
        "closed_sleeves": plan.closed_sleeves,
        "blocked_entries": plan.blocked_entries,
    }
    if include_id:
        payload["plan_id"] = plan.plan_id
    return _jsonable(payload)


def build_plan(
    *,
    policy: SignedNetPolicy,
    state: Mapping[str, Any],
    current_physical_quantity: Decimal,
    entry_scores: Sequence[Mapping[str, Any]],
    close_reasons: Mapping[str, str],
    equity: Decimal,
    reference_price: Decimal,
    quantity_step: Decimal,
    interval_minutes: int,
    execution_time: Any,
) -> SignedNetPlan:
    """Build one atomic virtual-ledger transition and aggregate broker delta."""

    if state.get("position_aggregation_mode") != policy.mode:
        raise RuntimeError("state is not initialized for signed-net hedge execution")
    equity = _decimal(equity, "equity", positive=True)
    price = _decimal(reference_price, "reference_price", positive=True)
    step = _decimal(quantity_step, "quantity_step", positive=True)
    current = _decimal(current_physical_quantity, "current_physical_quantity")
    if current != round_toward_zero(current, step):
        raise RuntimeError("physical quantity is not aligned with the exchange lot step")
    state_quantity = signed_state_quantity(state)
    tolerance = max(step / Decimal("10"), Decimal("0.0000000001"))
    if abs(state_quantity - current) > tolerance:
        raise RuntimeError(
            "physical/virtual signed quantity mismatch; automatic attribution refused: "
            f"state={state_quantity} physical={current}"
        )

    existing = copy.deepcopy(dict(state.get("open_sleeves", {})))
    unknown = set(existing) - set(policy.weights)
    if unknown:
        raise RuntimeError(f"unapproved virtual sleeves in state: {sorted(unknown)}")
    proposed = copy.deepcopy(existing)
    closed: list[str] = []
    for name, reason in close_reasons.items():
        if name not in existing:
            raise RuntimeError(f"cannot close unknown virtual sleeve: {name}")
        proposed.pop(name, None)
        closed.append(name)

    processed = copy.deepcopy(dict(state.get("processed_signals", {})))
    raw_history = state.get("processed_signal_ids", {})
    if not isinstance(raw_history, Mapping):
        raise RuntimeError("processed_signal_ids must be an object")
    history = {str(name): list(values) for name, values in raw_history.items()}
    if any(
        not isinstance(values, list)
        or any(not isinstance(signal_id, str) or not signal_id for signal_id in values)
        for values in history.values()
    ):
        raise RuntimeError("processed_signal_ids contains invalid history")

    opened: list[str] = []
    blocked: list[tuple[str, str]] = []
    seen: set[str] = set()
    now = _utc(execution_time)
    for raw in entry_scores:
        name = str(raw.get("name", ""))
        if name in seen:
            raise RuntimeError(f"duplicate sleeve score in one decision: {name}")
        seen.add(name)
        if name not in policy.weights:
            raise RuntimeError(f"unknown signed-net sleeve score: {name}")
        kind = str(raw.get("kind", "entry"))
        if kind not in {"entry", "target"}:
            raise RuntimeError(f"unsupported signed-net signal kind={kind!r}")
        ready = raw.get("ready", True)
        if not isinstance(ready, bool):
            raise RuntimeError(f"signal readiness must be boolean for sleeve={name}")
        if kind == "entry" and not isinstance(raw.get("active"), bool):
            raise RuntimeError(f"entry active must be boolean for sleeve={name}")
        scheduled = _utc(raw.get("execution_time", now))
        if scheduled.floor(f"{int(interval_minutes)}min") != now.floor(
            f"{int(interval_minutes)}min"
        ):
            raise RuntimeError(
                f"signal execution clock mismatch for sleeve={name}: "
                f"signal={scheduled} decision={now}"
            )
        signal_id = str(raw.get("signal_id", ""))
        if not signal_id:
            raise RuntimeError(f"signal identity required for sleeve={name}")
        if signal_id in history.get(name, []):
            blocked.append((name, "signal_already_processed"))
            continue
        score_weight = _decimal(raw.get("weight"), f"score weight[{name}]")
        if score_weight != policy.weights[name]:
            raise RuntimeError(
                f"score/config weight mismatch for {name}: {score_weight} != {policy.weights[name]}"
            )
        # Freeze every observed decision, including no-trade and unavailable
        # decisions.  A later source repair must not turn an already-observed
        # historical signal into a live catch-up order.
        _append_processed(processed, history, name=name, signal_id=signal_id)
        if not ready:
            blocked.append((name, "signal_not_ready"))
            continue
        if kind == "entry" and not raw["active"]:
            continue
        if kind == "target":
            if name != "macro_flow":
                raise RuntimeError("only macro_flow may submit target maintenance signals")
            fraction = _decimal(raw.get("target_fraction"), "macro target fraction")
            if abs(fraction) > 1:
                raise RuntimeError("macro target fraction exceeds source limit")
            existed = name in proposed
            if fraction == 0:
                proposed.pop(name, None)
                if existed and name not in closed:
                    closed.append(name)
            else:
                side = "LONG" if fraction > 0 else "SHORT"
                quantity = policy.weights[name] * abs(fraction) * equity / price
                proposed[name] = {
                    "name": name,
                    "side": side,
                    "quantity": str(quantity),
                    "weight": str(policy.weights[name]),
                    "target_fraction": str(fraction),
                    "signal_id": signal_id,
                    "signal_date": str(_utc(raw.get("date", now))),
                    "exit_at": None,
                    "entry_reference_price": float(price),
                    "entry_fill_price": float(price),
                    "entry_filled_at": str(now),
                    "allocation_mode": "signed_net_research_notional",
                    "dynamic_exit": None,
                    "barrier_exit": None,
                    "policy_metadata": copy.deepcopy(raw.get("policy_metadata")),
                }
                if not existed:
                    opened.append(name)
            continue
        if name in existing:
            blocked.append((name, "same_sleeve_open_at_decision"))
            continue
        side = str(raw.get("side", "")).upper()
        sign = side_sign(side)
        if name == "dollar_rally_short" and side != "SHORT":
            raise RuntimeError("dollar_rally_short must produce an effective SHORT signal")
        hold_bars = int(raw.get("hold_bars", 0))
        if hold_bars <= 0 or Decimal(str(raw.get("hold_bars"))) != hold_bars:
            raise RuntimeError(f"invalid hold_bars for sleeve={name}")
        signal_date = _utc(raw.get("date"))
        exit_at = signal_date + pd.Timedelta(
            minutes=int(interval_minutes) * (1 + hold_bars)
        )
        quantity = policy.weights[name] * equity / price
        proposed[name] = {
            "name": name,
            "side": side,
            "quantity": str(abs(sign * quantity)),
            "weight": str(policy.weights[name]),
            "signal_id": signal_id,
            "signal_date": str(signal_date),
            "exit_at": str(exit_at),
            "entry_reference_price": float(price),
            "entry_fill_price": float(price),
            "entry_filled_at": str(now),
            "allocation_mode": "signed_net_research_notional",
            "dynamic_exit": copy.deepcopy(raw.get("dynamic_exit")),
            "barrier_exit": copy.deepcopy(raw.get("barrier_exit")),
            "policy_metadata": copy.deepcopy(raw.get("policy_metadata")),
            "barrier_stream_session_id": raw.get("barrier_stream_session_id"),
            "barrier_stream_gap_count": raw.get("barrier_stream_gap_count"),
        }
        opened.append(name)

    desired = sum(
        side_sign(sleeve.get("side"))
        * _decimal(sleeve.get("quantity"), f"quantity[{name}]")
        for name, sleeve in proposed.items()
    )
    aligned = (Decimal("1") if desired >= 0 else Decimal("-1")) * current * price
    cap = policy.net_cap_after_fees
    fee = policy.fee_rate
    budget = cap * equity
    fee_cap = cap * fee
    if budget >= aligned:
        allowed_notional = (budget + fee_cap * aligned) / (Decimal("1") + fee_cap)
    else:
        allowed_notional = (budget - fee_cap * aligned) / (Decimal("1") - fee_cap)
    desired_notional = abs(desired * price)
    if desired_notional <= Decimal("0"):
        resize_scale = Decimal("1")
    else:
        resize_scale = min(
            Decimal("1"),
            max(Decimal("0"), allowed_notional) / desired_notional,
        )
    if resize_scale < 1:
        for sleeve in proposed.values():
            resized = _decimal(sleeve["quantity"], "virtual quantity") * resize_scale
            sleeve["quantity"] = str(resized)
            sleeve["net_cap_resize_scale"] = str(resize_scale)

    unrounded = sum(
        side_sign(sleeve.get("side"))
        * _decimal(sleeve.get("quantity"), f"quantity[{name}]")
        for name, sleeve in proposed.items()
    )
    target = round_toward_zero(unrounded, step)
    dust = target - unrounded
    estimated_fee = abs(target - current) * price * fee
    if equity <= estimated_fee:
        raise RuntimeError("estimated transition fee consumes account equity")
    if abs(target * price) > cap * (equity - estimated_fee) + Decimal("0.00000001"):
        raise RuntimeError("post-fee net exposure cap check failed")
    legs = build_hedge_legs(current, target, quantity_step=step)
    base_revision = int(state.get("signed_net_revision", 0) or 0)
    execution_time_text = str(now)
    provisional = SignedNetPlan(
        plan_id="",
        policy_hash=policy_digest(policy),
        base_revision=base_revision,
        base_quantity=current,
        target_quantity=target,
        unrounded_target_quantity=unrounded,
        proposed_dust_quantity=dust,
        reference_price=price,
        equity_before=equity,
        estimated_fee=estimated_fee,
        resize_scale=resize_scale,
        quantity_step=step,
        execution_time=execution_time_text,
        legs=legs,
        proposed_open_sleeves=proposed,
        proposed_processed_signals=processed,
        proposed_processed_signal_ids=history,
        opened_sleeves=tuple(opened),
        closed_sleeves=tuple(closed),
        blocked_entries=tuple(blocked),
    )
    plan_id = digest(_plan_payload(provisional, include_id=False))
    return SignedNetPlan(**{**provisional.__dict__, "plan_id": plan_id})


def retarget_plan_for_risk_reduction(
    plan: SignedNetPlan,
    target_quantity: Decimal,
    *,
    fee_rate: Decimal,
) -> SignedNetPlan:
    """Return an integrity-checked plan with a strictly safer physical target.

    Virtual sleeve quantities remain frozen.  The resulting difference is an
    explicit aggregate dust/risk-adjustment quantity, so the committed ledger
    still equals the authoritative physical position without inventing sleeve
    fills or silently changing their approved coefficients.
    """

    target = round_toward_zero(
        _decimal(target_quantity, "risk-reduction target"),
        plan.quantity_step,
    )
    old = plan.target_quantity
    tolerance = max(plan.quantity_step / Decimal("10"), Decimal("0.0000000001"))
    if abs(target) > abs(old) + tolerance:
        raise RuntimeError(
            f"risk correction may not increase exposure: old={old} target={target}"
        )
    if old != 0 and target != 0 and (old > 0) != (target > 0):
        raise RuntimeError(
            f"risk correction may not flip direction: old={old} target={target}"
        )
    fee = _decimal(fee_rate, "fee_rate")
    if fee < 0:
        raise RuntimeError("fee_rate must be non-negative")
    additional_fee = abs(target - old) * plan.reference_price * fee
    provisional = replace(
        plan,
        plan_id="",
        target_quantity=target,
        proposed_dust_quantity=target - plan.unrounded_target_quantity,
        estimated_fee=plan.estimated_fee + additional_fee,
        legs=build_hedge_legs(
            plan.base_quantity,
            target,
            quantity_step=plan.quantity_step,
        ),
    )
    return replace(
        provisional,
        plan_id=digest(_plan_payload(provisional, include_id=False)),
    )


def apply_plan(state: Mapping[str, Any], plan: SignedNetPlan) -> dict[str, Any]:
    """Commit a plan after an authoritative physical target reconciliation."""

    if plan.plan_id != digest(_plan_payload(plan, include_id=False)):
        raise RuntimeError("signed-net plan integrity check failed")
    revision = int(state.get("signed_net_revision", 0) or 0)
    if revision != plan.base_revision:
        raise RuntimeError(
            f"signed-net state revision changed: {revision} != {plan.base_revision}"
        )
    result = copy.deepcopy(dict(state))
    result["position_aggregation_mode"] = SIGNED_NET_HEDGE_MODE
    result["open_sleeves"] = copy.deepcopy(plan.proposed_open_sleeves)
    result["processed_signals"] = copy.deepcopy(plan.proposed_processed_signals)
    result["processed_signal_ids"] = copy.deepcopy(plan.proposed_processed_signal_ids)
    result["signed_net_dust_quantity"] = str(plan.proposed_dust_quantity)
    result["signed_net_revision"] = revision + 1
    result["last_signed_net_plan_id"] = plan.plan_id
    result["last_signed_net_execution_time"] = plan.execution_time
    result.pop("pending_signed_net_transition", None)
    return result


def plan_to_dict(plan: SignedNetPlan) -> dict[str, Any]:
    return _plan_payload(plan, include_id=True)


def plan_from_dict(payload: Mapping[str, Any]) -> SignedNetPlan:
    legs = tuple(
        HedgeLeg(
            side=str(item["side"]).upper(),  # type: ignore[arg-type]
            position_side=str(item["position_side"]).upper(),  # type: ignore[arg-type]
            quantity=_decimal(item["quantity"], "leg quantity", positive=True),
            reduce_position=bool(item["reduce_position"]),
        )
        for item in payload.get("legs", [])
    )
    plan = SignedNetPlan(
        plan_id=str(payload.get("plan_id", "")),
        policy_hash=str(payload["policy_hash"]),
        base_revision=int(payload["base_revision"]),
        base_quantity=_decimal(payload["base_quantity"], "base quantity"),
        target_quantity=_decimal(payload["target_quantity"], "target quantity"),
        unrounded_target_quantity=_decimal(
            payload["unrounded_target_quantity"], "unrounded target quantity"
        ),
        proposed_dust_quantity=_decimal(
            payload["proposed_dust_quantity"], "proposed dust quantity"
        ),
        reference_price=_decimal(payload["reference_price"], "reference price", positive=True),
        equity_before=_decimal(payload["equity_before"], "equity", positive=True),
        estimated_fee=_decimal(payload["estimated_fee"], "estimated fee"),
        resize_scale=_decimal(payload["resize_scale"], "resize scale"),
        quantity_step=_decimal(payload["quantity_step"], "quantity step", positive=True),
        execution_time=str(payload["execution_time"]),
        legs=legs,
        proposed_open_sleeves=copy.deepcopy(dict(payload["proposed_open_sleeves"])),
        proposed_processed_signals=copy.deepcopy(
            dict(payload["proposed_processed_signals"])
        ),
        proposed_processed_signal_ids={
            str(name): list(values)
            for name, values in dict(payload["proposed_processed_signal_ids"]).items()
        },
        opened_sleeves=tuple(str(value) for value in payload.get("opened_sleeves", [])),
        closed_sleeves=tuple(str(value) for value in payload.get("closed_sleeves", [])),
        blocked_entries=tuple(
            (str(item[0]), str(item[1])) for item in payload.get("blocked_entries", [])
        ),
    )
    if plan.plan_id != digest(_plan_payload(plan, include_id=False)):
        raise RuntimeError("persisted signed-net plan integrity check failed")
    return plan


def classify_pending_position(
    observed: Decimal,
    base: Decimal,
    target: Decimal,
    *,
    tolerance: Decimal = Decimal("0.00000001"),
) -> Literal["base", "target", "intermediate"]:
    observed = _decimal(observed, "observed quantity")
    base = _decimal(base, "base quantity")
    target = _decimal(target, "target quantity")
    if abs(observed - base) <= tolerance:
        return "base"
    if abs(observed - target) <= tolerance:
        return "target"
    if min(base, target) - tolerance <= observed <= max(base, target) + tolerance:
        return "intermediate"
    raise RuntimeError(
        "physical quantity is outside pending transition path; refusing recovery: "
        f"observed={observed} base={base} target={target}"
    )
