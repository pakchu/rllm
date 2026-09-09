from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from execution.portfolio_live import (
    _apply_signed_net_transition,
    _confirm_market_order,
    _enforce_signed_net_post_fill_cap,
    _execute_signed_net_plan,
    _finalize_signed_net_plan,
    _initialize_signed_net_state,
    _load_state,
    _resume_signed_net_transition,
    _run_signed_net_decision,
    _signed_net_eligible_entry_scores,
    _validate_state_aggregation_mode,
    _wait_with_signed_net_barrier_monitor,
)
from execution.signed_net_hedge import (
    SIGNED_NET_HEDGE_MODE,
    apply_plan,
    build_plan,
    classify_pending_position,
    migrate_policy_subset,
    parse_policy,
    physical_signed_quantity,
    plan_to_dict,
    policy_digest,
    retarget_plan_for_risk_reduction,
    signed_state_quantity,
)
from execution.wave_execution import WaveExecutionConfig


def _portfolio(*, cap: float = 4.5) -> dict:
    return {
        "base_sleeves": [
            {"name": "long", "side": "LONG", "weight": 1.0},
            {"name": "dollar_rally_short", "side": "SHORT", "weight": 0.5},
        ],
        "position_aggregation": {
            "mode": SIGNED_NET_HEDGE_MODE,
            "cross_sleeve_overlap_allowed": True,
            "same_sleeve_overlap": "source_policy",
            "offset_opposite_sides_before_costs": True,
            "net_cap_after_fees": cap,
            "fee_rate": 0.0006,
            "physical_position_model": "single_side_hedge_net",
        },
    }


def _score(name: str, side: str, weight: float, *, signal_id: str | None = None) -> dict:
    return {
        "name": name,
        "side": side,
        "weight": weight,
        "active": True,
        "ready": True,
        "signal_id": signal_id or f"{name}:1",
        "date": "2026-09-04T00:00:00Z",
        "current_close": 100.0,
        "hold_bars": 143,
        "dynamic_exit": None,
        "barrier_exit": None,
        "policy_metadata": {},
    }


def _empty_state() -> dict:
    return {
        "position_aggregation_mode": SIGNED_NET_HEDGE_MODE,
        "open_sleeves": {},
        "processed_signals": {},
        "processed_signal_ids": {},
        "signed_net_dust_quantity": "0",
        "signed_net_revision": 0,
    }


def test_policy_subset_migration_preserves_retained_exposure_and_dust():
    old_portfolio = _portfolio()
    old_portfolio["base_sleeves"].append(
        {"name": "macro_flow", "side": "AUTO", "weight": 1.0}
    )
    old_policy = parse_policy(old_portfolio)
    new_portfolio = _portfolio()
    new_portfolio["base_sleeves"] = [
        {"name": "long", "side": "LONG", "weight": 1.0}
    ]
    new_policy = parse_policy(new_portfolio)
    assert old_policy is not None
    assert new_policy is not None
    state = _empty_state()
    state["signed_net_policy_hash"] = policy_digest(old_policy)
    state["signed_net_revision"] = 7
    state["signed_net_dust_quantity"] = "-0.0004"
    state["open_sleeves"] = {
        "long": {
            "name": "long",
            "side": "LONG",
            "quantity": "0.0024",
            "signal_id": "long:old",
        }
    }

    migrated = migrate_policy_subset(
        state,
        old_policy=old_policy,
        new_policy=new_policy,
        retired_sleeves={"macro_flow", "dollar_rally_short"},
        physical_quantity=Decimal("0.002"),
        quantity_step=Decimal("0.001"),
        migrated_at="2026-09-09T01:00:00Z",
        reason="operator retired unrefined live alphas",
    )

    assert migrated["open_sleeves"] == state["open_sleeves"]
    assert migrated["signed_net_dust_quantity"] == "-0.0004"
    assert migrated["signed_net_revision"] == 8
    assert migrated["signed_net_policy_hash"] == policy_digest(new_policy)
    assert signed_state_quantity(migrated) == Decimal("0.0020")
    assert migrated["signed_net_policy_migration_history"][-1][
        "retired_sleeves"
    ] == ["dollar_rally_short", "macro_flow"]
    assert state["signed_net_policy_hash"] == policy_digest(old_policy)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("retired_open", "retired virtual sleeves are still open"),
        ("pending", "pending signed-net transition"),
        ("weight_drift", "retained signed-net weight changed"),
        ("physical_mismatch", "physical/virtual signed quantity mismatch"),
    ],
)
def test_policy_subset_migration_rejects_unsafe_state(mutation: str, message: str):
    old_portfolio = _portfolio()
    old_portfolio["base_sleeves"].append(
        {"name": "macro_flow", "side": "AUTO", "weight": 1.0}
    )
    old_policy = parse_policy(old_portfolio)
    new_portfolio = _portfolio()
    new_portfolio["base_sleeves"] = [
        {"name": "long", "side": "LONG", "weight": 1.0}
    ]
    if mutation == "weight_drift":
        new_portfolio["base_sleeves"][0]["weight"] = 1.1
    new_policy = parse_policy(new_portfolio)
    assert old_policy is not None
    assert new_policy is not None
    state = _empty_state()
    state["signed_net_policy_hash"] = policy_digest(old_policy)
    state["open_sleeves"] = {
        "long": {"side": "LONG", "quantity": "0.002"}
    }
    physical = Decimal("0.002")
    if mutation == "retired_open":
        state["open_sleeves"]["macro_flow"] = {
            "side": "SHORT",
            "quantity": "0.001",
        }
        physical = Decimal("0.001")
    elif mutation == "pending":
        state["pending_signed_net_transition"] = {"plan": {}}
    elif mutation == "physical_mismatch":
        physical = Decimal("0.003")

    with pytest.raises(RuntimeError, match=message):
        migrate_policy_subset(
            state,
            old_policy=old_policy,
            new_policy=new_policy,
            retired_sleeves={"macro_flow", "dollar_rally_short"},
            physical_quantity=physical,
            quantity_step=Decimal("0.001"),
            migrated_at="2026-09-09T01:00:00Z",
            reason="operator retired unrefined live alphas",
        )


def test_opposing_sleeves_net_before_one_physical_order():
    policy = parse_policy(_portfolio())
    plan = build_plan(
        policy=policy,
        state=_empty_state(),
        current_physical_quantity=Decimal("0"),
        entry_scores=[
            _score("long", "LONG", 1.0),
            _score("dollar_rally_short", "SHORT", 0.5),
        ],
        close_reasons={},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )

    assert plan.target_quantity == Decimal("5.000")
    assert plan.estimated_fee == Decimal("0.3000000")
    assert [(leg.side, leg.position_side, leg.quantity) for leg in plan.legs] == [
        ("BUY", "LONG", Decimal("5.000"))
    ]
    assert set(plan.proposed_open_sleeves) == {"long", "dollar_rally_short"}


def test_exact_offset_keeps_virtual_sleeves_without_broker_order():
    portfolio = _portfolio()
    portfolio["base_sleeves"][1]["weight"] = 1.0
    policy = parse_policy(portfolio)
    plan = build_plan(
        policy=policy,
        state=_empty_state(),
        current_physical_quantity=Decimal("0"),
        entry_scores=[
            _score("long", "LONG", 1.0),
            _score("dollar_rally_short", "SHORT", 1.0),
        ],
        close_reasons={},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )

    assert plan.target_quantity == 0
    assert plan.legs == ()
    committed = apply_plan(_empty_state(), plan)
    assert signed_state_quantity(committed) == 0
    assert set(committed["open_sleeves"]) == {"long", "dollar_rally_short"}


def test_sign_flip_closes_long_before_opening_short():
    state = _empty_state()
    state["open_sleeves"] = {
        "long": {
            "name": "long",
            "side": "LONG",
            "quantity": "2",
            "signal_id": "long:old",
            "entry_reference_price": 100.0,
            "exit_at": "2026-09-05T00:00:00Z",
        }
    }
    portfolio = _portfolio()
    portfolio["base_sleeves"][1]["weight"] = 0.5
    plan = build_plan(
        policy=parse_policy(portfolio),
        state=state,
        current_physical_quantity=Decimal("2"),
        entry_scores=[_score("dollar_rally_short", "SHORT", 0.5)],
        close_reasons={"long": "time_exit"},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )

    assert plan.target_quantity == Decimal("-5.000")
    assert [
        (leg.side, leg.position_side, leg.quantity, leg.reduce_position)
        for leg in plan.legs
    ] == [
        ("SELL", "LONG", Decimal("2.000"), True),
        ("SELL", "SHORT", Decimal("5.000"), False),
    ]


def test_simultaneous_close_and_open_are_planned_atomically():
    state = _empty_state()
    state["open_sleeves"] = {
        "dollar_rally_short": {
            "name": "dollar_rally_short",
            "side": "SHORT",
            "quantity": "5",
            "signal_id": "short:old",
            "entry_reference_price": 100.0,
            "exit_at": "2026-09-04T00:05:00Z",
        }
    }
    plan = build_plan(
        policy=parse_policy(_portfolio()),
        state=state,
        current_physical_quantity=Decimal("-5"),
        entry_scores=[_score("long", "LONG", 1.0)],
        close_reasons={"dollar_rally_short": "time_exit"},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )

    assert plan.target_quantity == Decimal("10.000")
    assert [(leg.side, leg.position_side, leg.quantity) for leg in plan.legs] == [
        ("BUY", "SHORT", Decimal("5.000")),
        ("BUY", "LONG", Decimal("10.000")),
    ]


def test_post_fee_cap_matches_approved_research_formula():
    portfolio = _portfolio()
    portfolio["base_sleeves"] = [{"name": "long", "side": "LONG", "weight": 5.0}]
    plan = build_plan(
        policy=parse_policy(portfolio),
        state=_empty_state(),
        current_physical_quantity=Decimal("0"),
        entry_scores=[_score("long", "LONG", 5.0)],
        close_reasons={},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )

    equity_after_fee = Decimal("1000") - plan.estimated_fee
    assert abs(plan.target_quantity * Decimal("100")) <= Decimal("4.5") * equity_after_fee
    assert plan.resize_scale < Decimal("1")


def test_same_sleeve_is_blocked_but_cross_sleeve_overlap_is_allowed():
    state = _empty_state()
    state["open_sleeves"] = {
        "long": {
            "name": "long",
            "side": "LONG",
            "quantity": "1",
            "signal_id": "long:old",
            "entry_reference_price": 100.0,
            "exit_at": "2026-09-05T00:00:00Z",
        }
    }
    plan = build_plan(
        policy=parse_policy(_portfolio()),
        state=state,
        current_physical_quantity=Decimal("1"),
        entry_scores=[
            _score("long", "LONG", 1.0, signal_id="long:new"),
            _score("dollar_rally_short", "SHORT", 0.5),
        ],
        close_reasons={},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )

    assert plan.blocked_entries == (("long", "same_sleeve_open_at_decision"),)
    assert set(plan.proposed_open_sleeves) == {"long", "dollar_rally_short"}


def test_hedge_snapshot_fails_closed_when_both_physical_sides_are_open():
    positions = [
        {"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": "0.1"},
        {"symbol": "BTCUSDT", "positionSide": "SHORT", "positionAmt": "-0.2"},
    ]
    with pytest.raises(RuntimeError, match="both LONG and SHORT"):
        physical_signed_quantity(positions, symbol="BTCUSDT")


def test_pending_recovery_accepts_only_base_target_or_path_intermediate():
    assert classify_pending_position(Decimal("2"), Decimal("2"), Decimal("-5")) == "base"
    assert classify_pending_position(Decimal("-5"), Decimal("2"), Decimal("-5")) == "target"
    assert classify_pending_position(Decimal("0"), Decimal("2"), Decimal("-5")) == "intermediate"
    with pytest.raises(RuntimeError, match="outside pending transition path"):
        classify_pending_position(Decimal("3"), Decimal("2"), Decimal("-5"))


def test_policy_contract_fails_closed_on_legacy_or_drifted_fields():
    assert parse_policy({"base_sleeves": []}) is None
    drifted = _portfolio()
    drifted["position_aggregation"]["offset_opposite_sides_before_costs"] = False
    with pytest.raises(RuntimeError, match="offset_opposite_sides_before_costs"):
        parse_policy(drifted)


def test_macro_target_updates_signed_virtual_units_without_resizing_other_sleeves():
    portfolio = _portfolio()
    portfolio["base_sleeves"].append(
        {"name": "macro_flow", "side": "AUTO", "weight": 1.0}
    )
    state = _empty_state()
    state["open_sleeves"] = {
        "long": {
            "name": "long",
            "side": "LONG",
            "quantity": "1",
            "signal_id": "long:old",
            "entry_reference_price": 100.0,
            "exit_at": "2026-09-05T00:00:00Z",
        }
    }
    target = {
        "name": "macro_flow",
        "kind": "target",
        "target_fraction": -0.5,
        "weight": 1.0,
        "signal_id": "macro:1",
        "date": "2026-09-04T00:00:00Z",
        "current_close": 100.0,
        "policy_metadata": {},
    }
    plan = build_plan(
        policy=parse_policy(portfolio),
        state=state,
        current_physical_quantity=Decimal("1"),
        entry_scores=[target],
        close_reasons={},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )

    assert plan.proposed_open_sleeves["long"]["quantity"] == "1"
    assert plan.proposed_open_sleeves["macro_flow"]["side"] == "SHORT"
    assert Decimal(plan.proposed_open_sleeves["macro_flow"]["quantity"]) == 5
    assert plan.target_quantity == Decimal("-4.000")


def test_state_initialization_only_migrates_a_flat_empty_ledger():
    policy = parse_policy(_portfolio())
    assert policy is not None
    state = {"open_sleeves": {}, "processed_signals": {}}
    initialized = _initialize_signed_net_state(
        state,
        policy=policy,
        physical_quantity=Decimal("0"),
    )
    assert initialized["position_aggregation_mode"] == SIGNED_NET_HEDGE_MODE
    assert initialized["signed_net_revision"] == 0

    with pytest.raises(RuntimeError, match="flat empty ledger"):
        _initialize_signed_net_state(
            {
                "open_sleeves": {
                    "long": {"side": "LONG", "quantity": "1"},
                },
                "processed_signals": {},
            },
            policy=policy,
            physical_quantity=Decimal("1"),
        )


def test_legacy_runner_refuses_to_interpret_a_signed_virtual_ledger():
    signed_state = _empty_state()
    with pytest.raises(RuntimeError, match="legacy portfolio cannot consume"):
        _validate_state_aggregation_mode(signed_state, policy=None)

    policy = parse_policy(_portfolio())
    assert policy is not None
    _validate_state_aggregation_mode(signed_state, policy=policy)


def test_live_adapter_submits_one_hedge_long_leg_for_signed_net_delta(tmp_path: Path):
    policy = parse_policy(_portfolio())
    assert policy is not None
    plan = build_plan(
        policy=policy,
        state=_empty_state(),
        current_physical_quantity=Decimal("0"),
        entry_scores=[
            _score("long", "LONG", 1.0),
            _score("dollar_rally_short", "SHORT", 0.5),
        ],
        close_reasons={},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )

    client = SimpleNamespace(get_positions=AsyncMock())
    client.get_positions.side_effect = [
        [],
        [
            {
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "positionAmt": "5",
            }
        ]
    ]
    raw = {
        "status": "FILLED",
        "orderId": 1,
        "clientOrderId": "net",
        "executedQty": "5",
        "avgPrice": "100",
        "transactTime": 1788480300000,
    }
    async def place_after_journal(**kwargs):
        journal = json.loads(state_file.read_text())
        intents = journal["pending_signed_net_transition"]["order_intents"]
        assert len(intents) == 1
        assert intents[0]["client_order_id"] == kwargs["client_order_id"]
        return raw

    state = _empty_state()
    state["pending_signed_net_transition"] = {
        "plan": plan_to_dict(plan),
        "order_intents": [],
        "order_records": [],
    }
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state))
    with (
        patch(
            "execution.portfolio_live._place_or_resolve_market_order",
            new=AsyncMock(side_effect=place_after_journal),
        ) as place,
        patch(
            "execution.portfolio_live._confirm_market_order",
            new=AsyncMock(return_value=raw),
        ),
    ):
        orders = asyncio.run(
            _execute_signed_net_plan(
                client=client,
                exec_cfg=SimpleNamespace(symbol="BTCUSDT"),
                plan=plan,
                observed_quantity=Decimal("0"),
                min_quantity=Decimal("0.001"),
                state=state,
                state_file=state_file,
            )
        )

    assert len(orders) == 1
    assert place.await_args.kwargs["side"] == "BUY"
    assert place.await_args.kwargs["position_side"] == "LONG"
    assert place.await_args.kwargs["quantity"] == Decimal("5.000")
    assert orders[0]["physical_quantity_after"] == "5"


def test_dry_transition_journals_then_commits_virtual_ledger(tmp_path: Path):
    state = _empty_state()
    portfolio = _portfolio()
    portfolio["base_sleeves"][1]["weight"] = 1.0
    plan = build_plan(
        policy=parse_policy(portfolio),
        state=state,
        current_physical_quantity=Decimal("0"),
        entry_scores=[
            _score("long", "LONG", 1.0),
            _score("dollar_rally_short", "SHORT", 1.0),
        ],
        close_reasons={},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )
    state_file = tmp_path / "state.json"

    committed, orders = asyncio.run(
        _apply_signed_net_transition(
            policy=parse_policy(portfolio),
            state=state,
            state_file=state_file,
            plan=plan,
            close_reasons={},
            client=None,
            exec_cfg=WaveExecutionConfig(dry_run=True),
            min_quantity=Decimal("0.001"),
            engine=None,
            strategy_name="rllm",
            execution_exchange="binance-mainnet",
            computing_wall_time_sec=0.1,
        )
    )

    assert orders == []
    assert "pending_signed_net_transition" not in committed
    assert committed["signed_net_revision"] == 1
    assert set(committed["open_sleeves"]) == {"long", "dollar_rally_short"}
    assert json.loads(state_file.read_text())["last_signed_net_plan_id"] == plan.plan_id


def test_restart_commits_pending_plan_when_exchange_already_reached_target(tmp_path: Path):
    policy = parse_policy(_portfolio())
    assert policy is not None
    base = _empty_state()
    plan = build_plan(
        policy=policy,
        state=base,
        current_physical_quantity=Decimal("0"),
        entry_scores=[
            _score("long", "LONG", 1.0),
            _score("dollar_rally_short", "SHORT", 0.5),
        ],
        close_reasons={},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )
    record = {
        "client_order_id": "owned-full-fill",
        "status": "FILLED",
        "execution_uncertain": False,
        "filled_quantity": str(plan.target_quantity),
        "side": "BUY",
        "reduce_position": False,
        "avg_price": "100",
        "finished_at": "2026-09-04T00:05:01Z",
    }
    base["pending_signed_net_transition"] = {
        "plan": plan_to_dict(plan),
        "policy_hash": plan.policy_hash,
        "close_reasons": {},
        "order_intents": [
            {
                "client_order_id": "owned-full-fill",
                "side": "BUY",
                "position_side": "LONG",
                "quantity": str(plan.target_quantity),
                "reduce_position": False,
            }
        ],
        "order_records": [record],
    }
    state_file = tmp_path / "state.json"
    client = SimpleNamespace(
        get_positions=AsyncMock(
            return_value=[
                {
                    "symbol": "BTCUSDT",
                    "positionSide": "LONG",
                    "positionAmt": str(plan.target_quantity),
                }
            ]
        )
    )
    with patch(
        "execution.portfolio_live._enforce_signed_net_post_fill_cap",
        new=AsyncMock(return_value=(plan, [])),
    ):
        committed, new_orders = asyncio.run(
            _resume_signed_net_transition(
                state=base,
                state_file=state_file,
                policy=policy,
                client=client,
                exec_cfg=WaveExecutionConfig(dry_run=False),
                physical_quantity=plan.target_quantity,
                min_quantity=Decimal("0.001"),
                engine=None,
                strategy_name="rllm",
                execution_exchange="binance-mainnet",
            )
        )

    assert new_orders == []
    assert committed["signed_net_revision"] == 1
    assert "pending_signed_net_transition" not in committed


def test_dry_decision_routes_overlapping_long_short_through_signed_net(tmp_path: Path):
    policy = parse_policy(_portfolio())
    assert policy is not None
    state_file = tmp_path / "state.json"
    market = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-09-04T00:00:00Z"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
            }
        ]
    )
    features = pd.DataFrame([{"placeholder": 1.0}])
    result = asyncio.run(
        _run_signed_net_decision(
            policy=policy,
            state={"open_sleeves": {}, "processed_signals": {}},
            state_file=state_file,
            sleeve_scores=[
                _score("long", "LONG", 1.0),
                _score("dollar_rally_short", "SHORT", 0.5),
            ],
            enriched=market,
            features=features,
            now=pd.Timestamp("2026-09-04T00:05:00Z"),
            client=None,
            exec_cfg=WaveExecutionConfig(dry_run=True, interval_minutes=5),
            engine=None,
            strategy_name="rllm",
            execution_exchange="binance-mainnet",
            computing_wall_time_sec=0.1,
        )
    )

    assert result["target_quantity"] == Decimal("0.500")
    assert result["opened"] == ["long", "dollar_rally_short"]
    assert result["state"]["position_aggregation_mode"] == SIGNED_NET_HEDGE_MODE


def test_stale_point_entry_fails_closed_but_held_macro_target_can_catch_up():
    portfolio = _portfolio()
    portfolio["base_sleeves"].append(
        {"name": "macro_flow", "side": "AUTO", "weight": 1.0}
    )
    policy = parse_policy(portfolio)
    assert policy is not None
    execution_time = "2026-09-04T00:05:00Z"
    now = pd.Timestamp("2026-09-04T00:06:00Z")
    point_entry = _score("long", "LONG", 1.0)
    point_entry["execution_time"] = execution_time
    macro_target = {
        "name": "macro_flow",
        "kind": "target",
        "weight": 1.0,
        "active": True,
        "ready": True,
        "emit": True,
        "side": "LONG",
        "signal_id": "macro_flow:2026-09-04T00:05:00+00:00",
        "date": "2026-09-04T00:00:00Z",
        "current_close": 100.0,
        "target_fraction": 0.25,
        "execution_time": execution_time,
        "policy_metadata": {"target_maintenance": True},
    }

    eligible = _signed_net_eligible_entry_scores(
        sleeve_scores=[point_entry, macro_target],
        policy=policy,
        now=now,
        interval_minutes=5,
    )

    by_name = {score["name"]: score for score in eligible}
    assert by_name["long"]["ready"] is False
    assert "signed_net_signal_age=60.000s:fail_closed" in by_name["long"][
        "reasons"
    ]
    assert by_name["macro_flow"]["ready"] is True
    assert "signed_net_target_maintenance_age=60.000s:pass" in by_name[
        "macro_flow"
    ]["reasons"]


def test_barrier_monitor_closes_virtual_sleeve_via_aggregate_flip(tmp_path: Path):
    policy = parse_policy(_portfolio())
    assert policy is not None
    state = _empty_state()
    state["signed_net_policy_hash"] = policy_digest(policy)
    state["open_sleeves"] = {
        "long": {
            "name": "long",
            "side": "LONG",
            "quantity": "1",
            "signal_id": "long:old",
            "entry_reference_price": 100.0,
            "entry_fill_price": 100.0,
            "entry_filled_at": "2026-09-04T00:00:00Z",
            "exit_at": "2026-09-05T00:00:00Z",
            "barrier_exit": {
                "type": "fixed_bps",
                "take_bps": None,
                "stop_bps": 500.0,
                "entry_price_source": "actual_fill_avg",
                "entry_execution": "market",
                "price_source": "last_trade",
                "same_bar_policy": "stop_before_take",
                "live_touch_policy": "first_aggtrade_touch",
                "stream_gap_policy": "market_close_fail_safe",
                "execution": "market",
                "monitor_interval_sec": 0.05,
            },
            "barrier_stream_session_id": "session-1",
            "barrier_stream_gap_count": 0,
        },
        "dollar_rally_short": {
            "name": "dollar_rally_short",
            "side": "SHORT",
            "quantity": "0.5",
            "signal_id": "short:old",
            "entry_reference_price": 100.0,
            "entry_fill_price": 100.0,
            "entry_filled_at": "2026-09-04T00:00:00Z",
            "exit_at": "2026-09-05T00:00:00Z",
            "barrier_exit": None,
        },
    }
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state))

    class Stream:
        session_id = "session-1"
        gap_count = 0
        healthy = True

        def __init__(self):
            self.sent = False

        async def collect(self, timeout_sec):
            if not self.sent:
                self.sent = True
                return [
                    SimpleNamespace(
                        price=94.0,
                        event_time_ms=int(
                            pd.Timestamp("2026-09-04T00:01:00Z").timestamp()
                            * 1000
                        ),
                    )
                ]
            await asyncio.sleep(timeout_sec)
            return []

    async def commit(**kwargs):
        plan = kwargs["plan"]
        return apply_plan(kwargs["state"], plan), []

    snapshot = AsyncMock(
        return_value=(
            Decimal("0.5"),
            Decimal("100"),
            Decimal("94"),
            Decimal("0.001"),
            Decimal("0.001"),
        )
    )
    with (
        patch(
            "execution.portfolio_live._signed_net_account_snapshot",
            new=snapshot,
        ),
        patch(
            "execution.portfolio_live._apply_signed_net_transition",
            new=AsyncMock(side_effect=commit),
        ) as transition,
    ):
        closed = asyncio.run(
            _wait_with_signed_net_barrier_monitor(
                wait_sec=0.02,
                poll_sec=0.01,
                state_file=state_file,
                policy=policy,
                client=SimpleNamespace(),
                exec_cfg=WaveExecutionConfig(dry_run=False, interval_minutes=5),
                engine=None,
                strategy_name="rllm",
                execution_exchange="binance-mainnet",
                db_lease=None,
                trade_stream=Stream(),
            )
        )

    assert closed == ["long"]
    plan = transition.await_args.kwargs["plan"]
    assert plan.target_quantity == Decimal("-0.500")
    assert [(leg.side, leg.position_side) for leg in plan.legs] == [
        ("SELL", "LONG"),
        ("SELL", "SHORT"),
    ]


def test_idle_barrier_monitor_does_not_poll_private_or_mark_rest(tmp_path: Path):
    policy = parse_policy(_portfolio())
    assert policy is not None
    state = _empty_state()
    state["signed_net_policy_hash"] = policy_digest(policy)
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state))

    snapshot = AsyncMock(
        side_effect=AssertionError(
            "an idle signed-net barrier monitor must not consume Binance REST quota"
        )
    )
    with patch(
        "execution.portfolio_live._signed_net_account_snapshot",
        new=snapshot,
    ):
        closed = asyncio.run(
            _wait_with_signed_net_barrier_monitor(
                wait_sec=0.02,
                poll_sec=0.005,
                state_file=state_file,
                policy=policy,
                client=SimpleNamespace(),
                exec_cfg=WaveExecutionConfig(dry_run=False, interval_minutes=5),
                engine=None,
                strategy_name="rllm",
                execution_exchange="binance-mainnet",
                db_lease=None,
                trade_stream=None,
            )
        )

    assert closed == []
    snapshot.assert_not_awaited()


def test_corrupt_existing_state_fails_closed(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text('{"open_sleeves":')
    with pytest.raises(RuntimeError, match="state is corrupt"):
        _load_state(state_file)


def test_exactly_offset_active_ledger_requires_policy_hash():
    portfolio = _portfolio()
    portfolio["base_sleeves"][1]["weight"] = 1.0
    policy = parse_policy(portfolio)
    assert policy is not None
    state = _empty_state()
    state["open_sleeves"] = {
        "long": {"side": "LONG", "quantity": "1"},
        "dollar_rally_short": {"side": "SHORT", "quantity": "1"},
    }
    with pytest.raises(RuntimeError, match="policy hash is missing"):
        _initialize_signed_net_state(
            state,
            policy=policy,
            physical_quantity=Decimal("0"),
        )


def test_inactive_signal_is_frozen_before_late_activation():
    policy = parse_policy(_portfolio())
    assert policy is not None
    inactive = _score("long", "LONG", 1.0, signal_id="long:frozen")
    inactive["active"] = False
    first = build_plan(
        policy=policy,
        state=_empty_state(),
        current_physical_quantity=Decimal("0"),
        entry_scores=[inactive],
        close_reasons={},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )
    state = apply_plan(_empty_state(), first)
    assert state["processed_signal_ids"]["long"] == ["long:frozen"]
    assert state["open_sleeves"] == {}

    late = _score("long", "LONG", 1.0, signal_id="long:frozen")
    second = build_plan(
        policy=policy,
        state=state,
        current_physical_quantity=Decimal("0"),
        entry_scores=[late],
        close_reasons={},
        equity=Decimal("1000"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:01Z",
    )
    assert second.opened_sleeves == ()
    assert ("long", "signal_already_processed") in second.blocked_entries


def test_partial_market_fill_is_cancelled_before_it_is_safe_to_continue():
    class Client:
        def __init__(self):
            self.cancelled = False

        async def get_order(self, symbol, order_id=None, client_order_id=None):
            return {
                "orderId": 17,
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.4",
                "avgPrice": "100",
            }

        async def cancel_order(self, symbol, order_id=None, client_order_id=None):
            self.cancelled = True
            return {
                "orderId": 17,
                "status": "CANCELED",
                "executedQty": "0.4",
                "avgPrice": "100",
            }

        async def get_trades(self, symbol, limit=1000):
            return [
                {
                    "orderId": 17,
                    "qty": "0.4",
                    "price": "100",
                    "quoteQty": "40",
                    "realizedPnl": "0",
                    "commission": "0.016",
                    "commissionAsset": "USDT",
                    "time": 1_788_480_300_000,
                }
            ]

    client = Client()
    result = asyncio.run(
        _confirm_market_order(
            client=client,
            symbol="BTCUSDT",
            raw_order={
                "orderId": 17,
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.4",
            },
            client_order_id="net-17",
            require_fill_details=True,
            expected_quantity=Decimal("1"),
        )
    )
    assert client.cancelled is True
    assert result["status"] == "CANCELED"
    assert result["execution_uncertain"] is False
    assert result["executedQty"] == "0.4"


def test_journaled_fills_must_explain_the_physical_position(tmp_path: Path):
    policy = parse_policy(_portfolio())
    assert policy is not None
    plan = build_plan(
        policy=policy,
        state=_empty_state(),
        current_physical_quantity=Decimal("0"),
        entry_scores=[_score("long", "LONG", 1.0)],
        close_reasons={},
        equity=Decimal("100"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )
    record = {
        "client_order_id": "owned",
        "status": "FILLED",
        "execution_uncertain": False,
        "filled_quantity": "0.5",
        "side": "BUY",
    }
    state = _empty_state()
    state["pending_signed_net_transition"] = {
        "plan": plan_to_dict(plan),
        "order_intents": [
            {
                "client_order_id": "owned",
                "side": "BUY",
                "position_side": "LONG",
                "quantity": "0.5",
                "reduce_position": False,
            }
        ],
        "order_records": [record],
    }
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="not attributable"):
        asyncio.run(
            _execute_signed_net_plan(
                client=SimpleNamespace(),
                exec_cfg=SimpleNamespace(symbol="BTCUSDT"),
                plan=plan,
                observed_quantity=Decimal("0.7"),
                min_quantity=Decimal("0.001"),
                state=state,
                state_file=state_file,
            )
        )


def test_restart_resolves_journaled_order_before_any_duplicate_post(tmp_path: Path):
    policy = parse_policy(_portfolio())
    assert policy is not None
    plan = build_plan(
        policy=policy,
        state=_empty_state(),
        current_physical_quantity=Decimal("0"),
        entry_scores=[_score("long", "LONG", 1.0)],
        close_reasons={},
        equity=Decimal("100"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )
    client_order_id = "journaled-before-crash"
    state = _empty_state()
    state["pending_signed_net_transition"] = {
        "plan": plan_to_dict(plan),
        "order_intents": [
            {
                "sequence": 0,
                "client_order_id": client_order_id,
                "side": "BUY",
                "position_side": "LONG",
                "quantity": str(plan.target_quantity),
                "reduce_position": False,
            }
        ],
        "order_records": [],
    }
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state))
    raw = {
        "status": "FILLED",
        "orderId": 44,
        "clientOrderId": client_order_id,
        "executedQty": str(plan.target_quantity),
        "avgPrice": "100",
        "transactTime": 1_788_480_301_000,
    }
    client = SimpleNamespace(
        get_order=AsyncMock(return_value=raw),
        get_positions=AsyncMock(
            return_value=[
                {
                    "symbol": "BTCUSDT",
                    "positionSide": "LONG",
                    "positionAmt": str(plan.target_quantity),
                }
            ]
        ),
    )
    with (
        patch(
            "execution.portfolio_live._confirm_market_order",
            new=AsyncMock(return_value=raw),
        ),
        patch(
            "execution.portfolio_live._place_or_resolve_market_order",
            new=AsyncMock(),
        ) as place,
    ):
        records = asyncio.run(
            _execute_signed_net_plan(
                client=client,
                exec_cfg=SimpleNamespace(symbol="BTCUSDT"),
                plan=plan,
                observed_quantity=plan.target_quantity,
                min_quantity=Decimal("0.001"),
                state=state,
                state_file=state_file,
            )
        )
    place.assert_not_awaited()
    assert records[0]["client_order_id"] == client_order_id


def test_virtual_barrier_anchor_uses_actual_exposure_increase_fill(tmp_path: Path):
    policy = parse_policy(_portfolio())
    assert policy is not None
    plan = build_plan(
        policy=policy,
        state=_empty_state(),
        current_physical_quantity=Decimal("0"),
        entry_scores=[_score("long", "LONG", 1.0)],
        close_reasons={},
        equity=Decimal("100"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )
    state = _empty_state()
    state["signed_net_policy_hash"] = policy_digest(policy)
    state["pending_signed_net_transition"] = {"plan": plan_to_dict(plan)}
    record = {
        "client_order_id": "fill-1",
        "status": "FILLED",
        "side": "BUY",
        "reduce_position": False,
        "filled_quantity": "1",
        "avg_price": "101.25",
        "finished_at": "2026-09-04T00:05:03Z",
        "raw_order": {
            "trade_report": {"last_fill_at": "2026-09-04T00:05:02Z"}
        },
    }
    committed = _finalize_signed_net_plan(
        state=state,
        state_file=tmp_path / "committed.json",
        plan=plan,
        close_reasons={},
        order_records=[record],
        engine=None,
        strategy_name="rllm",
        execution_exchange="binance-mainnet",
        symbol="BTCUSDT",
        computing_wall_time_sec=0.1,
    )
    sleeve = committed["open_sleeves"]["long"]
    assert sleeve["entry_fill_price"] == 101.25
    assert sleeve["entry_filled_at"] == "2026-09-04 00:05:02+00:00"
    assert sleeve["entry_fill_price_source"] == "aggregate_exposure_increase_fill"


def test_post_fill_retarget_is_strictly_risk_reducing_and_keeps_ledger_exact():
    policy = parse_policy(_portfolio(cap=1.0))
    assert policy is not None
    plan = build_plan(
        policy=policy,
        state=_empty_state(),
        current_physical_quantity=Decimal("0"),
        entry_scores=[_score("long", "LONG", 1.0)],
        close_reasons={},
        equity=Decimal("100"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )
    corrected = retarget_plan_for_risk_reduction(
        plan,
        Decimal("0.950"),
        fee_rate=policy.fee_rate,
    )
    assert corrected.target_quantity == Decimal("0.950")
    assert corrected.proposed_dust_quantity == (
        corrected.target_quantity - corrected.unrounded_target_quantity
    )
    committed = apply_plan(_empty_state(), corrected)
    assert signed_state_quantity(committed) == Decimal("0.950")
    with pytest.raises(RuntimeError, match="may not increase exposure"):
        retarget_plan_for_risk_reduction(
            corrected,
            Decimal("0.951"),
            fee_rate=policy.fee_rate,
        )


def test_actual_post_fill_cap_breach_is_reduced_before_commit(tmp_path: Path):
    policy = parse_policy(_portfolio(cap=1.0))
    assert policy is not None
    plan = build_plan(
        policy=policy,
        state=_empty_state(),
        current_physical_quantity=Decimal("0"),
        entry_scores=[_score("long", "LONG", 1.0)],
        close_reasons={},
        equity=Decimal("100"),
        reference_price=Decimal("100"),
        quantity_step=Decimal("0.001"),
        interval_minutes=5,
        execution_time="2026-09-04T00:05:00Z",
    )
    initial_record = {
        "client_order_id": "initial",
        "status": "FILLED",
        "execution_uncertain": False,
        "filled_quantity": str(plan.target_quantity),
        "avg_price": "100",
        "side": "BUY",
        "position_side": "LONG",
        "reduce_position": False,
        "finished_at": "2026-09-04T00:05:01Z",
    }
    state = _empty_state()
    state["pending_signed_net_transition"] = {
        "version": 1,
        "status": "PLANNED",
        "policy_hash": plan.policy_hash,
        "plan": plan_to_dict(plan),
        "close_reasons": {},
        "order_intents": [
            {
                "sequence": 0,
                "client_order_id": "initial",
                "side": "BUY",
                "position_side": "LONG",
                "quantity": str(plan.target_quantity),
                "reduce_position": False,
            }
        ],
        "order_records": [initial_record],
    }
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state))
    client = SimpleNamespace(
        get_positions=AsyncMock(
            side_effect=[
                [
                    {
                        "symbol": "BTCUSDT",
                        "positionSide": "LONG",
                        "positionAmt": str(plan.target_quantity),
                    }
                ],
                [
                    {
                        "symbol": "BTCUSDT",
                        "positionSide": "LONG",
                        "positionAmt": "0.990",
                    }
                ],
            ]
        )
    )
    correction_raw = {
        "status": "FILLED",
        "orderId": 22,
        "executedQty": "0.009",
        "avgPrice": "101",
        "transactTime": 1_788_480_301_000,
    }
    snapshots = AsyncMock(
        side_effect=[
            (
                plan.target_quantity,
                Decimal("100"),
                Decimal("101"),
                Decimal("0.001"),
                Decimal("0.001"),
            ),
            (
                Decimal("0.990"),
                Decimal("99.99"),
                Decimal("101"),
                Decimal("0.001"),
                Decimal("0.001"),
            ),
        ]
    )
    async def confirm_correction(**kwargs):
        return {
            **correction_raw,
            "clientOrderId": kwargs["client_order_id"],
        }

    with (
        patch(
            "execution.portfolio_live._signed_net_account_snapshot",
            new=snapshots,
        ),
        patch(
            "execution.portfolio_live._place_or_resolve_market_order",
            new=AsyncMock(return_value=correction_raw),
        ) as place,
        patch(
            "execution.portfolio_live._confirm_market_order",
            new=AsyncMock(side_effect=confirm_correction),
        ),
    ):
        corrected, records = asyncio.run(
            _enforce_signed_net_post_fill_cap(
                policy=policy,
                state=state,
                state_file=state_file,
                plan=plan,
                client=client,
                exec_cfg=WaveExecutionConfig(dry_run=False),
                engine=None,
                strategy_name="rllm",
                execution_exchange="binance-mainnet",
                computing_wall_time_sec=0.1,
            )
        )
    assert corrected.target_quantity == Decimal("0.990")
    assert records[-1]["side"] == "SELL"
    assert place.await_args.kwargs["position_side"] == "LONG"
    assert Decimal(
        json.loads(state_file.read_text())["pending_signed_net_transition"][
            "plan"
        ]["target_quantity"]
    ) == Decimal("0.990")
