import numpy as np
import pandas as pd
import pytest

from training.evaluate_alpha_state_card_ranker import (
    _feature_vocabulary,
    cluster_signflip,
    event_gate_rows_to_cards,
    parse_args,
    predict_schedule,
    select_and_fit,
    strict_economics,
)


def _event(date, winner="A", *, a=2.0, b=-2.0, entry=None, exit_=None):
    entry = entry or (pd.Timestamp(date, tz="UTC") + pd.Timedelta(minutes=5)).isoformat()
    exit_ = exit_ or (pd.Timestamp(date, tz="UTC") + pd.Timedelta(minutes=10)).isoformat()
    return {
        "date": date,
        "causal_state": {"volatility": 0.25, "future_return": 999.0},
        "options": [
            {
                "id": "A", "policy": "p_a", "side": "LONG",
                "features": {"signal": a}, "utility": 1.0 if winner == "A" else -1.0,
                "trade": {"entry_time": entry, "exit_time": exit_, "side": "LONG"},
            },
            {
                "id": "B", "policy": "p_b", "side": "SHORT",
                "features": {"signal": b}, "utility": 1.0 if winner == "B" else -1.0,
                "trade": {"entry_time": entry, "exit_time": exit_, "side": "SHORT"},
            },
        ],
    }


def _train_rows():
    return [
        _event("2023-07-05T00:00:00"),
        _event("2023-08-05T00:00:00", a=1.5, b=-1.0),
        _event("2023-09-05T00:00:00", a=1.0, b=-1.5),
        _event("2023-10-05T00:00:00", a=3.0, b=-2.0),
        _event("2023-11-05T00:00:00", a=2.5, b=-3.0),
        _event("2023-12-05T00:00:00", a=1.2, b=-0.5),
    ]


def test_train_only_fit_is_deterministic_and_excludes_outcome_features():
    rows = _train_rows()
    names = _feature_vocabulary(rows)
    assert not any("utility" in name or "future_return" in name for name in names)

    first, audit_a = select_and_fit(rows)
    second, audit_b = select_and_fit(sorted(rows, key=lambda row: row["date"]))
    assert first.public() == second.public()
    assert audit_a["oos_used_for_selection"] is False
    assert audit_b["method"] == "expanding_chronological_2023H2_only"


def test_online_reliability_is_not_updated_until_a_later_decision_after_exit():
    model, _ = select_and_fit(_train_rows())
    rows = [
        _event("2024-01-01T00:00:00", entry="2024-01-01T00:05:00Z", exit_="2024-01-01T00:20:00Z"),
        _event("2024-01-01T00:10:00", entry="2024-01-01T00:25:00Z", exit_="2024-01-01T00:35:00Z"),
        _event("2024-01-01T00:30:00", entry="2024-01-01T00:40:00Z", exit_="2024-01-01T00:50:00Z"),
    ]

    _, audit = predict_schedule(model, rows, online_reliability=True)

    assert len(audit["reliability_updates"]) == 1
    update = audit["reliability_updates"][0]
    assert update["available_at"] == "2024-01-01T00:20:00+00:00"
    assert update["applied_at"] == "2024-01-01T00:30:00+00:00"


def test_strict_economics_uses_exact_funding_interval_and_stress_cost():
    dates = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="5min")
    market = pd.DataFrame({
        "date": dates,
        "open": [100.0, 100.0, 102.0, 102.0],
        "high": [101.0, 103.0, 103.0, 103.0],
        "low": [99.0, 99.0, 101.0, 101.0],
        "close": [100.0, 102.0, 102.0, 102.0],
    })
    funding = pd.DataFrame({
        "funding_time": [dates[1], dates[2], dates[3]],
        "funding_rate": [0.01, 0.01, 100.0],
        "mark_price": [100.0, 102.0, 102.0],
    })
    schedule = [{
        "decision_time": dates[0], "entry_time": dates[1], "exit_time": dates[3],
        "side": 1, "policy": "p_a",
    }]

    base = strict_economics(schedule, market, funding, start=dates[0], end=dates[3], cost_per_side=0.0006, leverage=0.5)
    stress = strict_economics(schedule, market, funding, start=dates[0], end=dates[3], cost_per_side=0.001, leverage=0.5)

    trade = base["trade_rows"][0]
    assert trade["funding_events"] == 2  # exit-boundary event is excluded
    assert trade["funding_cash"] == pytest.approx(-0.0101)
    assert stress["ending_equity"] < base["ending_equity"]
    assert base["strict_mdd_pct"] > 0.0


def test_weekly_signflip_is_exact_and_deterministic_for_small_samples():
    trades = [
        {"entry_time": "2024-01-01T00:00:00Z", "net_return": 0.01},
        {"entry_time": "2024-01-08T00:00:00Z", "net_return": 0.02},
    ]
    first = cluster_signflip(trades, seed=1)
    second = cluster_signflip(trades, seed=999)
    assert first["method"] == "exact_weekly_cluster_signflip_one_sided"
    assert first["pvalue"] == 0.25
    assert first["pvalue"] == second["pvalue"]


def test_cli_exposes_required_inputs_and_optional_online_reliability():
    args = parse_args([
        "--input-jsonl", "cards.jsonl", "--market-csv", "market.csv",
        "--funding-csv", "funding.csv", "--output", "report.json",
        "--online-reliability",
    ])
    assert args.input_jsonl == "cards.jsonl"
    assert args.online_reliability is True


def test_original_event_gate_rows_convert_to_executable_listwise_cards():
    entry = pd.Timestamp("2023-12-20T00:00:00Z")
    dates = pd.date_range(end=entry + pd.Timedelta(hours=1), periods=950, freq="5min")
    prices = 100.0 * np.exp(np.linspace(0.0, 0.05, len(dates)))
    market = pd.DataFrame({
        "date": dates, "open": prices, "high": prices * 1.001,
        "low": prices * 0.999, "close": prices,
    })
    funding = pd.DataFrame({
        "funding_time": pd.date_range(end=entry, periods=20, freq="8h"),
        "funding_rate": np.linspace(-0.0001, 0.0001, 20),
        "mark_price": np.linspace(95.0, 100.0, 20),
    })

    def gate(policy, side, utility):
        return {
            "task": "alpha_event_gate", "stage": "train",
            "entry_time": entry.isoformat(),
            "exit_time": (entry + pd.Timedelta(hours=1)).isoformat(),
            "policy_id": policy, "slug": policy.lower(), "side": side,
            "research_train_pass": True,
            "prompt": (
                'frozen_formula: {"policy":{"hold_hours":1}}\n'
                'signal_time_event: {"causal_rank":0.75}'
            ),
            "metadata": {"net_return": utility, "funding_cash_over_pre_equity": 999.0},
        }

    cards, audit = event_gate_rows_to_cards(
        [gate("B", -1, -0.02), gate("A", 1, 0.03)], market, funding
    )

    assert audit["mode"] == "original_event_gate_to_listwise_event_card"
    assert len(cards) == 1
    assert [option["policy"] for option in cards[0]["options"]] == ["A", "B", "WAIT"]
    assert cards[0]["options"][-1]["target_utility"] == 0.0
    assert cards[0]["options"][0]["target_utility"] == 0.03
    names = _feature_vocabulary(cards)
    assert any("return_1h" in name for name in names)
    assert not any("funding_cash_over_pre_equity" in name for name in names)
