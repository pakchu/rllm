import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training import build_alpha_state_card_rank_sft as builder

ENTRY = pd.Timestamp("2024-01-04T00:00:00Z")


def _market() -> pd.DataFrame:
    dates = pd.date_range(end=ENTRY, periods=900, freq="5min", tz="UTC")
    close = 100.0 * np.exp(np.linspace(0.0, 0.09, len(dates)))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
        }
    )


def _funding() -> pd.DataFrame:
    dates = pd.date_range(end=ENTRY, periods=20, freq="8h", tz="UTC")
    return pd.DataFrame(
        {"date": dates, "funding_rate": np.linspace(-0.0002, 0.0003, len(dates))}
    )


def _event(stage: str, policy: str, net: float, *, entry: pd.Timestamp = ENTRY) -> dict:
    formula = {"features": {"rule": policy}, "policy": {"hold_hours": 8, "fixed": True}}
    return {
        "task": "alpha_event_gate",
        "stage": stage,
        "entry_time": entry.isoformat(),
        "exit_time": (entry + pd.Timedelta(hours=8)).isoformat(),
        "policy_id": policy,
        "slug": policy.lower(),
        "side": 1 if policy != "SHORT" else -1,
        "research_train_pass": True,
        "prompt": "header\nfrozen_formula: " + json.dumps(formula, sort_keys=True),
        "metadata": {"net_return": net},
    }


def _events() -> list[dict]:
    train_entry = ENTRY - pd.Timedelta(days=1)
    rows = [
        _event("train", f"T{i}", value, entry=train_entry + pd.Timedelta(minutes=5 * i))
        for i, value in enumerate((-0.04, -0.02, 0.0, 0.02, 0.04))
    ]
    rows += [_event("test", "A", -0.03), _event("test", "B", -0.01)]
    rows.append({**_event("test", "IGNORED", 99.0), "research_train_pass": False})
    return rows


def test_signal_features_are_strictly_pre_entry_and_prompts_have_no_outcomes() -> None:
    market, funding = _market(), _funding()
    before = builder.build_dataset(_events(), market, funding, min_utility_gap=0.005)
    changed_market, changed_funding = market.copy(), funding.copy()
    changed_market.loc[
        changed_market["date"].ge(ENTRY), ["open", "high", "low", "close"]
    ] = 1_000_000.0
    changed_funding.loc[changed_funding["date"].ge(ENTRY), "funding_rate"] = 100.0
    after = builder.build_dataset(
        _events(), changed_market, changed_funding, min_utility_gap=0.005
    )

    before_test = [row for row in before["pointwise"] if row["stage"] == "test"]
    after_test = [row for row in after["pointwise"] if row["stage"] == "test"]
    assert [row["prompt"] for row in before_test] == [
        row["prompt"] for row in after_test
    ]
    assert all(
        "net_return" not in row["prompt"] and "future" not in row["prompt"].lower()
        for row in before["pointwise"] + before["pairwise"]
    )
    assert "funding_latest" in before_test[0]["prompt"]
    assert '"hold_minutes":480' in before_test[0]["prompt"]


def test_quantiles_are_fitted_on_train_only_and_wait_wins_all_negative_entry() -> None:
    events = _events()
    first = builder.build_dataset(events, _market(), _funding(), min_utility_gap=0.005)
    changed = copy.deepcopy(events)
    for row in changed:
        if row["stage"] != "train":
            row["metadata"]["net_return"] *= 1_000_000
    second = builder.build_dataset(
        changed, _market(), _funding(), min_utility_gap=0.005
    )

    expected = np.quantile(
        np.asarray([-0.04, -0.02, 0.0, 0.02, 0.04]), [0.2, 0.4, 0.6, 0.8]
    ).tolist()
    assert first["thresholds"] == expected == second["thresholds"]
    test_pairs = [row for row in first["pairwise"] if row["stage"] == "test"]
    assert any(
        row["chosen"] == "WAIT" and row["rejected"] in {"A", "B"} for row in test_pairs
    )
    assert all(
        abs(row["metadata"]["utility_gap_a_minus_b"]) > 0.005 for row in test_pairs
    )


def test_file_build_is_byte_deterministic(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    market_path, funding_path = tmp_path / "market.csv", tmp_path / "funding.csv"
    events_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(_events()))
    )
    _market().to_csv(market_path, index=False)
    _funding().to_csv(funding_path, index=False)

    def run(suffix: str):
        cfg = builder.StateCardRankConfig(
            input_jsonls=str(events_path),
            market_csv=str(market_path),
            funding_csv=str(funding_path),
            pointwise_output=str(tmp_path / f"point-{suffix}.jsonl"),
            pairwise_output=str(tmp_path / f"pair-{suffix}.jsonl"),
            summary_output=str(tmp_path / f"summary-{suffix}.json"),
            min_utility_gap=0.005,
        )
        return builder.build(cfg), cfg

    report_a, cfg_a = run("a")
    report_b, cfg_b = run("b")
    assert report_a["sha256"] == report_b["sha256"]
    assert (
        Path(cfg_a.pointwise_output).read_bytes()
        == Path(cfg_b.pointwise_output).read_bytes()
    )
    assert (
        Path(cfg_a.pairwise_output).read_bytes()
        == Path(cfg_b.pairwise_output).read_bytes()
    )


def test_oos_pair_surface_and_wait_presence_do_not_depend_on_future_utility() -> None:
    events = _events()
    first = builder.build_dataset(events, _market(), _funding(), min_utility_gap=0.05)
    changed = copy.deepcopy(events)
    for row in changed:
        if row["stage"] == "test":
            row["metadata"]["net_return"] = 1.0
    second = builder.build_dataset(changed, _market(), _funding(), min_utility_gap=0.05)
    prompts_a = [row["prompt"] for row in first["pairwise"] if row["stage"] == "test"]
    prompts_b = [row["prompt"] for row in second["pairwise"] if row["stage"] == "test"]
    assert prompts_a == prompts_b
    assert any('"policy_id":"WAIT"' in prompt for prompt in prompts_a)
