from __future__ import annotations

import pandas as pd
import pytest

from training import (
    preregister_psim_d8_rllm2_s7_2021_report_only_transfer as prereg,
)
from training import psim_report_only_transfer as transfer

FIVE = pd.Timedelta(minutes=5)
STEP = pd.Timedelta(hours=6)
START = pd.Timestamp("2021-01-04T00:00:00Z")
HALF = START + pd.Timedelta(days=11)
END = START + pd.Timedelta(days=22)


def _alpha_target(index: int) -> str:
    return "TARGET_LONG" if index % 2 == 0 else "TARGET_SHORT"


def _market() -> pd.DataFrame:
    periods = int((END - START) / FIVE)
    opens: list[float] = []
    price = 100.0
    bars_per_decision = int(STEP / FIVE)
    for index in range(periods):
        opens.append(price)
        decision = index // bars_per_decision
        week = index // int(pd.Timedelta(days=7) / FIVE)
        edge = 0.00013 + 0.00002 * (week % 3)
        price *= (
            1.0 + edge
            if _alpha_target(decision) == "TARGET_LONG"
            else 1.0 - edge
        )
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                START,
                periods=periods,
                freq="5min",
                tz="UTC",
            ),
            "open": opens,
            "high": opens,
            "low": opens,
            "close": opens,
        }
    )


def _funding() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [],
            "settlement_mark": [],
            "funding_rate": [],
        }
    )


def _schedule(policy_id: str, *, alpha: bool) -> pd.DataFrame:
    rows = []
    current = START
    index = 0
    while current < END - FIVE:
        rows.append(
            {
                "policy_id": policy_id,
                "sequence_id": f"seq-{index:04d}",
                "entry_time": current,
                "target": (
                    _alpha_target(index) if alpha else "TARGET_FLAT"
                ),
            }
        )
        current += STEP
        index += 1
    return pd.DataFrame(rows, columns=transfer.SCHEDULE_COLUMNS)


def _families() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha_ids = {
        prereg.PRIMARY_POLICY_ID,
        f"{prereg.PRIMARY_POLICY_ID}_action_code_permutation",
    }

    def group(ids: tuple[str, ...]) -> pd.DataFrame:
        return pd.concat(
            [
                _schedule(policy_id, alpha=policy_id in alpha_ids)
                for policy_id in ids
            ],
            ignore_index=True,
        )

    primary = _schedule(prereg.PRIMARY_POLICY_ID, alpha=True)
    delayed = primary.copy()
    delayed["sequence_id"] = (
        delayed["sequence_id"].astype(str) + ":delay_5m"
    )
    delayed["entry_time"] = (
        pd.to_datetime(delayed["entry_time"], utc=True) + FIVE
    )
    return (
        group(prereg.s4.POLICY_FAMILY_IDS),
        group(prereg.s5.POLICY_FAMILY_IDS),
        group(prereg.s6r1.POLICY_FAMILY_IDS),
        delayed,
    )


def test_combined_family_and_delay_identity_are_strict() -> None:
    s4, s5, s6, delayed = _families()
    rows = len(_schedule(prereg.PRIMARY_POLICY_ID, alpha=True))
    combined = transfer.combine_schedule_family(
        s4,
        s5,
        s6,
        expected_rows_per_policy=rows,
    )

    assert tuple(combined["policy_id"].drop_duplicates()) == (
        prereg.FAMILY_IDS
    )
    transfer.validate_delayed_primary(
        combined,
        delayed,
        expected_rows=rows,
    )
    broken = delayed.copy()
    broken.loc[0, "entry_time"] = broken.loc[0, "entry_time"] + FIVE
    with pytest.raises(ValueError, match="identity changed"):
        transfer.validate_delayed_primary(
            combined,
            broken,
            expected_rows=rows,
        )


def test_family_order_and_action_domain_are_closed() -> None:
    s4, s5, s6, _ = _families()
    rows = len(_schedule(prereg.PRIMARY_POLICY_ID, alpha=True))
    reordered = pd.concat(
        [
            s4.loc[s4["policy_id"].ne(prereg.s4.POLICY_FAMILY_IDS[0])],
            s4.loc[s4["policy_id"].eq(prereg.s4.POLICY_FAMILY_IDS[0])],
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="family order"):
        transfer.combine_schedule_family(
            reordered,
            s5,
            s6,
            expected_rows_per_policy=rows,
        )
    s6.loc[s6.index[0], "target"] = "TARGET_BOGUS"
    with pytest.raises(RuntimeError, match="combined schedule"):
        transfer.combine_schedule_family(
            s4,
            s5,
            s6,
            expected_rows_per_policy=rows,
        )


def test_shared_max_stat_is_deterministic_and_family_adjusted() -> None:
    weekly = {
        policy_id: [
            ("2021-01-04T00:00:00Z", 0.01 if index == 0 else 0.0),
            ("2021-01-11T00:00:00Z", 0.02 if index == 0 else 0.0),
            ("2021-01-18T00:00:00Z", 0.015 if index == 0 else 0.0),
        ]
        for index, policy_id in enumerate(prereg.FAMILY_IDS)
    }
    cfg = transfer.StatisticalConfig(draws=1_000, seed=7, batch_draws=200)
    first = transfer.shared_weekly_max_stat(weekly, cfg=cfg)
    second = transfer.shared_weekly_max_stat(weekly, cfg=cfg)

    assert first == second
    assert first["family_ids"] == list(prereg.FAMILY_IDS)
    for policy_id in prereg.FAMILY_IDS:
        assert first["p_max"][policy_id] >= first["local_p"][policy_id]


def test_nonfinite_economic_metric_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transfer.economics,
        "simulate_target_schedule",
        lambda *_args, **_kwargs: {
            "final_equity": 1.0,
            "cagr": float("nan"),
            "max_drawdown": 0.0,
            "cost_rate": 0.0006,
            "intervals": [],
            "weekly_log_returns": [],
        },
    )

    with pytest.raises(ValueError, match="CAGR is not finite"):
        transfer.evaluate_one(
            _market(),
            _funding(),
            _schedule(prereg.PRIMARY_POLICY_ID, alpha=True),
            start=START,
            end=END,
            cost_rate=0.0006,
        )


def test_full_transfer_reports_absolute_cagr_mdd_trades_and_gate() -> None:
    s4, s5, s6, delayed = _families()
    rows = len(_schedule(prereg.PRIMARY_POLICY_ID, alpha=True))
    result = transfer.evaluate_transfer(
        _market(),
        _funding(),
        s4,
        s5,
        s6,
        delayed,
        start=START,
        half_split=HALF,
        end=END,
        expected_rows_per_policy=rows,
        statistical_config=transfer.StatisticalConfig(
            draws=2_000,
            seed=11,
            batch_draws=200,
        ),
    )

    base = result["primary_metrics"]["base_6bp"]
    assert base["absolute_return"] > 0
    assert base["cagr"] > 0
    assert base["strict_mdd"] >= 0
    assert base["cagr_to_strict_mdd"] > 1
    assert base["directional_entries_including_flips"] >= 80
    assert base["all_target_changes_including_terminal_flatten"] >= 81
    assert result["primary_metrics"]["stress_10bp"]["absolute_return"] > 0
    assert result["primary_metrics"]["delayed_5m_6bp"][
        "absolute_return"
    ] > 0
    assert result["primary_metrics"]["first_half_6bp"][
        "absolute_return"
    ] > 0
    assert result["primary_metrics"]["second_half_6bp"][
        "absolute_return"
    ] > 0
    assert result["action_code_permutation_schedule_identity"] is True
    assert result["robustness_semantics"] == {
        "half_metrics": (
            "standalone_reset_to_flat_equity_1_at_each_half_start"
        ),
        "continuous_full_path_subperiod_attribution": False,
    }
    assert result["gate"]["passed"] is True
    assert all(result["gate"]["checks"].values())
