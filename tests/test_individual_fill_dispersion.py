from __future__ import annotations

from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import pytest

from preprocessing.individual_fill_dispersion import (
    OUTPUT_COLUMNS,
    RAW_AGGTRADE_COLUMNS,
    RAW_TRADE_COLUMNS,
    aggregate_day,
)

DAY = date(2020, 1, 1)
DAY_START_MS = int(cast(pd.Timestamp, pd.Timestamp(DAY, tz="UTC")).timestamp() * 1_000)


def trades(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(RAW_TRADE_COLUMNS))


def aggtrades(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(RAW_AGGTRADE_COLUMNS))


def base_trade(
    trade_id: int,
    quote_qty: float,
    *,
    millis: int = 0,
    maker: bool = False,
    price: float = 100.0,
) -> dict[str, object]:
    return {
        "trade_id": trade_id,
        "price": price,
        "quantity": quote_qty / price,
        "quote_qty": quote_qty,
        "transact_time": DAY_START_MS + millis,
        "is_buyer_maker": maker,
    }


def base_agg(
    agg_trade_id: int,
    notional: float,
    *,
    first_trade_id: int,
    last_trade_id: int,
    millis: int = 0,
    maker: bool = False,
    price: float = 100.0,
) -> dict[str, object]:
    return {
        "agg_trade_id": agg_trade_id,
        "price": price,
        "quantity": notional / price,
        "first_trade_id": first_trade_id,
        "last_trade_id": last_trade_id,
        "transact_time": DAY_START_MS + millis,
        "is_buyer_maker": maker,
    }


def norm_eff(values: list[float]) -> float:
    total = sum(values)
    if not values or total == 0:
        return 0.0
    hhi = sum(value * value for value in values) / (total * total)
    return (1.0 / hhi) / len(values)


def test_complete_utc_daily_grid_and_empty_verified_bins_are_zero() -> None:
    output = aggregate_day(trades([]), aggtrades([]), DAY)

    assert list(output.columns) == list(OUTPUT_COLUMNS)
    assert len(OUTPUT_COLUMNS) == 72
    assert len(output) == 288
    assert str(output.loc[0, "date"].tzinfo) == "UTC"
    assert output.loc[0, "date"] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert output.loc[287, "date"] == pd.Timestamp("2020-01-01T23:55:00Z")
    assert bool(output["source_complete"].eq(True).all())
    assert bool(output["source_observed"].eq(False).all())
    assert bool(output["verified_zero_volume_empty"].eq(True).all())

    numeric = output.drop(
        columns=[
            "date",
            "source_observed",
            "source_complete",
            "source_gap_day",
            "verified_zero_volume_empty",
            "post_gap_quarantine",
        ]
    )
    assert (numeric.to_numpy() == 0).all()


def test_frozen_ifda_formula_uses_side_fill_hhi_and_quote_qty() -> None:
    individual = trades(
        [
            base_trade(10, 100.0, millis=1_000, maker=False),
            base_trade(11, 100.0, millis=2_000, maker=False),
            base_trade(12, 99.0, millis=3_000, maker=True),
            base_trade(13, 1.0, millis=4_000, maker=True),
        ]
    )
    aggregate = aggtrades(
        [
            base_agg(1, 200.0, first_trade_id=10, last_trade_id=11, millis=1_500, maker=False),
            base_agg(2, 99.0, first_trade_id=12, last_trade_id=12, millis=3_500, maker=True),
            base_agg(3, 1.0, first_trade_id=13, last_trade_id=13, millis=4_500, maker=True),
        ]
    )

    row = aggregate_day(individual, aggregate, DAY).iloc[0]

    buy_norm = norm_eff([100.0, 100.0])
    sell_norm = norm_eff([99.0, 1.0])
    flow_coherence = 100.0 / 300.0
    expected_gap = buy_norm - sell_norm
    expected_score = flow_coherence * buy_norm * expected_gap

    assert bool(row["source_observed"]) is True
    assert row["first_trade_id"] == 10
    assert row["last_trade_id"] == 13
    assert row["fill_count"] == 4
    assert row["buy_fill_count"] == 2
    assert row["sell_fill_count"] == 2
    assert row["quote_notional"] == pytest.approx(300.0)
    assert row["buy_quote_notional"] == pytest.approx(200.0)
    assert row["sell_quote_notional"] == pytest.approx(100.0)
    assert row["signed_quote_notional"] == pytest.approx(100.0)
    assert row["flow_coherence"] == pytest.approx(flow_coherence)
    assert row["buy_fill_hhi"] == pytest.approx(0.5)
    assert row["buy_fill_effective_count"] == pytest.approx(2.0)
    assert row["buy_fill_normalized_effective_count"] == pytest.approx(buy_norm)
    assert row["sell_fill_hhi"] == pytest.approx((99.0**2 + 1.0) / 100.0**2)
    assert row["sell_fill_effective_count"] == pytest.approx(1.0 / ((99.0**2 + 1.0) / 100.0**2))
    assert row["sell_fill_normalized_effective_count"] == pytest.approx(sell_norm)
    assert row["all_fill_hhi"] == pytest.approx((100**2 + 100**2 + 99**2 + 1) / 300**2)
    assert row["all_fill_effective_count"] == pytest.approx(1.0 / row["all_fill_hhi"])
    assert row["all_fill_normalized_effective_count"] == pytest.approx(
        row["all_fill_effective_count"] / 4.0
    )
    assert row["dominant_side"] == 1
    assert row["dominant_quote_notional"] == pytest.approx(200.0)
    assert row["opposing_quote_notional"] == pytest.approx(100.0)
    assert row["dominant_equalization"] == pytest.approx(buy_norm)
    assert row["opposing_equalization"] == pytest.approx(sell_norm)
    assert row["equalization_gap"] == pytest.approx(expected_gap)
    assert row["ifda_score"] == pytest.approx(expected_score)

    agg_sell_norm = norm_eff([99.0, 1.0])
    agg_expected_gap = 1.0 - agg_sell_norm
    assert row["agg_event_count"] == 3
    assert row["agg_buy_event_count"] == 1
    assert row["agg_sell_event_count"] == 2
    assert row["agg_buy_quote_notional"] == pytest.approx(200.0)
    assert row["agg_sell_quote_notional"] == pytest.approx(100.0)
    assert row["agg_buy_event_hhi"] == pytest.approx(1.0)
    assert row["agg_buy_event_normalized_effective_count"] == pytest.approx(1.0)
    assert row["agg_sell_event_hhi"] == pytest.approx((99.0**2 + 1.0) / 100.0**2)
    assert row["agg_dominant_equalization"] == pytest.approx(1.0)
    assert row["agg_opposing_equalization"] == pytest.approx(agg_sell_norm)
    assert row["agg_equalization_gap"] == pytest.approx(agg_expected_gap)
    assert row["agg_ifda_score"] == pytest.approx(flow_coherence * 1.0 * agg_expected_gap)

    empty = aggregate_day(individual, aggregate, DAY).iloc[1]
    assert bool(empty["source_complete"]) is True
    assert bool(empty["verified_zero_volume_empty"]) is True
    assert empty["fill_count"] == 0
    assert empty["quote_notional"] == 0
    assert empty["ifda_score"] == 0


def test_equal_and_eccentric_side_hhi_cases() -> None:
    equal = aggregate_day(
        trades(
            [
                base_trade(1, 25.0, millis=0, maker=False),
                base_trade(2, 25.0, millis=1, maker=False),
                base_trade(3, 25.0, millis=2, maker=True),
                base_trade(4, 25.0, millis=3, maker=True),
            ]
        ),
        aggtrades([]),
        DAY,
    ).iloc[0]
    assert equal["buy_fill_hhi"] == pytest.approx(0.5)
    assert equal["sell_fill_hhi"] == pytest.approx(0.5)
    assert equal["buy_fill_normalized_effective_count"] == pytest.approx(1.0)
    assert equal["sell_fill_normalized_effective_count"] == pytest.approx(1.0)
    assert equal["all_fill_hhi"] == pytest.approx(0.25)
    assert equal["all_fill_effective_count"] == pytest.approx(4.0)
    assert equal["all_fill_normalized_effective_count"] == pytest.approx(1.0)
    assert equal["dominant_side"] == 0
    assert equal["dominant_equalization"] == pytest.approx(0.0)
    assert equal["opposing_equalization"] == pytest.approx(0.0)
    assert equal["equalization_gap"] == pytest.approx(0.0)
    assert equal["ifda_score"] == pytest.approx(0.0)

    eccentric = aggregate_day(
        trades(
            [
                base_trade(1, 97.0, millis=0, maker=True),
                base_trade(2, 1.0, millis=1, maker=True),
                base_trade(3, 1.0, millis=2, maker=True),
                base_trade(4, 1.0, millis=3, maker=True),
            ]
        ),
        aggtrades([]),
        DAY,
    ).iloc[0]
    sell_hhi = (97**2 + 1 + 1 + 1) / 100**2
    sell_norm = (1.0 / sell_hhi) / 4.0
    assert eccentric["dominant_side"] == -1
    assert eccentric["signed_quote_notional"] == pytest.approx(-100.0)
    assert eccentric["flow_coherence"] == pytest.approx(1.0)
    assert eccentric["buy_fill_hhi"] == pytest.approx(0.0)
    assert eccentric["sell_fill_hhi"] == pytest.approx(sell_hhi)
    assert eccentric["sell_fill_effective_count"] == pytest.approx(1.0 / sell_hhi)
    assert eccentric["sell_fill_normalized_effective_count"] == pytest.approx(sell_norm)
    assert eccentric["dominant_equalization"] == pytest.approx(sell_norm)
    assert eccentric["opposing_equalization"] == pytest.approx(0.0)
    assert eccentric["equalization_gap"] == pytest.approx(sell_norm)
    assert eccentric["ifda_score"] == pytest.approx(sell_norm * sell_norm)


def test_quote_qty_reconciliation_accepts_official_cent_boundary_and_rejects_above() -> None:
    ok = trades(
        [
            {
                "trade_id": 1,
                "price": 7189.43,
                "quantity": 0.03,
                "quote_qty": 215.68,
                "transact_time": DAY_START_MS,
                "is_buyer_maker": False,
            },
            {
                "trade_id": 2,
                "price": 100.0,
                "quantity": 1.0,
                "quote_qty": 100.011,
                "transact_time": DAY_START_MS + 1,
                "is_buyer_maker": True,
            },
        ]
    )
    output = aggregate_day(ok, aggtrades([]), DAY)
    assert output.loc[0, "quote_notional"] == pytest.approx(315.691)

    bad = ok.copy()
    bad.loc[1, "quote_qty"] = 100.0111
    with pytest.raises(ValueError, match="quote_qty"):
        aggregate_day(bad, aggtrades([]), DAY)


def test_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match=r"exactly \+1 continuous"):
        aggregate_day(
            trades([base_trade(1, 10.0), base_trade(3, 10.0, millis=1)]),
            aggtrades([]),
            DAY,
        )

    with pytest.raises(ValueError, match="boolean maker"):
        aggregate_day(
            trades([{**base_trade(1, 10.0), "is_buyer_maker": "yes"}]),
            aggtrades([]),
            DAY,
        )

    with pytest.raises(ValueError, match="within the requested UTC day"):
        aggregate_day(
            trades([{**base_trade(1, 10.0), "transact_time": DAY_START_MS - 1}]),
            aggtrades([]),
            DAY,
        )

    with pytest.raises(ValueError, match="positive finite"):
        aggregate_day(
            trades([{**base_trade(1, 10.0), "price": np.inf}]),
            aggtrades([]),
            DAY,
        )

    with pytest.raises(ValueError, match="monotone, unique"):
        aggregate_day(
            trades([]),
            aggtrades(
                [
                    base_agg(2, 10.0, first_trade_id=1, last_trade_id=1),
                    base_agg(2, 10.0, first_trade_id=2, last_trade_id=2, millis=1),
                ]
            ),
            DAY,
        )

    with pytest.raises(ValueError, match=r"aggregate trade IDs must be exactly \+1"):
        aggregate_day(
            trades([]),
            aggtrades(
                [
                    base_agg(1, 10.0, first_trade_id=1, last_trade_id=1),
                    base_agg(3, 10.0, first_trade_id=2, last_trade_id=2, millis=1),
                ]
            ),
            DAY,
        )
