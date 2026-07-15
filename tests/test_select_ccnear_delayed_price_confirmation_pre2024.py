from __future__ import annotations

import pandas as pd

from training import select_ccnear_delayed_price_confirmation_pre2024 as selector


def test_confirmation_is_single_fixed_rule() -> None:
    assert selector.CONFIRMATION_BARS == 6
    assert selector.HOLD_BARS == 282
    assert selector.CONFIRMATION_BARS + selector.HOLD_BARS == 288


def test_all_windows_remain_pre2024() -> None:
    assert max(pd.Timestamp(end) for _, end in selector.WINDOWS.values()) == pd.Timestamp(
        "2024-01-01"
    )
