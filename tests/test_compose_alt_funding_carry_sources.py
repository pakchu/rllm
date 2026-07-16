from __future__ import annotations

import pandas as pd
import pytest

from training.compose_alt_funding_carry_sources import (
    END,
    EXPECTED_PROTOCOL_HASH,
    HANDOFF,
    START,
    compose_market_dates,
)
from training.preregister_alt_funding_carry_harvest import canonical_hash, protocol


def test_protocol_and_composition_boundaries_are_frozen() -> None:
    assert canonical_hash(protocol()) == EXPECTED_PROTOCOL_HASH
    assert START == pd.Timestamp("2023-01-01")
    assert HANDOFF == pd.Timestamp("2024-01-01")
    assert END == pd.Timestamp("2026-01-01")


def test_composition_uses_old_before_handoff_and_recent_after() -> None:
    old = pd.Series(pd.date_range("2023-01-01", "2024-01-01", freq="5min"))
    recent = pd.Series(pd.date_range("2024-01-01", "2025-12-31 23:55", freq="5min"))
    combined = compose_market_dates(old, recent)
    assert combined.min() == START
    assert combined.max() == END - pd.Timedelta(minutes=5)
    assert len(combined) == 3 * 365 * 288 + 288


def test_composition_rejects_missing_bar() -> None:
    old = pd.Series(pd.date_range("2023-01-01", "2023-12-31 23:55", freq="5min")).drop(index=10)
    recent = pd.Series(pd.date_range("2024-01-01", "2025-12-31 23:55", freq="5min"))
    with pytest.raises(RuntimeError, match="not exact"):
        compose_market_dates(old, recent)
