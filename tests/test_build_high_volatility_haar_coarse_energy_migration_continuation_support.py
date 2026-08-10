import json
import math

import numpy as np
import pandas as pd

from training import build_high_volatility_haar_coarse_energy_migration_continuation_support as support


def test_haar_parseval_and_level_energy() -> None:
    values = np.arange(128, dtype=float) - 63.5
    energies, final_approximation = support.haar_pyramid(values)
    assert len(energies) == 7
    assert math.isclose(sum(energies) + final_approximation**2, float(values @ values), rel_tol=1e-12)

    alternating = np.tile([1.0, -1.0], 64)
    alternating_energies = support.haar_detail_energies(alternating)
    assert math.isclose(alternating_energies[0], float(alternating @ alternating), rel_tol=1e-12)
    assert np.allclose(alternating_energies[1:], 0.0)


def test_strict_prior_midrank_excludes_current_and_uses_only_finite_history() -> None:
    values = pd.Series([np.nan, *range(252), 251.0], dtype=float)
    ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[252])
    assert ranks.iloc[253] == (251.0 + 0.5) / 252.0


def _eligible_states() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2023-07-02T00:00:00Z", "2023-07-02T12:00:00Z", "2023-07-03T06:00:00Z"],
                utc=True,
            ),
            "source_valid": [True, True, True],
            "completed_return": [0.02, -0.01, 0.03],
            "realized_variation": [0.03, 0.04, 0.05],
            "coarse_energy_share": [0.4, 0.5, 0.6],
            "coarse_energy_migration": [0.1, 0.1, 0.1],
            "variation_rank": [0.8, 0.8, 0.8],
            "migration_rank": [0.9, 0.9, 0.9],
            "coarse_share_rank": [0.9, 0.9, 0.9],
        }
    )


def test_signal_clock_side_and_half_open_reservation() -> None:
    clock = support.build_clock(_eligible_states())
    assert clock.side.tolist() == [1, -1]
    assert clock.entry_time.tolist() == [
        pd.Timestamp("2023-07-02T00:05:00Z"),
        pd.Timestamp("2023-07-02T12:05:00Z"),
    ]
    assert clock.exit_time.iloc[0] == clock.entry_time.iloc[1]
    assert support.build_clock(_eligible_states(), "direction_flip").side.tolist() == [-1, 1]


def test_outcome_blind_contract_and_exact_controls() -> None:
    assert support.PREREG_SHA == "524df960b81c1791ea32b97521cbd17db0e3c8c008bc7f58469f309d68ac4900"
    assert support.CONTROLS == (
        "no_migration_gate",
        "no_volatility_gate",
        "coarse_share_level",
        "one_boundary_stale_migration",
        "direction_flip",
    )
    assert "SELECT ts,open,high,low,close" in support.QUERY
    forbidden = ("funding", "entry_price", "exit_price", "pnl", "postentry")
    query = support.QUERY.lower()
    assert not any(name in query for name in forbidden)

    preregistration = json.loads(support.prereg.DEFAULT_OUTPUT.read_text())
    assert preregistration["outcomes_opened"] is False
    assert preregistration["research_boundary"]["postentry_return_or_pnl_opened"] is False
