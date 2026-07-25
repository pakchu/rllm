from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from training import freeze_block_clearing_target_position_evaluator as freeze


def test_closed_form_quantity_matches_independent_bisection() -> None:
    generator = np.random.default_rng(20_260_725)
    for _ in range(1_000):
        equity = float(generator.uniform(0.25, 10.0))
        price = float(generator.uniform(100.0, 100_000.0))
        old_exposure = float(generator.uniform(-0.85, 0.85))
        old_quantity = old_exposure * equity / price
        cost = float(generator.choice([0.0, 0.0006, 0.0010]))
        target = float(generator.choice(freeze.ACTIONS))
        analytic = freeze.solve_target_quantity(
            equity,
            old_quantity,
            price,
            cost,
            target,
        )
        numeric = freeze.solve_target_quantity_bisection(
            equity,
            old_quantity,
            price,
            cost,
            target,
        )
        assert analytic == pytest.approx(
            numeric,
            rel=1e-10,
            abs=1e-12,
        )
        if target:
            post_equity = equity - cost * abs(
                analytic - old_quantity
            ) * price
            assert analytic * price / post_equity == pytest.approx(
                target,
                rel=1e-12,
                abs=1e-12,
            )


def test_quantity_solver_charges_reversal_once() -> None:
    equity = 1.0
    price = 20_000.0
    old_quantity = 0.5 * equity / price
    new_quantity = freeze.solve_target_quantity(
        equity,
        old_quantity,
        price,
        0.0006,
        -0.5,
    )
    changed_notional = abs(new_quantity - old_quantity) * price
    post_equity = equity - 0.0006 * changed_notional
    assert new_quantity < 0.0
    assert new_quantity * price / post_equity == pytest.approx(-0.5)
    assert changed_notional < 1.01


@pytest.mark.parametrize("target", [-0.5, 0.5])
@pytest.mark.parametrize("cost", [0.0, 0.0006, 0.0010])
def test_quantity_solver_accepts_same_target_kink_without_duplicate_branch(
    target: float,
    cost: float,
) -> None:
    equity = 1.0
    price = 20_000.0
    old_quantity = target / price
    new_quantity = freeze.solve_target_quantity(
        equity,
        old_quantity,
        price,
        cost,
        target,
    )
    assert new_quantity == pytest.approx(old_quantity)


@pytest.mark.parametrize(
    "values",
    [
        (float("nan"), 0.0, 1.0, 0.0006, 0.5),
        (1.0, 0.0, 0.0, 0.0006, 0.5),
        (1.0, 0.0, 1.0, -0.1, 0.5),
        (1.0, 0.0, 1.0, 0.0006, 0.25),
    ],
)
def test_quantity_solver_fails_closed(values: tuple[float, ...]) -> None:
    assert freeze.solve_target_quantity(*values) == 0.0


def test_conservative_boundary_funding_retains_at_most_one_debit() -> None:
    mark = 50_000.0
    rate = 0.0001
    long_quantity = 0.5 / mark
    short_quantity = -0.5 / mark
    assert freeze.conservative_boundary_funding_cash(
        long_quantity,
        short_quantity,
        mark,
        rate,
    ) == pytest.approx(-0.00005)
    assert freeze.conservative_boundary_funding_cash(
        short_quantity,
        0.0,
        mark,
        rate,
    ) == 0.0


def test_transition_utility_is_exactly_frozen() -> None:
    expected = math.log(1.02) - (1.0 / 3.0) * 0.03 - 0.001 * 1.0
    assert freeze.transition_utility(
        1.02, 0.03, 0.5, -0.5
    ) == pytest.approx(
        expected
    )
    with pytest.raises(ValueError):
        freeze.transition_utility(0.0, 0.0, 0.0, 0.0)


def _family(
    first: np.ndarray,
    *,
    second: np.ndarray | None = None,
) -> dict[str, list[tuple[str, float]]]:
    start = datetime(2020, 1, 6, tzinfo=timezone.utc)
    keys = [
        (start + timedelta(days=7 * index))
        .isoformat()
        .replace("+00:00", "Z")
        for index in range(len(first))
    ]
    zero = np.zeros_like(first)
    values = {
        variant: list(zip(keys, zero.tolist()))
        for variant in freeze.FAMILY_IDS
    }
    values["categorical_linear_fqi"] = list(
        zip(keys, first.tolist())
    )
    if second is not None:
        values["categorical_ridge_fqi"] = list(
            zip(keys, second.tolist())
        )
    return values


def test_shared_max_stat_is_deterministic_and_familywise() -> None:
    first = np.asarray([0.01, 0.02, 0.01, 0.03, 0.02])
    second = np.asarray([0.005, 0.01, -0.005, 0.01, 0.005])
    family = _family(first, second=second)
    left = freeze.shared_weekly_max_stat(family)
    right = freeze.shared_weekly_max_stat(family)
    assert left == right
    primary = "categorical_linear_fqi"
    assert left["p_max"][primary] >= left["local_p"][primary]
    assert left["method"] == "exact_rademacher_enumeration"
    assert left["shared_signs"] is True
    assert left["observed_t"]["always_flat"] is None


def test_shared_max_stat_calculates_finite_negative_statistic() -> None:
    family = _family(np.asarray([-0.10, 0.09, 0.01, -0.02, -0.01]))
    result = freeze.shared_weekly_max_stat(family)
    primary = "categorical_linear_fqi"
    assert result["observed_t"][primary] < 0.0
    assert 0.0 < result["local_p"][primary] < 1.0
    assert result["p_max"][primary] >= result["local_p"][primary]


def test_shared_max_stat_rejects_family_order_drift() -> None:
    values = _family(np.asarray([0.1, 0.2]))
    reordered = dict(reversed(tuple(values.items())))
    with pytest.raises(ValueError, match="family order"):
        freeze.shared_weekly_max_stat(reordered)


def test_shared_max_stat_rejects_same_length_misaligned_weeks() -> None:
    values = _family(np.asarray([0.1, 0.2]))
    variant = freeze.FAMILY_IDS[-1]
    values[variant] = [
        ("2020-01-13T00:00:00Z", 0.0),
        ("2020-01-20T00:00:00Z", 0.0),
    ]
    with pytest.raises(ValueError, match="misaligned"):
        freeze.shared_weekly_max_stat(values)


def test_shared_max_stat_rejects_non_utc_monday_key() -> None:
    values = _family(np.asarray([0.1, 0.2]))
    for variant in freeze.FAMILY_IDS:
        values[variant][0] = ("2020-01-06T09:00:00+09:00", 0.0)
    with pytest.raises(ValueError, match="keys are invalid"):
        freeze.shared_weekly_max_stat(values)


def test_manifest_binds_no_outcome_access(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    original = freeze.sha256_file

    def guarded(path: str) -> str:
        text = str(path)
        if text in (str(freeze.MARKET), str(freeze.FUNDING)):
            opened.append(text)
            raise AssertionError("outcome payload was hashed")
        return original(path)

    monkeypatch.setattr(freeze, "sha256_file", guarded)
    payload = freeze.build_manifest(freeze_commit="0" * 40)
    freeze.validate_manifest(payload)
    assert opened == []
    assert payload["outcome_boundary"] == {
        "market_rows_parsed": 0,
        "funding_rows_parsed": 0,
        "market_or_funding_payload_bytes_hashed": False,
        "future_returns_created": 0,
        "rewards_created": 0,
        "models_fit": 0,
        "economic_metrics_computed": 0,
        "bctp_2023_market_outcomes_opened": False,
        "post_2023_source_or_outcomes_opened": False,
    }
    assert payload["mutable_parameters"] == []
    assert tuple(payload["family_ids"]) == freeze.FAMILY_IDS
    assert payload["family_size"] == 31
    changed = dict(payload)
    changed["cheap_policy"] = {
        **payload["cheap_policy"],
        "discount": 0.98,
    }
    changed["manifest_hash"] = freeze.canonical_hash(
        {
            key: value
            for key, value in changed.items()
            if key != "manifest_hash"
        }
    )
    with pytest.raises(ValueError, match="frozen contract"):
        freeze.validate_manifest(changed)


def test_official_freeze_requires_committed_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(freeze, "_git", lambda *args: Result())
    with pytest.raises(RuntimeError, match="not committed"):
        freeze.freeze_evaluator(tmp_path / "freeze.json")
