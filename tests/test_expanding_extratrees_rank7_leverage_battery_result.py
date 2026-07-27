from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import (
    evaluate_expanding_extratrees_rank7_leverage_battery as battery,
)

RESULT = Path(
    "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json"
)
SUMMARY = Path(
    "docs/expanding-extratrees-rank7-leverage-battery-2026-07-27.md"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_self_hashed_and_bound_to_committed_runner() -> None:
    payload = _payload()
    result_hash = payload.pop("result_hash")

    assert result_hash == (
        "9eb4f1a850a164667222e7f43474232110631e594875c9949aac21dc6caab148"
    )
    assert battery.canonical_hash(payload) == result_hash
    assert _sha256(RESULT) == (
        "e079f7a70d4e5eea7de962cf5daad93fd634fdf5779854d1783f83a837dc41ab"
    )
    assert _sha256(SUMMARY) == (
        "b954cd13026aaff9d1797a61d91ca05dbf3a2d4fa894caf450af3f426097103f"
    )
    assert payload["execution"] == {
        "git_head": "961595ea8761c75c1a027dc21d5b4400671796c8",
        "origin_main": "961595ea8761c75c1a027dc21d5b4400671796c8",
        "runner_sha256": (
            "df54e27f2de30d4ed81ab6f4dcc58ee2f729096f00c47d73760aded92f3e0bbe"
        ),
    }


def test_selection_is_pre2025_only_and_chooses_highest_passing_cell() -> None:
    payload = _payload()
    cells = payload["selection_grid"]

    assert [cell["leverage"] for cell in cells] == [0.5, 0.75, 1.0, 1.25, 1.5]
    assert all(cell["passes"] is True for cell in cells)
    assert payload["selected_leverage"] == 1.5
    assert payload["integrity"]["selection_uses_only_pre_2025_windows"] is True
    assert payload["integrity"]["future_repair_or_reselection"] is False
    for cell in cells:
        assert set(cell["base"]) == {"2023", "2024", "selection"}
        assert set(cell["stress"]) == {"2023", "2024", "selection"}
        assert set(cell["schedule_hashes"]) == {"2023", "2024", "selection"}

    for window in ("2023", "2024", "selection"):
        hashes = {cell["schedule_hashes"][window] for cell in cells}
        assert len(hashes) == 1


def test_fixed_report_only_target_metrics_and_significance() -> None:
    payload = _payload()
    assert payload["verdict"] == "TARGET_HIT_PROTOCOL_ISOLATED"
    assert payload["robustness_pass"] is True
    assert payload["robustness_failure_reasons"] == []
    assert payload["user_target_hit"] is True
    assert payload["user_target_failure_reasons"] == []

    expected = {
        "2025": (55.591824399946496, 55.63894238020619, 13.834946101651834, 4.0216233566362165, 21),
        "2026h1": (22.865293771966687, 64.0192771972065, 12.467372943366195, 5.134945227676913, 12),
        "future": (91.16835213415683, 58.057346326473414, 13.834946101651834, 4.1964273586538665, 33),
        "all": (326.50946692342717, 52.88333052570388, 13.834946101651857, 3.8224457209406655, 74),
    }
    for window, values in expected.items():
        row = payload["report_only"]["base"][window]
        assert row["absolute_return_pct"] == pytest.approx(values[0])
        assert row["cagr_pct"] == pytest.approx(values[1])
        assert row["strict_mdd_pct"] == pytest.approx(values[2])
        assert row["cagr_to_strict_mdd"] == pytest.approx(values[3])
        assert row["trades"] == values[4]

    stress = payload["report_only"]["stress_10bp_per_side"]
    assert stress["future"]["absolute_return_pct"] == pytest.approx(
        83.73727542756136
    )
    assert stress["all"]["absolute_return_pct"] == pytest.approx(
        290.2267079850334
    )
    significance = payload["report_only"]["significance"]
    assert significance["future"]["weekly_cluster_sign_flip"][
        "p_value_one_sided"
    ] == pytest.approx(0.006539869202615948)
    assert significance["future"]["stationary_trade_bootstrap"][
        "one_sided_p_value"
    ] == pytest.approx(5.999880002399952e-05)
    assert significance["all"]["weekly_cluster_sign_flip"][
        "p_value_one_sided"
    ] == pytest.approx(1.999960000799984e-05)
    assert significance["all"]["stationary_trade_bootstrap"][
        "one_sided_p_value"
    ] == pytest.approx(1.999960000799984e-05)


def test_summary_discloses_protocol_isolation_and_absolute_returns() -> None:
    summary = SUMMARY.read_text(encoding="utf-8")

    assert "not globally pristine discovery OOS" in summary
    assert "future repair/reselection: `False`" in summary
    assert "| 2025 | 55.59% | 55.64% | 13.83% | 4.02 | 21 | 51.72% |" in summary
    assert "| all | 326.51% | 52.88% | 13.83% | 3.82 | 74 | 290.23% |" in summary
