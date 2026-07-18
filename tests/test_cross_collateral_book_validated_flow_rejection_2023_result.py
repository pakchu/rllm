from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


RESULT = Path(
    "results/cross_collateral_book_validated_flow_rejection_"
    "selection_2023_2026-07-18.json"
)
DOCS = Path(
    "docs/cross-collateral-book-validated-flow-rejection-"
    "selection-2023-2026-07-18.md"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cbfr72_is_rejected_without_opening_later_years() -> None:
    payload = json.loads(RESULT.read_text())
    assert _sha256(RESULT) == "eddb580e558b1f1ca02d0c3f72d2a6896a87cddfd217b79cbe3184246399be76"
    assert _sha256(DOCS) == "f870e56fdae6c47039da7e693d04ae72116cc722498464cbe542b5cac8319fd6"
    assert payload["decision"] == "rejected_before_2024"
    assert payload["2024_test_opened"] is False
    assert payload["2025_eval_opened"] is False
    assert payload["2026_holdout_opened"] is False
    assert payload["evaluation"]["passes_2023_selection"] is False


def test_cbfr72_headline_and_significance_are_frozen() -> None:
    payload = json.loads(RESULT.read_text())
    annual = payload["evaluation"]["primary"]["2023"]
    assert annual["trades"] == 144
    assert annual["absolute_return_pct"] == pytest.approx(-15.684476538491399)
    assert annual["strict_mdd_pct"] == pytest.approx(16.775647893071977)
    assert annual["cagr_to_strict_mdd"] == pytest.approx(-0.9349550394991772)
    assert annual["funding_cash_pct_initial"] == pytest.approx(0.016538674462611792)
    signflip = payload["evaluation"]["weekly_cluster_signflip"]
    assert signflip["cluster_count"] == 46
    assert signflip["p_value_one_sided"] == pytest.approx(0.99350006499935)


def test_direction_flip_is_not_a_profitable_repair() -> None:
    payload = json.loads(RESULT.read_text())
    flipped = payload["evaluation"]["direction_flip"]
    assert flipped["absolute_return_pct"] == pytest.approx(-0.5355441937870964)
    assert flipped["mean_net_bps"] == pytest.approx(-0.26842353786677337)
    assert payload["evaluation"]["selection_gates"]["direction_flip_cagr_lower"] is False
    assert "no sign, threshold, hold" in payload["anti_repair"]
