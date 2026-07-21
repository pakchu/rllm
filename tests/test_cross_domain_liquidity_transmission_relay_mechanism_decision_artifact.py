from __future__ import annotations

import hashlib
from pathlib import Path


DECISION = Path(
    "docs/cross-domain-liquidity-transmission-relay-mechanism-decision-2026-07-21.md"
)
NETWORK_SOURCE = Path("data/coinmetrics_btc_network_daily_2020_2023.csv.gz")
NETWORK_MANIFEST = Path(
    "results/coinmetrics_btc_network_daily_pre2024_manifest_2026-07-16.json"
)
RRP_SOURCE = Path(
    "data/new_york_fed_overnight_rrp_2018_2023/"
    "new_york_fed_overnight_rrp_2018-01-01_2023-12-31.csv.gz"
)
RRP_MANIFEST = Path("data/new_york_fed_overnight_rrp_2018_2023/build_manifest.json")
CBOE_SOURCE = Path(
    "data/cboe_volatility_term_structure_2018_2023/"
    "cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz"
)
CBOE_MANIFEST = Path(
    "data/cboe_volatility_term_structure_2018_2023/build_manifest.json"
)
NETWORK_BUILDER = Path("training/download_coinmetrics_btc_network_daily.py")
RRP_BUILDER = Path("training/build_new_york_fed_overnight_rrp.py")
CBOE_BUILDER = Path("training/build_cboe_volatility_term_structure.py")

EXPECTED_HASHES = {
    DECISION: "970a114b7dab6b39bea8110264eb4ab05fd9794b5cb239bc643acb53619eebe5",
    NETWORK_SOURCE: (
        "97ab2ca9d0c347d85221b51734f98072763370072ca51f1c40e3214191159b42"
    ),
    NETWORK_MANIFEST: (
        "66b185769800c4732cf748b40ca9cb48c5eee239abf0425ff193c0688111c372"
    ),
    RRP_SOURCE: ("49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27"),
    RRP_MANIFEST: ("4f87e2219da71c94832c8708086ba01387efc145e3488b62cd3b3d07c62d8fee"),
    CBOE_SOURCE: ("6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7"),
    CBOE_MANIFEST: ("42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27"),
    NETWORK_BUILDER: (
        "73929fe4f7b8463ee187d008657cdad5be45df0d5f2c74ded1d541f61e87b763"
    ),
    RRP_BUILDER: ("0567157dde18b1c6ccfb37b669ceead521360f23dd0b73033fccc08e37c0d42c"),
    CBOE_BUILDER: ("0cd9fb50d6f665e9cc4f20539bacde328d1b7587b624ebc15f8a7b3489eeec2d"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cdltr_mechanism_and_source_dependencies_are_hash_frozen() -> None:
    for path, expected in EXPECTED_HASHES.items():
        assert _sha256(path) == expected


def test_cdltr_decision_freezes_causal_relay_and_execution_contract() -> None:
    text = DECISION.read_text(encoding="utf-8")
    normalized = " ".join(text.replace("**", "").replace("`", "").split())
    required_clauses = (
        "candidate-level new interaction, not a globally pristine hypothesis",
        "accepted_now - accepted_fifth_prior_slot",
        "A quarantine therefore breaks the baseline",
        "09:35 America/New_York on the next date present",
        "row exactly seven calendar days earlier",
        "network_vote = LONG if at least two metric votes are positive",
        "All state ages use actual UTC source-availability timestamps",
        "Each vote expires exactly 36 hours",
        "first network report with available_at strictly after the onset",
        "No later network report may retry that episode",
        "ceil_to_5m(decision_time) + 5 minutes",
        "exit: entry plus exactly 72 hours",
        "events are accepted in global chronological order",
        "source warm-up only: calendar 2020",
        "train clock: [2021-01-01, 2023-01-01) UTC",
        "selection clock: [2023-01-01, 2024-01-01) UTC",
        "2024 or later remains closed",
    )
    for clause in required_clauses:
        assert clause in normalized


def test_cdltr_decision_freezes_support_novelty_controls_and_llm_boundary() -> None:
    text = DECISION.read_text(encoding="utf-8")
    normalized = " ".join(text.replace("**", "").replace("`", "").split())
    required_clauses = (
        "train: at least 60 events, at least 25 in each calendar year",
        "at least 12 in each half-year",
        "selection: at least 30 events and at least 12 in each half-year",
        "maximum UTC calendar-month share: 20% in each split",
        "maximum UTC weekday share: 35% in each split",
        "macro_only",
        "network_only",
        "reverse_order",
        "one_network_report_delay",
        "direction_flip",
        "deterministic_random_side",
        "decision-date Jaccard > 0.30",
        "fraction of CDLTR dates within +/-1 UTC day > 0.50",
        "absolute signed occupied-exposure Pearson > 0.40",
        "Only an unchanged source-support and novelty pass",
        "constrained TRADE/ABSTAIN veto",
        "not authorized to create direction, alter the relay, search the hold",
        "penalize strict drawdown and turnover",
        "Implement and test a preregistration artifact",
    )
    for clause in required_clauses:
        assert clause in normalized


def test_cdltr_decision_keeps_all_outcomes_closed() -> None:
    text = DECISION.read_text(encoding="utf-8")
    normalized = " ".join(text.replace("**", "").replace("`", "").split())
    assert "opens no source values, feature incidence, BTC market row" in normalized
    assert "funding value, return, PnL, equity, CAGR, MDD" in normalized
    assert "No BTC price, return, exchange flow, funding" in normalized
    assert "Commit it before any real CDLTR feature or event incidence" in normalized
