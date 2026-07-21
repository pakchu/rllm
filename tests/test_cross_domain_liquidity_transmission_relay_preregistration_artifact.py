from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_cross_domain_liquidity_transmission_relay as prereg


MECHANISM = Path(
    "docs/cross-domain-liquidity-transmission-relay-mechanism-decision-2026-07-21.md"
)
AMENDMENT = Path("docs/cdltr72a-preincidence-comparator-amendment-2026-07-21.md")
DOCUMENT = Path(
    "docs/cross-domain-liquidity-transmission-relay-preregistration-2026-07-21.md"
)
SOURCE = Path("training/preregister_cross_domain_liquidity_transmission_relay.py")
COMPARATOR_CLOCK = Path("results/cdltr_prior_comparator_views_2026-07-21.csv.gz")
COMPARATOR_MANIFEST = Path(
    "results/cdltr_prior_comparator_views_manifest_2026-07-21.json"
)
ARTIFACT = Path(
    "results/cross_domain_liquidity_transmission_relay_preregistration_2026-07-21.json"
)
EXPECTED_HASHES = {
    MECHANISM: "970a114b7dab6b39bea8110264eb4ab05fd9794b5cb239bc643acb53619eebe5",
    AMENDMENT: "fba002d78e0c29d5824d2bfd922d74c1d5477f2eb63f55959f14aafd88661064",
    DOCUMENT: "b768a63da6809230e4fbd87bc7106e19460817aa3f5bb645d20645e00b18582a",
    SOURCE: "85d81e605afde5c94219ab752c9d52a4e7f8ed8a5b7a97ff1dc725cf6e1c5021",
    COMPARATOR_CLOCK: "bffdcf158d7d4e38db5794fb4761de528fb73b0b772ae950f3a087a93ab63f1a",
    COMPARATOR_MANIFEST: "a795f384287f24200e00d2cc5a5721610bb5282d1b044b3a653a053190c44261",
    ARTIFACT: "a70329eb292aff8a334e986450959661f633cc61deb232874c883c3d1b5982e0",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact() -> dict[str, Any]:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_cdltr72a_preregistration_dependencies_are_hash_frozen() -> None:
    for path, expected in EXPECTED_HASHES.items():
        assert _sha256(path) == expected


def test_cdltr72a_preregistration_is_canonical_and_self_consistent() -> None:
    artifact = _artifact()
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    assert artifact["candidate"] == "CDLTR-72A"
    assert artifact["protocol_version"] == prereg.PROTOCOL_VERSION
    assert artifact["manifest_hash"] == _canonical_hash(core)
    assert artifact["manifest_hash"] == (
        "7b094ece1c86ca84081476a1fb4b5035df1149a0eb1075aa5d657fd7eca799c7"
    )
    assert artifact["policy_hash"] == _canonical_hash(artifact["policy"])
    assert prereg.load_preregistration(ARTIFACT) == artifact


def test_cdltr72a_preregistration_binds_complete_capability_aware_comparators() -> None:
    binding = _artifact()["comparator_binding"]
    assert binding["clock"] == str(COMPARATOR_CLOCK)
    assert binding["clock_sha256"] == EXPECTED_HASHES[COMPARATOR_CLOCK]
    assert binding["manifest"] == str(COMPARATOR_MANIFEST)
    assert binding["manifest_sha256"] == EXPECTED_HASHES[COMPARATOR_MANIFEST]
    assert binding["rows"] == 9_985
    assert binding["directional_rows"] == 1_788
    assert binding["timestamp_only_rows"] == 8_197
    assert binding["directional_comparators"] == list(prereg.DIRECTIONAL_COMPARATORS)
    assert binding["timestamp_only_comparators"] == list(
        prereg.TIMESTAMP_ONLY_COMPARATORS
    )
    assert binding["event_rows_read_during_preregistration"] == 0
    assert binding["manifest_values_parsed_during_preregistration"] == 0


def test_cdltr72a_preregistration_keeps_every_outcome_boundary_closed() -> None:
    artifact = _artifact()
    assert artifact["outcomes_opened"] is False
    assert artifact["source_incidence_opened"] is False
    assert artifact["comparator_incidence_opened"] is False
    assert artifact["performance_values_opened"] is False
    assert artifact["outcome_boundary"] == prereg.EXPECTED_OUTCOME_BOUNDARY
    assert all(value == 0 for value in artifact["outcome_boundary"].values())


def test_cdltr72a_preregistration_freezes_amendment_controls_and_llm_boundary() -> None:
    artifact = _artifact()
    assert artifact["mechanism_decision"] == {
        "path": str(MECHANISM),
        "sha256": EXPECTED_HASHES[MECHANISM],
    }
    assert artifact["comparator_amendment"] == {
        "path": str(AMENDMENT),
        "sha256": EXPECTED_HASHES[AMENDMENT],
    }
    policy = artifact["policy"]
    assert (
        "CDLTR-72 rejected before preregistration" in policy["predecessor_disposition"]
    )
    assert "CDLTR-72|20260721|" in policy["controls"]["deterministic_random_side"]
    assert policy["novelty_gates"]["failure_action"] == (
        "reject CDLTR-72A without repair or outcomes"
    )
    assert policy["llm_boundary"] == {
        "authorized_before_deterministic_train_and_selection_pass": False,
        "later_role": "TRADE/ABSTAIN veto only",
        "may_change_side_timing_hold_or_relay": False,
        "rl_reward_requirements": ["strict drawdown penalty", "turnover penalty"],
    }


def test_cdltr72a_document_explicitly_freezes_no_repair_and_no_incidence() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    normalized = " ".join(text.replace("**", "").replace("`", "").split())
    required = (
        "CDLTR-72 was rejected before preregistration and before incidence",
        "It inherits the exact source votes, relay, execution, windows, support gates, controls, and LLM/RL boundary",
        "Preregistration hashes each file and reads at most the CSV header",
        "CDLTR-72|20260721|<entry_time_utc>",
        "The clock has exactly 9,985 rows",
        "Every FLCC candidate must pass independently",
        "The evaluator may not invent, infer, or search a missing side, exit, hold, union, or conflict resolver",
        "Preregistration reads only the bundle header",
        "Only an unchanged pass of source integrity, primary support, every control support check, and every novelty gate",
        "A failure permanently rejects CDLTR-72A",
    )
    for clause in required:
        assert clause in normalized
