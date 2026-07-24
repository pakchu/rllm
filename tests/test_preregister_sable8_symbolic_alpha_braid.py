from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import pytest

from training import preregister_sable8_symbolic_alpha_braid as p


def _tokens(*, context: str = "MIDDLE") -> OrderedDict[str, str]:
    return OrderedDict(
        (
            primitive,
            "MIDDLE" if primitive in p.CORE_PRIMITIVES else context,
        )
        for primitive in p.PRIMITIVES
    )


def test_manifest_is_source_only_and_reserves_2023_as_gate() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)

    assert payload["policy"]["policy_id"] == "SABLE-8"
    assert payload["research_history_boundary"] == {
        "component_family_outcomes_seen": True,
        "sable_source_values_seen": False,
        "sable_token_incidence_seen": False,
        "sable_rewards_seen": False,
        "sable_model_outcomes_seen": False,
        "global_pristine_holdout_claimed": False,
        "claim_scope": "contaminated-history candidate research MDP",
    }
    assert payload["temporal_roles"]["candidate_gate"] == [
        "2023-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ]
    assert payload["temporal_roles"]["candidate_gate_may_select"] is False
    assert payload["stage_authority"]["authorized"] == [
        "source_cut",
        "primitive",
        "rank",
        "token_support",
    ]
    assert all(
        value in (0, False)
        for value in payload["outcome_boundary"].values()
    )


def test_sources_are_bound_to_hash_allowlist_and_physical_cut() -> None:
    payload = p.build_manifest()
    sources = payload["sources"]

    assert sources["market"]["path"] == p.MARKET_SOURCE
    assert sources["market"]["sha256"] == p.MARKET_SOURCE_SHA256
    assert sources["market"]["physical_header"] == list(
        p.MARKET_PHYSICAL_HEADER
    )
    assert sources["market"]["physical_header_sha256"] == (
        p.MARKET_HEADER_SHA256
    )
    assert sources["market"]["cut_allowlist"] == list(p.MARKET_ALLOWLIST)
    assert sources["funding"]["cut_allowlist"] == list(p.FUNDING_ALLOWLIST)
    assert sources["premium"]["cut_allowlist"] == list(p.PREMIUM_ALLOWLIST)
    assert sources["funding"]["physical_header"] == list(
        p.FUNDING_PHYSICAL_HEADER
    )
    assert sources["premium"]["physical_header"] == list(
        p.PREMIUM_PHYSICAL_HEADER
    )
    assert all(
        contract["physical_stop_before_other_field_conversion"] is True
        for contract in sources.values()
    )
    assert payload["physical_cuts"]["paths"] == p.PRE2024_CUTS
    assert payload["physical_cuts"]["gzip_mtime"] == 0
    assert payload["physical_cuts"]["all_support_reads_cut_only"] is True


def test_strict_prior_midrank_excludes_current_handles_ties_and_caps() -> None:
    assert p.strict_prior_midrank(
        2.0,
        [1.0, 2.0, 3.0],
        minimum=3,
    ) == 0.5
    assert p.strict_prior_midrank(
        2.0,
        [2.0, 2.0],
        minimum=2,
    ) == 0.5
    assert p.strict_prior_midrank(
        5.0,
        [100.0, 0.0, 0.0],
        minimum=2,
        maximum=2,
    ) == 1.0
    with pytest.raises(ValueError, match="not ready"):
        p.strict_prior_midrank(1.0, [0.0], minimum=2)
    with pytest.raises(ValueError, match="finite"):
        p.strict_prior_midrank(float("nan"), [0.0], minimum=1)


@pytest.mark.parametrize(
    ("rank", "expected"),
    (
        (0.0, "EXTREME_LOW"),
        (0.2 - 1e-12, "EXTREME_LOW"),
        (0.2, "LOW"),
        (0.4 - 1e-12, "LOW"),
        (0.4, "MIDDLE"),
        (0.6, "MIDDLE"),
        (0.6 + 1e-12, "HIGH"),
        (0.8, "HIGH"),
        (0.8 + 1e-12, "EXTREME_HIGH"),
        (1.0, "EXTREME_HIGH"),
    ),
)
def test_rank_band_exact_boundaries(rank: float, expected: str) -> None:
    assert p.rank_band(rank) == expected


def test_token_line_is_ordered_and_only_context_can_be_stale() -> None:
    tokens = _tokens(context="STALE")
    line = p.canonical_line(tokens)
    assert line.startswith("PRICE_RETURN_1D=MIDDLE")
    assert line.endswith("DXY_CHANGE_1D=STALE")
    assert len(line.split(" | ")) == len(p.PRIMITIVES)

    reversed_tokens = OrderedDict(reversed(tuple(tokens.items())))
    with pytest.raises(ValueError, match="order or schema"):
        p.validate_token_line(reversed_tokens)

    invalid = _tokens()
    invalid[p.CORE_PRIMITIVES[0]] = "STALE"
    with pytest.raises(ValueError, match="core token"):
        p.validate_token_line(invalid)


def test_sequence_signature_requires_six_consecutive_oldest_first_lines() -> None:
    base = 1_600_000_000
    boundaries = [base + index * 28_800 for index in range(6)]
    lines = [f"line-{index}" for index in range(6)]
    first = p.sequence_signature(boundaries, lines)
    assert first == p.sequence_signature(boundaries, lines)
    assert first != p.sequence_signature(boundaries, list(reversed(lines)))

    broken = list(boundaries)
    broken[3] += 300
    with pytest.raises(ValueError, match="not consecutive"):
        p.sequence_signature(broken, lines)
    with pytest.raises(ValueError, match="exactly six"):
        p.sequence_signature(boundaries[:-1], lines[:-1])


def test_write_once_is_deterministic_and_rejects_drift(tmp_path: Path) -> None:
    output = tmp_path / "prereg.json"
    payload = p.build_manifest()
    first = p.write_once(output, payload)
    second = p.write_once(output, payload)
    assert first == second
    assert json.loads(output.read_text()) == payload

    changed = dict(payload)
    changed["protocol_version"] = "drift"
    with pytest.raises(RuntimeError, match="write-once"):
        p.write_once(output, changed)


def test_boundary_hash_is_current() -> None:
    assert p.sha256_file(p.BOUNDARY_DOCUMENT) == p.BOUNDARY_DOCUMENT_SHA256
