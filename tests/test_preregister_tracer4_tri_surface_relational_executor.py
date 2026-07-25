from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import pytest
import pandas as pd

from training import preregister_tracer4_tri_surface_relational_executor as p


def _tokens() -> OrderedDict[str, str]:
    return OrderedDict(
        (name, values[0]) for name, values in p.TOKEN_SCHEMA
    )


def test_manifest_is_source_only_and_chronological() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)

    assert payload["policy"]["policy_id"] == "TRACER-4H"
    assert payload["stage_authority"]["authorized"] == [
        "source_cut",
        "primitive",
        "rank",
        "token_support",
    ]
    assert payload["stage_authority"]["support_pass_required_for_stage_0_5"] is True
    assert payload["temporal_roles"]["fit"][0].startswith("2020-")
    assert payload["temporal_roles"]["test"][0].startswith("2021-")
    assert payload["temporal_roles"]["eval"][0].startswith("2022-")
    assert payload["temporal_roles"]["confirmation"][0].startswith("2023-")
    assert all(value == 0 for value in payload["outcome_boundary"].values())


def test_sources_are_hash_header_allowlist_and_cut_bound() -> None:
    payload = p.build_manifest()
    sources = payload["sources"]

    assert sources["leadership"]["path"] == p.LEADERSHIP_SOURCE
    assert sources["leadership"]["sha256"] == p.LEADERSHIP_SOURCE_SHA256
    assert sources["leadership"]["physical_header"] == list(p.LEADERSHIP_PHYSICAL_HEADER)
    assert sources["leadership"]["cut_allowlist"] == list(p.LEADERSHIP_ALLOWLIST)
    assert sources["aggtrade"]["cut_allowlist"] == list(p.AGGTRADE_ALLOWLIST)
    assert sources["premium"]["cut_allowlist"] == list(p.PREMIUM_ALLOWLIST)
    assert all(contract["load_all_then_drop_forbidden"] for contract in sources.values())
    assert payload["physical_cuts"]["paths"] == p.PRE2024_CUTS
    assert payload["physical_cuts"]["gzip_mtime"] == 0


def test_state_times_freeze_premium_confirmation_before_decision() -> None:
    boundary = pd.Timestamp("2021-03-04T08:00:00Z")
    times = p.state_times(boundary)
    assert times["window_start"] == boundary - pd.Timedelta(hours=4)
    assert times["window_end"] == boundary
    assert times["premium_cutoff"] == boundary + pd.Timedelta(seconds=61)
    assert times["decision_time"] == boundary + pd.Timedelta(minutes=5)
    assert times["execution_time"] == boundary + pd.Timedelta(minutes=10)
    assert times["premium_cutoff"] < times["decision_time"]
    with pytest.raises(ValueError, match="canonical"):
        p.state_times("2021-03-04T09:00:00Z")


def test_strict_prior_band_excludes_current_caps_and_freezes_ties() -> None:
    history = list(range(600))
    assert p.strict_prior_band(-1.0, history, minimum=360, maximum=540) == "LOW"
    assert p.strict_prior_band(599.0, history, minimum=360, maximum=540) == "HIGH"
    assert p.strict_prior_band(300.0, history, minimum=360, maximum=540) == "MID"
    tied = [2.0] * 360
    assert p.strict_prior_band(2.0, tied, minimum=360, maximum=540) == "LOW"
    with pytest.raises(ValueError, match="not ready"):
        p.strict_prior_band(1.0, [0.0], minimum=2)
    with pytest.raises(ValueError, match="finite"):
        p.strict_prior_band(float("nan"), [0.0, 1.0], minimum=2)


def test_token_line_is_ordered_and_safety_is_not_a_learned_token() -> None:
    tokens = _tokens()
    line = p.canonical_line(tokens)
    assert line.startswith("sponsor=CASH_LEADS")
    assert len(line.split(" | ")) == len(p.TOKEN_COLUMNS)
    reversed_tokens = OrderedDict(reversed(tuple(tokens.items())))
    with pytest.raises(ValueError, match="order or schema"):
        p.validate_tokens(reversed_tokens)
    invalid = _tokens()
    invalid["sponsor"] = "SOURCE_INVALID"
    with pytest.raises(ValueError, match="invalid"):
        p.validate_tokens(invalid)
    assert p.safety_line().startswith("SOURCE_INVALID|")


def test_jsd_uses_complete_vocabulary_and_zero_mass_convention() -> None:
    vocab = ("A", "B", "C")
    assert p.jensen_shannon_divergence({"A": 1}, {"A": 2}, vocab) == 0.0
    assert p.jensen_shannon_divergence({"A": 1}, {"B": 1}, vocab) == 1.0
    mixed = p.jensen_shannon_divergence({"A": 1, "C": 1}, {"B": 1, "C": 1}, vocab)
    assert 0.0 < mixed < 1.0
    with pytest.raises(ValueError, match="positive mass"):
        p.jensen_shannon_divergence({}, {"A": 1}, vocab)


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


def test_boundary_hash_and_latest_commit_are_current() -> None:
    assert p.sha256_file(p.BOUNDARY_DOCUMENT) == p.BOUNDARY_DOCUMENT_SHA256
    p.assert_boundary_committed()
