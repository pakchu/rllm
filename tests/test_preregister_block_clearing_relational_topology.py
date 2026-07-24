from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import gzip
import itertools
import json
from pathlib import Path

import pytest

from training import preregister_block_clearing_relational_topology as p


def _ranks(values: tuple[float, ...]) -> OrderedDict[str, float]:
    return OrderedDict(zip(p.PRIMITIVES, values, strict=True))


def _valid_tokens() -> OrderedDict[str, str]:
    return OrderedDict(
        (
            ("cadence_utilization", "CADENCE_LEADS"),
            ("utilization_fee", "BALANCED"),
            ("packing_witness", "WITNESS_LEADS"),
            ("utxo_fee", "UTXO_LEADS"),
            ("load_fee_dispersion", "FEE_WIDER"),
            ("high_leader", "CADENCE"),
            ("low_leader", "UTXO"),
            ("rank_breadth", "HIGH_BROAD"),
            ("extreme_occupancy", "FOCUSED"),
            ("relation_breadth", "LEFT_BROAD"),
            ("order_transition", "ROTATING"),
            ("leader_transition", "HIGH_ROTATED"),
        )
    )


def _ranks_with_inversions(inversions: int) -> OrderedDict[str, float]:
    for permutation in itertools.permutations(range(len(p.PRIMITIVES))):
        count = sum(
            permutation[left] > permutation[right]
            for left in range(len(permutation))
            for right in range(left + 1, len(permutation))
        )
        if count == inversions:
            return _ranks(tuple((value + 1) / 9.0 for value in permutation))
    raise AssertionError(f"no permutation with {inversions} inversions")


def test_manifest_is_outcome_blind_and_2023_source_is_report_only() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "BCRT-72"
    assert payload["research_history_boundary"] == {
        "base_chain_source_values_seen": True,
        "base_chain_family_outcomes_seen": True,
        "bate_2021_2022_outcomes_seen": True,
        "bcrt_values_seen": False,
        "bcrt_tokens_or_incidence_seen": False,
        "bcrt_market_outcomes_seen": False,
        "global_pristine_holdout_claimed": False,
        "claim_scope": "candidate-level frozen 2023 outcome window",
    }
    assert payload["temporal_roles"]["selection"] == [
        "2022-01-01T00:00:00Z",
        "2023-01-01T00:00:00Z",
    ]
    assert payload["temporal_roles"]["untouched_eval"] == [
        "2023-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ]
    assert payload["temporal_roles"]["adaptation"] is None
    report = payload["eval_source_report_only"]
    assert report["boolean_gate"] is False
    assert report["may_authorize_continue_retire_repair_or_selection"] is False
    assert report["unseen_token_value"] == "frozen policy ABSTAIN"
    assert payload["source_support_gates"]["boolean_scope"] == (
        "[2020-01-01,2023-01-01) development only"
    )
    assert all(
        value == 0
        for key, value in payload["outcome_boundary"].items()
        if key
        not in {
            "source_artifact_bytes_hashed",
            "source_manifest_aggregate_metadata_read",
            "source_header_read",
        }
    )


def test_source_anchor_prefix_closure_and_weight_contracts_are_frozen() -> None:
    payload = p.build_manifest()
    source = payload["source_contract"]
    assert source["allowlist"] == list(p.SOURCE_ALLOWLIST)
    assert source["loader"] == (
        "pandas.read_csv(usecols=allowlist); no load-and-drop"
    )
    assert source["validation"]["weight_bounds"] == (
        "size <= weight <= 4*size"
    )
    assert source["validation"]["timestamps_before_cutoff"] is True
    assert source["production_contract"] == {
        "owned_bitcoin_core_required": True,
        "actual_first_seen_timestamps_required": True,
        "raw_response_hashing_required": True,
        "forward_field_parity_required": True,
        "live_mismatch_action": (
            "block BCRT; do not rewrite historical values"
        ),
    }

    bucket = payload["bucket_contract"]
    assert bucket["anchor_height"].startswith("first height whose mediantime")
    assert bucket["confirmation_height"] == "anchor_height+288"
    assert "height<=confirmation_height" in bucket["membership"]
    assert bucket["late_backdated_members"].startswith("ignore")
    assert "byte-identical" in bucket["prefix_replay"]
    assert bucket[
        "previous_state_includes_reservation_or_split_suppressed"
    ] is True
    assert bucket["latest_unconfirmed_buckets"] == "omit"

    execution = payload["execution_contract"]
    assert execution["raw_available"] == (
        "max(bucket_end,prefix_max_timestamp_through_confirmation,"
        "prefix_max_mediantime_through_confirmation)+172800"
    )
    assert execution["global_action_independent_reservation"] is True
    assert execution["abstention_releases_reservation"] is False
    assert execution["split_containment_fields"] == [
        "source_bucket",
        "anchor",
        "confirmation",
        "signal_available",
        "latency_bar",
        "entry",
        "held_bars",
        "exit",
    ]
    assert execution["split_interval"] == "half-open"
    assert execution["live_clock"] == (
        "max(frozen historical clock, actual node receipt/validation clock)"
    )


def test_development_support_gates_do_not_use_eval_incidence() -> None:
    gates = p.build_manifest()["source_support_gates"]
    incidence = gates["development_incidence"]
    assert incidence == {
        "development_2020_2022_min": 2000,
        "train_2020_2021_min": 1250,
        "year_2020_min": 570,
        "each_year_2021_2022_min": 700,
        "year_2020_active_months_min": 9,
        "each_year_2021_2022_active_months_min": 12,
        "each_half_2021_2022_min": 340,
        "each_quarter_2021_2022_min": 165,
        "year_2020_max_month_share": 0.13,
        "each_year_2021_2022_max_month_share": 0.10,
        "max_entry_gap_days_2020_2022": 3,
    }
    assert gates["token_scope"] == [
        "train_2020_2021",
        "selection_2022",
    ]
    assert gates["token_support"]["selection_values_must_exist_in_train"] is (
        True
    )
    assert "2023" not in json.dumps(
        {
            "development_incidence": incidence,
            "token_scope": gates["token_scope"],
            "token_support": gates["token_support"],
        },
        sort_keys=True,
    )


def test_strict_prior_midrank_excludes_current_handles_ties_and_truncates() -> None:
    assert p.strict_prior_midrank(2.0, [1.0, 2.0, 3.0], minimum=3) == 0.5
    assert p.strict_prior_midrank(2.0, [2.0, 2.0], minimum=2) == 0.5
    assert p.strict_prior_midrank(
        5.0,
        [100.0, 0.0, 0.0],
        minimum=2,
        maximum=2,
    ) == 1.0
    with pytest.raises(ValueError, match="history is not ready"):
        p.strict_prior_midrank(1.0, [0.0], minimum=2)
    with pytest.raises(ValueError, match="finite"):
        p.strict_prior_midrank(float("nan"), [0.0], minimum=1)


def test_pair_relations_use_strict_one_sixth_boundaries() -> None:
    assert p.pair_relation(
        1.0 / 6.0,
        0.0,
        left_token="LEFT",
        right_token="RIGHT",
    ) == "BALANCED"
    assert p.pair_relation(
        1.0 / 6.0 + 1e-12,
        0.0,
        left_token="LEFT",
        right_token="RIGHT",
    ) == "LEFT"
    assert p.pair_relation(
        0.0,
        1.0 / 6.0,
        left_token="LEFT",
        right_token="RIGHT",
    ) == "BALANCED"
    assert p.pair_relation(
        0.0,
        1.0 / 6.0 + 1e-12,
        left_token="LEFT",
        right_token="RIGHT",
    ) == "RIGHT"


def test_leaders_breadths_and_extreme_occupancy_are_exact() -> None:
    ascending = _ranks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
    assert p.extreme_leader(ascending, highest=True) == "FEE_DISPERSION"
    assert p.extreme_leader(ascending, highest=False) == "CADENCE"
    tied = _ranks((0.8, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
    assert p.extreme_leader(tied, highest=True) == "TIE"

    assert p.rank_breadth(
        _ranks((0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4))
    ) == "HIGH_BROAD"
    assert p.rank_breadth(
        _ranks((0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4))
    ) == "LOW_BROAD"
    assert p.rank_breadth(
        _ranks((0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4))
    ) == "MIXED"

    assert p.extreme_occupancy(
        _ranks((0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0 / 6.0, 5.0 / 6.0))
    ) == "COMPACT"
    assert p.extreme_occupancy(
        _ranks((0.0, 1.0, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5))
    ) == "FOCUSED"
    assert p.extreme_occupancy(
        _ranks((0.0, 1.0, 0.1, 0.9, 0.05, 0.95, 0.5, 0.5))
    ) == "FRACTURED"


@pytest.mark.parametrize(
    ("scores", "expected"),
    (
        ((1, 1, 0, 0, 0), "LEFT_BROAD"),
        ((-1, -1, 0, 0, 0), "RIGHT_BROAD"),
        ((1, -1, 1, -1, 0), "MIXED"),
    ),
)
def test_relation_breadth(scores: tuple[int, ...], expected: str) -> None:
    assert p.relation_breadth(scores) == expected


@pytest.mark.parametrize(
    ("inversions", "expected"),
    (
        (0, "STABLE"),
        (6, "STABLE"),
        (7, "ROTATING"),
        (13, "ROTATING"),
        (14, "RESET"),
        (28, "RESET"),
    ),
)
def test_order_transition_exact_change_bands(
    inversions: int,
    expected: str,
) -> None:
    previous = _ranks(tuple((index + 1) / 9.0 for index in range(8)))
    current = _ranks_with_inversions(inversions)
    assert p.order_transition(current, previous) == expected


@pytest.mark.parametrize(
    ("leaders", "expected"),
    (
        (("TIE", "UTXO", "CADENCE", "UTXO"), "TIE_INVOLVED"),
        (("CADENCE", "UTXO", "CADENCE", "UTXO"), "BOTH_STABLE"),
        (("FEE", "UTXO", "CADENCE", "UTXO"), "HIGH_ROTATED"),
        (("CADENCE", "FEE", "CADENCE", "UTXO"), "LOW_ROTATED"),
        (("FEE", "PACKING", "CADENCE", "UTXO"), "BOTH_ROTATED"),
    ),
)
def test_leader_transition(
    leaders: tuple[str, str, str, str],
    expected: str,
) -> None:
    assert p.leader_transition(*leaders) == expected


def test_token_order_vocabulary_and_neutral_code_orders_fail_closed() -> None:
    assert p.validate_tokens(_valid_tokens()) == dict(_valid_tokens())
    reversed_tokens = OrderedDict(reversed(tuple(_valid_tokens().items())))
    with pytest.raises(ValueError, match="order or schema"):
        p.validate_tokens(reversed_tokens)
    bad = _valid_tokens()
    bad["order_transition"] = "UNKNOWN"
    with pytest.raises(ValueError, match="token value is invalid"):
        p.validate_tokens(bad)
    assert p.neutral_code_orders() == (("Q1", "Q2"), ("Q2", "Q1"))


def test_causal_clock_uses_max_then_embargo_ceiling_latency_and_six_hour_hold() -> None:
    exact = p.opportunity_times(
        bucket_end_seconds=172_800,
        prefix_max_timestamp=172_500,
        prefix_max_mediantime=172_800,
    )
    assert exact["signal_available"] == datetime.fromtimestamp(
        345_600,
        tz=timezone.utc,
    )
    assert exact["entry"] - exact["signal_available"] == timedelta(minutes=5)
    assert exact["exit"] - exact["entry"] == timedelta(hours=6)

    later_prefix = p.opportunity_times(
        bucket_end_seconds=172_800,
        prefix_max_timestamp=172_801,
        prefix_max_mediantime=172_799,
    )
    raw = 172_801 + 172_800
    assert later_prefix["signal_available"] == datetime.fromtimestamp(
        p.ceil_5m(raw),
        tz=timezone.utc,
    )
    assert later_prefix["signal_available"] > exact["signal_available"]


def test_global_reservation_is_action_independent_and_half_open() -> None:
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "id": "overlap",
            "entry": start + timedelta(hours=5),
            "exit": start + timedelta(hours=11),
            "action": "LONG",
        },
        {
            "id": "first",
            "entry": start,
            "exit": start + timedelta(hours=6),
            "action": "ABSTAIN",
        },
        {
            "id": "touching",
            "entry": start + timedelta(hours=6),
            "exit": start + timedelta(hours=12),
            "action": "SHORT",
        },
    ]
    first = p.reserve_intervals(rows)
    second = p.reserve_intervals(
        [
            {
                **row,
                "action": "SHORT" if row["action"] != "SHORT" else "LONG",
            }
            for row in rows
        ]
    )
    assert [(row["id"], row["reserved"]) for row in first] == [
        ("first", True),
        ("overlap", False),
        ("touching", True),
    ]
    assert [row["reserved"] for row in first] == [
        row["reserved"] for row in second
    ]


def test_familywise_prior_ablation_and_masked_adapter_controls_are_frozen() -> None:
    payload = p.build_manifest()
    baseline = payload["baseline_contract"]
    assert baseline["circular_block_shift_offsets"] == [
        62,
        93,
        124,
        155,
        186,
        217,
        248,
        279,
        310,
        341,
        372,
        403,
        434,
        465,
        496,
        527,
    ]
    assert len(baseline["token_groups"]) == 5
    assert baseline["shuffle_seeds"] == list(
        range(20_260_724, 20_260_756)
    )
    assert baseline["selection_2022"][
        "single_token_action_reproduction_max"
    ] == 0.70
    orientation = baseline["orientation_flipped_inference_controls"]
    assert orientation["refit"] is False
    assert orientation["can_qualify"] is False
    assert "non-BALANCED" in orientation["pair_direction_flip"]
    assert "LEFT_BROAD" in orientation["relation_breadth_flip"]
    for split in ("transfer_2021", "selection_2022"):
        gate = baseline[split]
        assert gate["beat_always_abstain_return"] is True
        assert gate[
            "beat_always_long_short_exact_memory_return_and_ratio"
        ] is True
        assert gate["beat_both_orientation_flipped_controls"] is True

    correction = payload["familywise_signflip_contract"]
    assert correction["draws"] == 100_000
    assert correction["seed"] == 20_260_724
    assert "same Rademacher sign" in correction["shared_null"]
    assert correction["post_result_omission"] == "forbidden"
    assert correction["transfer_p_max_below"] == 0.25
    assert correction["cheap_selection_p_max_below"] == 0.10
    assert correction["rllm_selection_p_max_below"] == 0.05
    assert "pair_direction_flipped_inference" in correction["family"]
    assert "relation_breadth_flipped_inference" in correction["family"]

    rllm = payload["rllm_contract"]
    masked = rllm["masked_token_prior_adapter"]
    assert masked["input"].endswith("literal MASKED")
    assert masked["can_qualify"] is False
    prompt = rllm["prompt_serialization"]
    assert prompt["state"].startswith("KEY=VALUE")
    assert prompt["completion_candidates"] == ["CHOICE=Q1", "CHOICE=Q2"]
    assert prompt["text_only"] is True
    assert prompt["position_in_prompt"] is False
    selection = rllm["selection_2022"]
    assert selection["beat_every_frozen_non_rllm_prior_ablation_null"] is True
    assert selection["beat_both_orientation_flipped_controls"] is True
    assert selection["beat_masked_token_prior_adapter"] is True
    assert selection["ratio_margin_over_strongest_non_rllm"] == 0.25
    assert selection["max_token_value_non_abstain_action_share"] == 0.60

    evaluation = payload["eval_2023_gate"]
    assert evaluation["one_policy_weekly_cluster_p_below"] == 0.05
    assert evaluation["beat_both_orientation_flipped_controls"] is True
    assert evaluation["ratio_margin_over_strongest_non_rllm"] == 0.50
    assert evaluation["single_token_action_reproduction_max"] == 0.70


def test_frozen_hashes_headers_comparators_and_sealed_fetd_report() -> None:
    p.validate_frozen_dependencies()
    assert p.sha256_csv_header(p.SOURCE) == p.SOURCE_HEADER_SHA256
    assert p.csv_header(p.SOURCE) == list(p.SOURCE_ALLOWLIST)
    assert p.sha256_csv_header(p.REFERENCE) == p.REFERENCE_HEADER_SHA256
    assert p.csv_header(p.REFERENCE) == list(p.REFERENCE_ALLOWLIST)
    comparators = p.comparator_contracts()
    assert [item["id"] for item in comparators] == [
        "BATE-288",
        "UFCP-1",
        "MCR-7",
        "NTB-7",
        "BFC-3",
        "WCTR-288",
        "BFRT-288",
        "EMFC-864",
        "CCHR-live-pre2024",
    ]
    for comparator in comparators:
        assert comparator["loader_allowlist"] == [
            comparator["group_column"],
            comparator["entry_column"],
            comparator["exit_column"],
            comparator["side_column"],
        ]
        assert p.sha256_csv_header(comparator["path"]) == comparator[
            "header_sha256"
        ]
    dependencies = p.frozen_dependencies()
    assert dependencies[
        "results/fee_endpoint_topology_disagreement_support_2026-07-20.json"
    ] == "03ba910a314ba6efb647f6588dff603261d414e5114680ca33bdc27d59aed035"


def test_canonical_preregistration_artifact_matches_code() -> None:
    artifact = (
        "results/block_clearing_relational_topology_"
        "preregistration_2026-07-24.json"
    )
    stored = json.loads(Path(artifact).read_text(encoding="utf-8"))
    p.validate_manifest(stored)
    assert stored["manifest_hash"] == (
        "c9f08196f5a25dd05320a2c7cf3fbf951403d10f2362e67e2b0169b03fec194f"
    )
    assert p.sha256_file(artifact) == (
        "322f91b41fce1aee06250a010d5a569557b83cc3f493ee3c47f5d6974aafe6a8"
    )


def test_header_reader_does_not_decode_rows(tmp_path) -> None:
    plain = tmp_path / "panel.csv"
    plain.write_bytes(b"a,b\n\xff\xfe\x00not-csv")
    assert p.csv_header(plain) == ["a", "b"]

    compressed = tmp_path / "panel.csv.gz"
    with gzip.open(compressed, "wb") as handle:
        handle.write(b"x,y\n\xff\xfe\x00not-csv")
    assert p.csv_header(compressed) == ["x", "y"]


def test_write_once_is_reproducible_and_rejects_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        p,
        "validate_frozen_dependencies",
        lambda: calls.append(True),
    )
    output = tmp_path / "freeze.json"
    payload = p.build_manifest()
    assert p.write_once(output, payload) == "created"
    assert calls == [True]
    assert output.read_text(encoding="utf-8") == p._canonical_manifest_text()
    assert p.write_once(output, p.build_manifest()) == "verified_existing"
    assert calls == [True, True]

    stored = json.loads(output.read_text(encoding="utf-8"))
    stored["policy"]["hold_bars"] = 71
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(stored)


def test_validate_manifest_rejects_self_rehashed_drift() -> None:
    payload = p.build_manifest()
    payload["eval_2023_gate"]["ratio_min"] = 2.99
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(payload)
