from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import copy
import json

import pytest

import training.preregister_tri_party_composition_topology as tpct

UTC = timezone.utc


def ranks(values: list[Fraction]) -> OrderedDict[str, Fraction]:
    assert len(values) == len(tpct.PRIMITIVES)
    return OrderedDict(zip(tpct.PRIMITIVES, values, strict=True))


def test_policy_math_constants_are_internally_consistent() -> None:
    policy = tpct.Policy()

    assert policy.bar_seconds == 300
    assert policy.latency_bars == 1
    assert policy.hold_hours == 120
    assert policy.hold_bars == 1_440
    assert policy.hold_bars * policy.bar_seconds == policy.hold_hours * 60 * 60
    assert Fraction(policy.pair_delta_numerator, policy.pair_delta_denominator) == Fraction(1, 6)
    assert Fraction(policy.extreme_low_numerator, policy.extreme_low_denominator) == Fraction(1, 6)
    assert Fraction(policy.extreme_high_numerator, policy.extreme_high_denominator) == Fraction(5, 6)
    assert policy.random_seed == 20_260_724


def test_parse_exact_decimal_accepts_only_canonical_fraction_text() -> None:
    assert tpct.parse_exact_decimal("0") == Fraction(0)
    assert tpct.parse_exact_decimal("123.4500") == Fraction(2469, 20)
    assert tpct.parse_exact_decimal("-3.125") == Fraction(-25, 8)

    for forbidden in ["+1", "01", "1.", ".5", "1e-3", "1,000", "NaN", "Infinity", "-0", "-0.00"]:
        with pytest.raises(ValueError, match="canonical exact decimal|negative zero"):
            tpct.parse_exact_decimal(forbidden)


def test_build_primitives_uses_exact_rational_arithmetic_and_simplex_identity() -> None:
    values = OrderedDict(
        (
            ("AR_OO", Fraction(1, 100)),
            ("TV_OO", Fraction(10)),
            ("AR_B27", Fraction(2, 100)),
            ("TV_B27", Fraction(20)),
            ("AR_B830", Fraction(3, 100)),
            ("TV_B830", Fraction(30)),
            ("AR_G30", Fraction(4, 100)),
            ("TV_G30", Fraction(40)),
            ("AR_T", Fraction(1, 100)),
            ("TV_T", Fraction(50)),
            ("AR_AG", Fraction(3, 100)),
            ("TV_AG", Fraction(70)),
            ("AR_CORD", Fraction(5, 100)),
            ("TV_CORD", Fraction(110)),
            ("AR_O", Fraction(7, 100)),
            ("TV_O", Fraction(130)),
        )
    )

    primitives = tpct.build_primitives(values)

    assert tuple(primitives) == tpct.PRIMITIVES
    assert primitives["OVERNIGHT"] == Fraction(1, 10)
    assert primitives["NEAR_TERM"] == Fraction(1, 5)
    assert primitives["MEDIUM_TERM"] == Fraction(3, 10)
    assert primitives["LONG_TERM"] == Fraction(2, 5)
    assert sum(primitives[key] for key in ("OVERNIGHT", "NEAR_TERM", "MEDIUM_TERM", "LONG_TERM")) == 1
    assert primitives["TERM_PREMIUM"] == Fraction(1, 45)
    assert primitives["GOVERNMENT_SHARE"] == Fraction(1, 3)
    assert primitives["TREASURY_WITHIN_GOV"] == Fraction(5, 12)
    assert primitives["CORPORATE_WITHIN_PRIVATE"] == Fraction(11, 24)
    assert primitives["PRIVATE_PREMIUM"] == (Fraction(5, 100) * 110 + Fraction(7, 100) * 130) / 240 - (Fraction(1, 100) * 50 + Fraction(3, 100) * 70) / 120
    assert primitives["CONCENTRATION_GAP"] == (Fraction(1, 100) + Fraction(4, 100) + Fraction(9, 100) + Fraction(16, 100)) - (Fraction(25, 1296) + Fraction(49, 1296) + Fraction(121, 1296) + Fraction(169, 1296))


def test_build_primitives_rejects_float_inputs_and_nonpositive_transaction_volume() -> None:
    values = OrderedDict((key, Fraction(1)) for key in tpct.VALUE_KEYS)
    values["AR_OO"] = 1.0
    with pytest.raises(TypeError, match="exact Fractions"):
        tpct.build_primitives(values)

    values = OrderedDict((key, Fraction(1)) for key in tpct.VALUE_KEYS)
    values["TV_OO"] = Fraction(0)
    with pytest.raises(ValueError, match="volume must be positive"):
        tpct.build_primitives(values)


def test_strict_prior_midrank_requires_exact_252_fraction_history_and_handles_ties() -> None:
    prior = [Fraction(i) for i in range(100)] + [Fraction(100)] * 52 + [Fraction(i) for i in range(101, 201)]
    assert len(prior) == 252

    assert tpct.strict_prior_midrank(Fraction(100), prior) == Fraction(100 * 2 + 52, 2 * 252)

    with pytest.raises(ValueError, match="exactly 252"):
        tpct.strict_prior_midrank(Fraction(1), prior[:-1])
    with pytest.raises(TypeError, match="must be Fractions"):
        tpct.strict_prior_midrank(Fraction(1), [Fraction(0)] * 251 + [0.0])


def test_pair_relation_boundaries_are_exact_fractions() -> None:
    assert tpct.pair_relation(Fraction(2, 3), Fraction(1, 2), left_token="LEFT", right_token="RIGHT") == "BALANCED"
    assert tpct.pair_relation(Fraction(1, 2), Fraction(2, 3), left_token="LEFT", right_token="RIGHT") == "BALANCED"
    assert tpct.pair_relation(Fraction(2, 3) + Fraction(1, 252), Fraction(1, 2), left_token="LEFT", right_token="RIGHT") == "LEFT"
    assert tpct.pair_relation(Fraction(1, 2), Fraction(2, 3) + Fraction(1, 252), left_token="LEFT", right_token="RIGHT") == "RIGHT"


def test_leader_breadth_and_occupancy_boundaries_are_strict_and_tie_aware() -> None:
    tied_extremes = ranks([Fraction(1, 2)] * len(tpct.PRIMITIVES))
    assert tpct.extreme_leader(tied_extremes, highest=True) == "TIE"
    assert tpct.extreme_leader(tied_extremes, highest=False) == "TIE"
    assert tpct.rank_breadth(tied_extremes) == "MIXED"
    assert tpct.extreme_occupancy(tied_extremes) == "COMPACT"

    high_broad = ranks([Fraction(2, 3)] * 3 + [Fraction(1, 2)] * 7)
    low_broad = ranks([Fraction(1, 3)] * 3 + [Fraction(1, 2)] * 7)
    assert tpct.rank_breadth(high_broad) == "HIGH_BROAD"
    assert tpct.rank_breadth(low_broad) == "LOW_BROAD"

    exact_boundaries = ranks([Fraction(1, 6), Fraction(5, 6)] * 5)
    assert tpct.extreme_occupancy(exact_boundaries) == "COMPACT"
    four_extremes = ranks([Fraction(1, 6) - Fraction(1, 252)] * 4 + [Fraction(1, 2)] * 6)
    seven_extremes = ranks([Fraction(5, 6) + Fraction(1, 252)] * 7 + [Fraction(1, 2)] * 3)
    assert tpct.extreme_occupancy(four_extremes) == "FOCUSED"
    assert tpct.extreme_occupancy(seven_extremes) == "FRACTURED"


def test_order_transition_counts_all_45_unordered_pairs_and_thresholds() -> None:
    previous = ranks([Fraction(i, 20) for i in range(10)])
    stable_current = ranks([Fraction(i, 20) for i in range(10)])
    rotating_current = ranks([Fraction(3, 20), Fraction(4, 20), Fraction(5, 20), Fraction(0), Fraction(1, 20), Fraction(2, 20), Fraction(6, 20), Fraction(7, 20), Fraction(8, 20), Fraction(9, 20)])
    reset_current = ranks(list(reversed([Fraction(i, 20) for i in range(10)])))

    assert len(tpct._pair_order_states(previous)) == 45
    assert tpct.order_transition(stable_current, previous) == "STABLE"
    assert tpct.order_transition(rotating_current, previous) == "ROTATING"
    assert tpct.order_transition(reset_current, previous) == "RESET"


def test_build_tokens_emits_canonical_twelve_token_state() -> None:
    current = ranks([
        Fraction(9, 10), Fraction(7, 10), Fraction(2, 10), Fraction(6, 10), Fraction(1, 10),
        Fraction(8, 10), Fraction(75, 100), Fraction(1, 4), Fraction(55, 100), Fraction(65, 100),
    ])
    previous = ranks([Fraction(i, 20) for i in range(10)])

    tokens = tpct.build_tokens(current, previous)

    assert tuple(tokens) == tpct.TOKEN_COLUMNS
    assert tokens == OrderedDict(
        (
            ("maturity_wings", "OVERNIGHT_LEADS"),
            ("term_belly", "NEAR_TERM_LEADS"),
            ("term_volume_rate", "LONG_TERM_VOLUME_LEADS"),
            ("collateral_volume_rate", "GOVERNMENT_VOLUME_LEADS"),
            ("safe_risky_composition", "TREASURY_LEADS"),
            ("rate_surface", "PRIVATE_RATE_LEADS"),
            ("high_leader", "OVERNIGHT"),
            ("low_leader", "TERM_PREMIUM"),
            ("rank_breadth", "HIGH_BROAD"),
            ("extreme_occupancy", "COMPACT"),
            ("order_transition", "RESET"),
            ("leader_transition", "BOTH_ROTATED"),
        )
    )


def test_clock_decode_reservation_and_split_boundaries_are_exact() -> None:
    exact_bar = datetime(2022, 12, 26, 23, 50, tzinfo=UTC)
    subsecond = datetime(2022, 12, 26, 23, 50, 0, 1, tzinfo=UTC)
    assert tpct.ceil_5m(exact_bar) == exact_bar
    assert tpct.ceil_5m(subsecond) == datetime(2022, 12, 26, 23, 55, tzinfo=UTC)

    assert tpct.source_value_may_decode(datetime(2020, 9, 9, 23, 55, tzinfo=UTC)) is True
    assert tpct.source_value_may_decode(datetime(2020, 9, 9, 23, 50, tzinfo=UTC)) is False
    assert tpct.source_value_may_decode(datetime(2022, 12, 26, 23, 50, tzinfo=UTC)) is True
    assert tpct.source_value_may_decode(datetime(2022, 12, 26, 23, 55, tzinfo=UTC)) is False

    start = datetime(2022, 1, 1, tzinfo=UTC)
    entry = start
    exit_time = start + timedelta(hours=120)
    assert tpct.split_contains(entry, exit_time, start, exit_time + timedelta(seconds=1)) is True
    assert tpct.split_contains(entry, exit_time, start, exit_time) is False

    rows = [
        {"id": "first", "entry": start, "exit": start + timedelta(hours=120)},
        {"id": "overlap", "entry": start + timedelta(hours=1), "exit": start + timedelta(hours=121)},
        {"id": "touching", "entry": start + timedelta(hours=120), "exit": start + timedelta(hours=240)},
    ]
    reserved = tpct.reserve_intervals(rows)
    assert [(row["id"], row["reserved"]) for row in reserved] == [("first", True), ("overlap", False), ("touching", True)]


def test_repository_path_rejects_absolute_and_parent_escape() -> None:
    assert tpct.repository_path("docs/tri-party-composition-topology-mechanism-decision-2026-07-24.md").is_absolute()
    for escaped in ["../secret", "docs/../secret", "/tmp/secret"]:
        with pytest.raises(ValueError, match="relative|escaped"):
            tpct.repository_path(escaped)


def test_source_family_multiplicity_and_confirmatory_budget_are_frozen() -> None:
    manifest = tpct.build_manifest(validate_files=False)
    boundary = manifest["contract"]["research_boundary"]
    confirmation = manifest["contract"]["confirmation_2024"]

    assert tuple(boundary["source_family_ledger"]) == ("RVFC-72", "RMSR-72", "RCRE-72", "DMSH-168", "TPCT-120")
    assert boundary["source_family_concepts"] == 5
    assert boundary["source_support_seen"] is True
    assert boundary["ofr_market_outcomes_seen"] is False
    assert boundary["tpct_values_seen"] is False
    assert confirmation["source_family_concepts"] == 5
    assert confirmation["source_family_alpha"] == 0.01
    assert confirmation["bonferroni_combined_2023_2024_p_strictly_below"] == 0.002
    assert confirmation["first_confirmatory_statistical_claim"] is True


def test_full_comparator_novelty_cohort_and_thresholds_are_frozen() -> None:
    novelty = tpct.build_manifest(validate_files=False)["contract"]["novelty"]
    comparator_ids = [row["id"] for row in novelty["comparators"]]

    assert comparator_ids == [identifier for identifier, _, _ in tpct.COMPARATORS]
    assert set(comparator_ids) == {"RVFC", "RMSR", "RCRE", "DMSH", "FED_LIQUIDITY_COMPONENTS", "FROZEN_LIVE_SLEEVES"}
    assert novelty["full_comparator_cohort_required"] is True
    assert novelty["exact_entry_jaccard_max"] == 0.20
    assert novelty["tolerant_24h_jaccard_max"] == 0.50
    assert novelty["unsigned_occupancy_abs_correlation_max"] == 0.75
    assert novelty["directional_signed_exposure_abs_correlation_max"] == 0.50
    assert novelty["live_exact_entry_jaccard_max"] == 0.10
    assert novelty["live_tolerant_24h_jaccard_max_preoutcome"] == 0.35
    assert novelty["live_tolerant_24h_jaccard_max_eval"] == 0.30
    assert novelty["live_signed_exposure_abs_correlation_max_eval"] == 0.35
    assert novelty["live_unsigned_occupancy_abs_correlation_max"] == 0.60
    assert novelty["missing_required_directional_side_fails"] is True


def test_familywise_t_selector_contract_cannot_fall_back_to_ratio_or_return() -> None:
    contract = tpct.build_manifest(validate_files=False)["contract"]

    assert contract["accounting"]["familywise_shared_max_stat"] is True
    assert contract["accounting"]["familywise_statistic"] == "weekly_return_t_policy"
    assert contract["accounting"]["development_primary_selector"] == "highest_observed_t_policy"
    assert contract["accounting"]["family_includes_failed_secondary_gate_policies"] is True
    assert contract["accounting"]["secondary_selection_tie_break_only_after_exact_t_tie"] is True
    assert contract["cheap_gate_2022"]["primary_selector"] == "highest_observed_t_policy"
    assert contract["cheap_gate_2022"]["tie_break"] == ["higher_ratio", "higher_return", "lower_mdd", "lexicographically_smaller_policy_id"]
    assert contract["model"]["selection_2022"]["primary_selector"] == "highest_observed_t_policy"
    assert contract["model"]["selection_2022"]["tie_break"] == ["higher_ratio", "higher_return", "lower_mdd", "earlier_optimizer_step"]


def test_prompt_contract_contains_only_tokens_task_and_neutral_options() -> None:
    tokens = tpct.validate_tokens(tpct.SERIALIZATION_SPECIMEN)
    option_order = ("Q3", "Q1", "Q2")
    text = tpct.build_user_text(tokens, option_order)

    assert not text.endswith("\n")
    assert "\r" not in text
    assert text.splitlines() == [
        "STATE:",
        *(f"{key}={tokens[key]}" for key in tpct.TOKEN_COLUMNS),
        "TASK=TPCT_ACTION",
        "OPTIONS:",
        "Q3=SHORT",
        "Q1=ABSTAIN",
        "Q2=LONG",
        "Return exactly CHOICE=<one option>.",
    ]
    forbidden_fragments = ["2022", "2023", "BTC", "return", "PnL", "source", "rank=", "value", "path", "hash"]
    assert all(fragment not in text for fragment in forbidden_fragments)

    assert len(tpct.action_option_orders()) == 6
    assert set(tpct.action_option_orders()) == set(__import__("itertools").permutations(tpct.NEUTRAL_CODES))
    for code in tpct.NEUTRAL_CODES:
        assert tpct.completion_text(code) == f"CHOICE={code}"
    with pytest.raises(ValueError, match="permutation"):
        tpct.build_user_text(tokens, ("Q1", "Q1", "Q2"))


def test_chat_messages_are_outcome_blind_text_only_user_surface() -> None:
    user_text = tpct.build_user_text(tpct.SERIALIZATION_SPECIMEN, ("Q1", "Q2", "Q3"))

    messages = tpct.chat_messages(user_text)
    assert messages == [{"role": "user", "content": [{"type": "text", "text": user_text}]}]

    train_messages = tpct.chat_messages(user_text, assistant_completion="CHOICE=Q2")
    assert [message["role"] for message in train_messages] == ["user", "assistant"]
    assert train_messages[1]["content"] == [{"type": "text", "text": "CHOICE=Q2"}]
    with pytest.raises(ValueError, match="completion"):
        tpct.chat_messages(user_text, assistant_completion="LONG")


def test_manifest_is_self_hashing_and_outcome_blind_before_later_stages() -> None:
    payload = tpct.build_manifest(validate_files=False)

    tpct.validate_manifest(
        payload,
        allow_unvalidated_anchors=True,
        revalidate_files=False,
    )
    assert payload["anchors"] == {"validation_skipped_for_unit_test": True}
    assert payload["decision"] == {
        "source_support_authorized": True,
        "comparator_novelty_authorized_after_support": True,
        "market_outcomes_authorized": False,
        "model_training_authorized": False,
        "sealed_eval_authorized": False,
        "next_action": "commit and run source-only TPCT support builder without decoding sealed-boundary values",
    }
    assert payload["outcome_boundary"] == {
        "source_artifact_bytes_hashed": False,
        "source_manifest_aggregate_metadata_read": False,
        "source_header_read": False,
        "selected_metadata_objects_read": 0,
        "source_values_decoded": 0,
        "primitives_or_ranks_derived": 0,
        "token_rows_derived": 0,
        "opportunity_rows_derived": 0,
        "sealed_boundary_values_decoded": 0,
        "comparator_rows_parsed": 0,
        "market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "model_labels_created": 0,
        "model_training_runs": 0,
        "network_calls": 0,
    }

    mutated = copy.deepcopy(payload)
    mutated["decision"]["market_outcomes_authorized"] = True
    mutated["manifest_hash"] = tpct.canonical_hash({key: value for key, value in mutated.items() if key != "manifest_hash"})
    with pytest.raises(RuntimeError, match="later stage"):
        tpct.validate_manifest(
            mutated,
            allow_unvalidated_anchors=True,
            revalidate_files=False,
        )

    mutated = copy.deepcopy(payload)
    mutated["outcome_boundary"]["return_or_pnl_fields_read"] = 1
    mutated["manifest_hash"] = tpct.canonical_hash({key: value for key, value in mutated.items() if key != "manifest_hash"})
    with pytest.raises(RuntimeError, match="outcome boundary opened"):
        tpct.validate_manifest(
            mutated,
            allow_unvalidated_anchors=True,
            revalidate_files=False,
        )

    with pytest.raises(RuntimeError, match="unvalidated anchors"):
        tpct.validate_manifest(payload, revalidate_files=False)


def test_serialization_self_test_contract_with_frozen_local_processor() -> None:
    processor = tpct.load_model_processor()
    report = tpct.build_serialization_self_test(processor)

    assert report["self_test_sha256"] == tpct.SERIALIZATION_SELF_TEST_SHA256
    assert report["specimen"] == dict(tpct.SERIALIZATION_SPECIMEN)
    assert len(report["prompt_prefixes"]) == 6
    assert len(report["completions"]) == 3
    assert len(report["full_examples"]) == 18
    assert report["chat_arguments"]["enable_thinking"] is False
    assert report["chat_arguments"]["preserve_thinking"] is False
    assert report["chat_arguments"]["tools"] is None
    assert report["chat_arguments"]["completion_includes_eos_or_turn_close"] is False


def test_persisted_preregistration_revalidates_all_frozen_anchors() -> None:
    path = tpct.repository_path(tpct.DEFAULT_OUTPUT)
    payload = json.loads(path.read_text(encoding="utf-8"))

    tpct.validate_manifest(payload, revalidate_files=True)

    mutated = copy.deepcopy(payload)
    mutated["anchors"] = {"validation_skipped_for_unit_test": True}
    mutated["manifest_hash"] = tpct.canonical_hash(
        {
            key: value
            for key, value in mutated.items()
            if key != "manifest_hash"
        }
    )
    with pytest.raises(RuntimeError, match="unvalidated anchors"):
        tpct.validate_manifest(mutated, revalidate_files=False)
