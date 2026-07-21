from __future__ import annotations

import json
import math
import builtins
import csv
import gzip
import io
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from training import preregister_gdelt_narrative_rotation_clearing as prereg


UTC = timezone.utc
FROZEN_PREREGISTRATION = Path(
    "results/gdelt_narrative_rotation_clearing_preregistration_2026-07-20.json"
)
FROZEN_PREREGISTRATION_SHA256 = (
    "ae175a242db1fa850164789e4a3e6f3f39b4ac8eae0fb877ce79e915ae3d67f3"
)
FROZEN_MANIFEST_HASH = (
    "481aae4d1ebbf147333cfdfe8534d695761932049775354f3bb242a5a786715e"
)


def _count_rows(*, failure_tail: int, constraint_tail: int, adoption_tail: int):
    rows = []
    start = date(2020, 1, 1)
    for index in range(56):
        current = start + timedelta(days=index)
        tail = index >= 49
        rows.append(
            {
                "date": current.isoformat(),
                "available_at": (
                    datetime.combine(current, datetime.min.time(), tzinfo=UTC)
                    + timedelta(hours=48, minutes=15)
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "global_article_count": 100_000 + index * 100,
                "broad_article_count": 1_000 + index * 3,
                "failure_article_count": failure_tail if tail else 20 + index % 5,
                "constraint_article_count": constraint_tail if tail else 25 + index % 7,
                "adoption_article_count": adoption_tail if tail else 30 + index % 9,
            }
        )
    return rows


def test_lattice_is_exactly_twenty_four_unique_variants() -> None:
    variants = prereg.variants()
    assert len(variants) == len({row["variant_id"] for row in variants}) == 24
    assert {row["score"] for row in variants} == set(prereg.SCORE_ARCHETYPES)
    assert {(row["fast_days"], row["slow_days"]) for row in variants} == set(
        prereg.WINDOW_PAIRS
    )
    assert {row["threshold"] for row in variants} == set(prereg.THRESHOLDS)
    assert {row["hold_days"] for row in variants} == set(prereg.HOLD_DAYS)


def test_preregistration_opens_no_source_or_market_outcome() -> None:
    payload = prereg.build_payload()
    assert payload["source_transport"]["daily_artifact_opened"] is False
    assert payload["source_transport"]["raw_bundle_opened"] is False
    assert payload["outcome_boundary"] == {
        "gdelt_daily_rows_read": 0,
        "gdelt_raw_responses_read": 0,
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "outcomes_opened": False,
    }
    oos = payload["economic_protocol"]["post_selection_oos"]
    assert oos["source_start"] == "2024-01-01"
    assert oos["source_end_exclusive"] == "2026-07-01"
    assert oos["minimum_cagr_to_strict_mdd"] == 3.0
    assert oos["maximum_strict_mdd_percent"] == 15.0
    assert oos["minimum_trades"] == 50


def test_source_transport_and_protocol_are_hash_bound() -> None:
    contract = prereg.validate_source_contract()
    assert contract["start_date"] == "2020-01-01"
    assert contract["end_date_exclusive"] == "2024-01-01"
    assert prereg.sha256_file(prereg.SOURCE_BUILDER) == prereg.SOURCE_BUILDER_SHA256
    assert (
        prereg.sha256_file(prereg.SOURCE_PROTOCOL_DOCUMENT)
        == prereg.SOURCE_PROTOCOL_DOCUMENT_SHA256
    )


def test_write_once_preserves_manifest_hash(tmp_path: Path) -> None:
    output = tmp_path / "preregistration.json"
    payload = prereg.write_once(output)
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored == payload
    unhashed = dict(restored)
    unhashed.pop("manifest_hash")
    assert prereg.canonical_hash(unhashed) == restored["manifest_hash"]
    with pytest.raises(FileExistsError, match="write-once"):
        prereg.write_once(output)


def test_frozen_preregistration_artifact_is_exactly_hash_bound() -> None:
    assert prereg.sha256_file(FROZEN_PREREGISTRATION) == FROZEN_PREREGISTRATION_SHA256
    payload = json.loads(
        prereg.repository_path(FROZEN_PREREGISTRATION).read_text(encoding="utf-8")
    )
    unhashed = dict(payload)
    assert unhashed.pop("manifest_hash") == FROZEN_MANIFEST_HASH
    assert prereg.canonical_hash(unhashed) == FROZEN_MANIFEST_HASH
    assert payload["preregistration_source_sha256"] == prereg.sha256_file(
        prereg.PREREGISTRATION_SOURCE
    )
    assert payload["preregistration_document_sha256"] == prereg.sha256_file(
        prereg.PREREGISTRATION_DOCUMENT
    )
    assert payload["outcome_boundary"]["outcomes_opened"] is False


def test_rllm_cannot_control_event_identity_or_side() -> None:
    boundary = prereg.build_payload()["rllm_boundary"]
    assert boundary["part_of_gnrc_primary_claim"] is False
    assert boundary["gnrc_primary_oos_may_be_reused_for_rllm_claim"] is False
    assert boundary["may_retime_reverse_or_create_events"] is False
    assert boundary["status"] == "exploratory_only_until_separate_preregistration"


def test_rule_to_adoption_has_distinct_constructive_and_destructive_sides() -> None:
    constructive = prereg.compute_score_state(
        _count_rows(failure_tail=3, constraint_tail=120, adoption_tail=140), 7, 28
    )
    destructive = prereg.compute_score_state(
        _count_rows(failure_tail=160, constraint_tail=90, adoption_tail=3), 7, 28
    )
    assert constructive["evidence_ok"] is True
    assert destructive["evidence_ok"] is True
    assert (
        constructive["rule_to_adoption"]["long_score"]
        > constructive["rule_to_adoption"]["short_score"]
    )
    assert (
        destructive["rule_to_adoption"]["short_score"]
        > destructive["rule_to_adoption"]["long_score"]
    )
    assert all(
        math.isfinite(state[side])
        for state in (
            constructive["rotation"],
            constructive["clearing"],
            constructive["rule_to_adoption"],
        )
        for side in ("long_score", "short_score")
    )


def test_feature_clock_rejects_gaps_and_wrong_availability() -> None:
    rows = _count_rows(failure_tail=10, constraint_tail=10, adoption_tail=10)
    rows[10] = {**rows[10], "available_at": "2020-01-12T00:15:00Z"}
    with pytest.raises(ValueError, match="availability clock"):
        prereg.compute_score_state(rows, 7, 28)
    rows = _count_rows(failure_tail=10, constraint_tail=10, adoption_tail=10)
    rows.pop(10)
    with pytest.raises(ValueError, match="daily and complete"):
        prereg.compute_score_state(rows, 7, 28)


def test_scheduler_locks_conflict_nonoverlap_and_strict_split_exit() -> None:
    def decision(
        source_date: date, long_score: float, short_score: float
    ) -> prereg.ScoreDecision:
        return prereg.ScoreDecision(
            source_date=source_date,
            available_at=datetime.combine(source_date, datetime.min.time(), tzinfo=UTC)
            + timedelta(hours=48, minutes=15),
            long_score=long_score,
            short_score=short_score,
        )

    source_start = date(2020, 12, 30)
    score_by_index = {
        0: (1.2, 0.0),
        1: (1.2, 0.0),
        3: (0.0, 1.2),
        4: (1.2, 1.2),
        5: (0.0, 1.2),
    }
    decisions = [
        decision(
            source_start + timedelta(days=index),
            *score_by_index.get(index, (0.0, 0.0)),
        )
        for index in range(16)
    ]
    result = prereg.schedule_events(
        decisions,
        threshold=1.0,
        hold_days=3,
        split_start=datetime(2021, 1, 1, tzinfo=UTC),
        split_end_exclusive=datetime(2021, 1, 20, tzinfo=UTC),
    )
    assert result.eligible_decisions == 16
    assert result.raw_directional_triggers == 5
    assert result.side_conflicts == 1
    assert [event.source_date for event in result.admitted_events] == [
        date(2020, 12, 30),
        date(2021, 1, 4),
    ]
    assert [event.side for event in result.admitted_events] == [1, -1]
    rates = prereg.support_rates(result)
    assert rates["active_decision_share"] == pytest.approx(2 / 16)
    assert rates["maximum_month_share"] == 1.0

    with pytest.raises(ValueError, match="complete split source grid"):
        prereg.schedule_events(
            decisions[1:],
            threshold=1.0,
            hold_days=3,
            split_start=datetime(2021, 1, 1, tzinfo=UTC),
            split_end_exclusive=datetime(2021, 1, 20, tzinfo=UTC),
        )


def test_build_payload_reads_only_frozen_definition_files(monkeypatch) -> None:
    allowed = {
        prereg.repository_path(path).resolve()
        for path in (
            prereg.SOURCE_BUILDER,
            prereg.SOURCE_PROTOCOL_DOCUMENT,
            prereg.PREREGISTRATION_SOURCE,
            prereg.PREREGISTRATION_DOCUMENT,
        )
    }
    opened: set[Path] = set()
    original_path_open = Path.open
    original_builtin_open = builtins.open
    original_io_open = io.open

    def guard(file) -> Path:
        resolved = Path(file).resolve()
        if resolved not in allowed:
            raise AssertionError(f"unexpected file access: {resolved}")
        opened.add(resolved)
        return resolved

    def recording_open(path: Path, *args, **kwargs):
        guard(path)
        return original_path_open(path, *args, **kwargs)

    def recording_builtin_open(file, *args, **kwargs):
        guard(file)
        return original_builtin_open(file, *args, **kwargs)

    def recording_io_open(file, *args, **kwargs):
        guard(file)
        return original_io_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    monkeypatch.setattr(builtins, "open", recording_builtin_open)
    monkeypatch.setattr(io, "open", recording_io_open)
    prereg.build_payload()
    assert opened <= allowed


def test_familywise_protocol_is_operationally_frozen() -> None:
    familywise = prereg.ECONOMIC_PROTOCOL["familywise_test"]
    assert familywise["draws"] == 100_000
    assert familywise["seed"] == 20_260_720
    assert familywise["block_days"] == 7
    assert familywise["adjusted_p_maximum"] == 0.10
    assert (
        familywise["source_unsupported_train_ineligible_or_zero_variance_adjusted_p"]
        == 1.0
    )
    assert tuple(familywise["family_variant_ids"]) == prereg.FAMILY_VARIANT_IDS


def test_variant_support_predicate_includes_year_half_and_side_gates() -> None:
    train = {
        "admitted_events": 30,
        "entry_year_counts": {"2021": 15, "2022": 15},
        "entry_half_counts": {},
        "long_events": 15,
        "short_events": 15,
        "active_decision_share": 0.10,
        "maximum_month_share": 0.10,
    }
    selection = {
        "admitted_events": 12,
        "entry_year_counts": {"2023": 12},
        "entry_half_counts": {"2023-H1": 6, "2023-H2": 6},
        "long_events": 6,
        "short_events": 6,
        "active_decision_share": 0.08,
        "maximum_month_share": 0.25,
    }
    assert all(prereg.evaluate_variant_support(train, selection).values())
    selection["entry_half_counts"] = {"2023-H1": 10, "2023-H2": 2}
    assert (
        prereg.evaluate_variant_support(train, selection)[
            "minimum_events_each_selection_half"
        ]
        is False
    )


def test_family_support_predicate_is_fail_closed_and_representative() -> None:
    passing = {
        prereg.variant_id("rotation", 7, 28, 0.5, 3),
        prereg.variant_id("rotation", 14, 56, 0.5, 3),
        prereg.variant_id("clearing", 7, 28, 0.5, 3),
        prereg.variant_id("clearing", 14, 56, 0.5, 3),
        prereg.variant_id("rule_to_adoption", 7, 28, 0.5, 3),
        prereg.variant_id("rule_to_adoption", 14, 56, 0.5, 3),
        prereg.variant_id("rotation", 7, 28, 1.0, 7),
        prereg.variant_id("clearing", 14, 56, 1.0, 7),
    }
    support = {
        variant_id: {
            check: variant_id in passing for check in prereg.VARIANT_SUPPORT_CHECK_NAMES
        }
        for variant_id in prereg.FAMILY_VARIANT_IDS
    }
    result = prereg.evaluate_family_support(support)
    assert result["family_advances"] is True
    assert result["passing_variant_count"] == 8
    support.pop(next(iter(support)))
    with pytest.raises(ValueError, match="exactly 24"):
        prereg.evaluate_family_support(support)


def test_integrated_market_path_requires_complete_bars_and_funding() -> None:
    assert prereg.execution_cost(2.0, 100.0, 2.0) == pytest.approx(0.04)
    assert prereg.funding_cash_change(2.0, 100.0, 0.001) == pytest.approx(-0.2)
    assert prereg.funding_cash_change(-2.0, 100.0, 0.001) == pytest.approx(0.2)
    assert prereg.strict_held_bar_prices(1, 110.0, 90.0, 105.0) == (
        110.0,
        90.0,
        105.0,
    )
    assert prereg.strict_held_bar_prices(-1, 110.0, 90.0, 95.0) == (
        90.0,
        110.0,
        95.0,
    )
    split_start = datetime(2021, 1, 1, tzinfo=UTC)
    split_end = datetime(2021, 1, 5, tzinfo=UTC)
    event = prereg.ScheduledEvent(
        source_date=date(2020, 12, 30),
        decision_time=datetime(2021, 1, 1, 0, 15, tzinfo=UTC),
        entry_time=datetime(2021, 1, 1, 0, 25, tzinfo=UTC),
        exit_time=datetime(2021, 1, 4, 0, 25, tzinfo=UTC),
        side=1,
    )
    schedule = prereg.ScheduleResult(0, 0, 0, (event,))
    bars = [
        prereg.MarketBar(
            open_time=split_start + timedelta(minutes=5 * index),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
        )
        for index in range(4 * 24 * 12)
    ]
    funding = [
        prereg.FundingMark(
            timestamp=split_start + timedelta(hours=8 * index),
            mark_price=100.0,
            funding_rate=(0.001 if index == 3 else 0.0),
        )
        for index in range(12)
    ]
    metrics = prereg.evaluate_market_path(
        schedule,
        bars,
        funding,
        split_start=split_start,
        split_end_exclusive=split_end,
        side_cost_bps=2.0,
    )
    assert metrics["absolute_return"] == pytest.approx(-0.0014)
    assert metrics["strict_mdd"] > 0.0
    assert metrics["full_calendar_days"] == 4.0
    assert metrics["trade_count"] == 1
    with pytest.raises(ValueError, match="complete UTC 5m grid"):
        prereg.evaluate_market_path(
            schedule,
            bars[:-1],
            funding,
            split_start=split_start,
            split_end_exclusive=split_end,
            side_cost_bps=2.0,
        )
    with pytest.raises(ValueError, match="complete UTC eight-hour grid"):
        prereg.evaluate_market_path(
            schedule,
            bars,
            funding[:-1],
            split_start=split_start,
            split_end_exclusive=split_end,
            side_cost_bps=2.0,
        )


def test_two_stage_oos_seals_bind_files_and_stage_lineage(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(prereg, "REPOSITORY_ROOT", tmp_path)
    champion = prereg.FAMILY_VARIANT_IDS[0]
    policy_hash = "a" * 64

    def write(relative: str, content: str) -> Path:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_source(dates: list[date]) -> Path:
        path = tmp_path / "data/oos_source.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=prereg.OOS_SOURCE_COLUMNS)
            writer.writeheader()
            for source_date in dates:
                writer.writerow(
                    {
                        "date": source_date.isoformat(),
                        "available_at": (
                            datetime.combine(
                                source_date, datetime.min.time(), tzinfo=UTC
                            )
                            + timedelta(hours=48, minutes=15)
                        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "global_article_count": 100_000,
                        "broad_article_count": 1_000,
                        "failure_article_count": 10,
                        "constraint_article_count": 10,
                        "adoption_article_count": 10,
                    }
                )
        return path

    selection = write(
        "results/selection.json",
        json.dumps(
            {
                "champion_variant_id": champion,
                "champion_policy_hash": policy_hash,
            }
        ),
    )
    source_builder = write("training/oos_source.py", "source\n")
    evaluator = write("training/oos_evaluator.py", "evaluator\n")
    source_seal: dict[str, object] = {
        "protocol_version": "gnrc_oos_access_seal_v1",
        "champion_variant_id": champion,
        "champion_policy_hash": policy_hash,
        "selection_report_path": "results/selection.json",
        "selection_report_sha256": prereg.sha256_file(selection),
        "oos_source_builder_path": "training/oos_source.py",
        "oos_source_builder_sha256": prereg.sha256_file(source_builder),
        "oos_evaluator_path": "training/oos_evaluator.py",
        "oos_evaluator_sha256": prereg.sha256_file(evaluator),
        "oos_source_output_path": "data/oos_source.csv.gz",
        "source_start": "2024-01-01",
        "source_end_exclusive": "2026-07-01",
        "sealed_at": "2026-07-20T00:00:00Z",
        "no_interim_oos_access": True,
    }
    prereg.validate_oos_seal(source_seal, stage="source_access")

    source_seal_path = write(
        str(prereg.OOS_SOURCE_ACCESS_SEAL),
        json.dumps(source_seal, sort_keys=True),
    )
    expected_dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(912)]
    source_output = write_source(expected_dates)
    source_manifest = write("results/oos_source_manifest.json", "{}\n")
    market_seal = {
        **source_seal,
        "sealed_at": "2026-07-20T00:01:00Z",
        "source_access_seal_path": str(prereg.OOS_SOURCE_ACCESS_SEAL),
        "source_access_seal_sha256": prereg.sha256_file(source_seal_path),
        "oos_source_output_sha256": prereg.sha256_file(source_output),
        "oos_source_manifest_path": "results/oos_source_manifest.json",
        "oos_source_manifest_sha256": prereg.sha256_file(source_manifest),
        "oos_source_rows": 912,
        "oos_source_feature_values_inspected": False,
        "oos_source_outcomes_inspected": False,
    }
    prereg.validate_oos_seal(market_seal, stage="market_access")
    market_seal["no_interim_oos_access"] = False
    with pytest.raises(ValueError, match="interim access"):
        prereg.validate_oos_seal(market_seal, stage="market_access")
    market_seal["no_interim_oos_access"] = True
    divergent = dict(market_seal)
    alternate_evaluator = write("training/alternate_evaluator.py", "alternate\n")
    divergent["oos_evaluator_path"] = "training/alternate_evaluator.py"
    divergent["oos_evaluator_sha256"] = prereg.sha256_file(alternate_evaluator)
    with pytest.raises(ValueError, match="diverges from its source seal"):
        prereg.validate_oos_seal(divergent, stage="market_access")
    market_seal["oos_source_rows"] = 911
    with pytest.raises(ValueError, match="row count"):
        prereg.validate_oos_seal(market_seal, stage="market_access")
    malformed_grids = {
        "truncated": expected_dates[:-1],
        "duplicate": [*expected_dates[:-1], expected_dates[-2]],
        "missing": [*expected_dates[:400], *expected_dates[401:]],
        "extra": [*expected_dates, date(2026, 7, 1)],
    }
    for dates in malformed_grids.values():
        source_output = write_source(dates)
        market_seal["oos_source_output_sha256"] = prereg.sha256_file(source_output)
        market_seal["oos_source_rows"] = len(dates)
        with pytest.raises(ValueError, match="complete 912-day grid"):
            prereg.validate_oos_seal(market_seal, stage="market_access")
