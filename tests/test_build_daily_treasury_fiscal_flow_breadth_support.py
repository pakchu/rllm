from __future__ import annotations

from datetime import date
from pathlib import Path

import gzip
import json
import math

import pandas as pd
import pytest

from training import build_daily_treasury_fiscal_flow_breadth_support as builder


def _report_dates() -> list[tuple[str, str]]:
    warmup = pd.bdate_range("2019-01-01", "2020-12-31")[:503]
    train = pd.bdate_range("2021-01-01", "2022-12-31")[:502]
    selection = pd.bdate_range("2023-01-01", "2023-12-31")[:250]
    rows = [(day.date().isoformat(), "warmup") for day in warmup]
    rows += [(day.date().isoformat(), "train") for day in train]
    rows += [(day.date().isoformat(), "selection") for day in selection]
    rows.append(("2023-12-29", "boundary_quarantine"))
    return rows


def _source_row(
    record_date: str,
    stage: str,
    *,
    table_id: str,
    side: str,
    parent: str,
    label: str,
    amount: str,
    raw_label: str | None = None,
    row_kind: str = "detail",
) -> dict[str, str]:
    available = f"{record_date}T21:00:00Z"
    execution = f"{record_date}T21:05:00Z"
    return {
        "record_date": record_date,
        "source_available_not_before_utc": available,
        "earliest_execution_time_utc": execution,
        "research_stage": stage,
        "table_id": table_id,
        "side": side,
        "parent_section": parent,
        "raw_category_label": raw_label or label,
        "normalized_category_label": label,
        "today_amount_usd_millions": amount,
        "today_amount_literal": amount,
        "month_to_date_amount_usd_millions": "",
        "month_to_date_amount_literal": "",
        "fiscal_year_to_date_amount_usd_millions": "",
        "fiscal_year_to_date_amount_literal": "",
        "footnote_markers": "",
        "missing_value_tokens": "",
        "row_kind": row_kind,
        "page_number": "1",
        "source_order": "1",
        "source_pdf_sha256": "0" * 64,
    }


def _synthetic_source(
    *,
    amount_for: dict[tuple[int, str], str | None] | None = None,
    omit_for: set[tuple[int, str]] | None = None,
    parent_for: dict[tuple[int, str], str] | None = None,
    label_for: dict[tuple[int, str], str] | None = None,
) -> pd.DataFrame:
    amount_for = amount_for or {}
    omit_for = omit_for or set()
    parent_for = parent_for or {}
    label_for = label_for or {}
    rows: list[dict[str, str]] = []
    for index, (record_date, stage) in enumerate(_report_dates()):
        rows.append(
            _source_row(
                record_date,
                stage,
                table_id="II",
                side="deposit",
                parent="Account totals",
                label="Total TGA Deposits",
                amount=str(1_000 + index),
            )
        )
        rows.append(
            _source_row(
                record_date,
                stage,
                table_id="II",
                side="withdrawal",
                parent="Account totals",
                label="Total TGA Withdrawals",
                amount=str(900 + index),
            )
        )
        for table_id, side in (
            ("II", "deposit"),
            ("II", "withdrawal"),
            ("IIIA", "issue"),
            ("IIIA", "redemption"),
        ):
            key = (index, side)
            if key in omit_for:
                continue
            amount = amount_for.get(key, "100")
            rows.append(
                _source_row(
                    record_date,
                    stage,
                    table_id=table_id,
                    side=side,
                    parent=parent_for.get(key, "Parent A"),
                    label=label_for.get(key, f"{side.title()} Category"),
                    amount="" if amount is None else amount,
                )
            )
    return pd.DataFrame(rows, columns=builder.SOURCE_COLUMNS)


def _features(records: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "policy_id": builder.POLICY_ID,
        "research_stage": "train",
        "deposit_breadth": 1.0,
        "withdrawal_breadth": 1.0,
        "issue_breadth": 1.0,
        "redemption_breadth": 1.0,
        "deposit_eligible_categories": 1,
        "withdrawal_eligible_categories": 1,
        "issue_eligible_categories": 1,
        "redemption_eligible_categories": 1,
        "cash_impulse": 1.0,
        "debt_impulse": 1.0,
        "cash_rank126": 1.0,
        "debt_rank126": 1.0,
        "total_net_cash": 1,
        "total_net_cash_rank126": 1.0,
    }
    normalized = [{**defaults, **record} for record in records]
    frame = pd.DataFrame(normalized)
    for column in ("decision_time_utc", "entry_time_utc", "exit_time_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame[builder.REPORT_COLUMNS]


def _clock_rows(specs: list[tuple[str, int]]) -> pd.DataFrame:
    rows = []
    for stamp, side in specs:
        entry = pd.Timestamp(stamp, tz="UTC")
        rows.append(
            {
                "policy_id": builder.POLICY_ID,
                "clock": "primary",
                "window": "train" if entry.year < 2023 else "selection",
                "signal_record_date": entry.date().isoformat(),
                "execution_record_date": entry.date().isoformat(),
                "decision_time_utc": entry - pd.Timedelta(minutes=5),
                "entry_time_utc": entry,
                "exit_time_utc": entry + pd.Timedelta(hours=24),
                "side": side,
                **{column: 1 for column in builder.FEATURE_COLUMNS},
            }
        )
    return pd.DataFrame(rows, columns=builder.CLOCK_COLUMNS)


def test_label_exclusion_and_canonical_identity_are_source_only() -> None:
    assert builder.exclusion_key(" Public\u2011Debt  Cash   Issues ") == (
        "public-debt cash issues"
    )
    assert builder.identity_key("Résumé — A/B") == "rsumab"
    excluded = _source_row(
        "2021-01-04",
        "train",
        table_id="II",
        side="deposit",
        parent="Totals",
        label="Total Federal Reserve Account",
        amount="1",
    )
    bridge = _source_row(
        "2021-01-04",
        "train",
        table_id="II",
        side="deposit",
        parent="Bridge",
        label="Public Debt Cash Issues, Table III-A",
        amount="1",
    )
    kept = _source_row(
        "2021-01-04",
        "train",
        table_id="IIIA",
        side="issue",
        parent="Bills",
        label="Bill Issue",
        amount="1",
    )
    assert not builder._retained_detail_row(pd.Series(excluded))
    assert not builder._retained_detail_row(pd.Series(bridge))
    assert builder._retained_detail_row(pd.Series(kept))

    duplicate = _synthetic_source()
    first_date, first_stage = _report_dates()[0]
    duplicate = pd.concat(
        [
            duplicate,
            pd.DataFrame(
                [
                    _source_row(
                        first_date,
                        first_stage,
                        table_id="II",
                        side="deposit",
                        parent="Parent-A",
                        label="Deposit Category!!",
                        amount="5",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="canonical category collision"):
        builder.build_report_features(duplicate)


def test_signed_integer_missing_and_absent_amount_handling() -> None:
    assert builder._parse_amount("-17", context="synthetic") == -17
    assert builder._parse_amount("", context="synthetic") is None
    with pytest.raises(RuntimeError, match="invalid integer"):
        builder._parse_amount("01", context="synthetic")

    source = _synthetic_source(
        amount_for={(60, "deposit"): None},
        omit_for={(60, "withdrawal")},
    )
    features = builder.build_report_features(source)
    row = features.iloc[60]
    assert math.isnan(row["deposit_breadth"])
    assert row["withdrawal_breadth"] == 0.0


def test_strict_prior_windows_exclude_current_values_and_handle_ties() -> None:
    assert builder.strict_prior_midrank(3, [1, 2, 3, 3]) == pytest.approx(0.75)
    assert builder.strict_prior_midrank(0, [1, 2, 3]) == 0.0
    with pytest.raises(ValueError, match="requires prior"):
        builder.strict_prior_midrank(1, [])

    amounts: dict[tuple[int, str], str | None] = {}
    for side in builder.SIDES:
        amounts[(60, side)] = "100"
        amounts[(61, side)] = "200"
    source = _synthetic_source(amount_for=amounts)
    features = builder.build_report_features(source)
    assert features.iloc[60]["cash_impulse"] == 0.0
    assert features.iloc[61]["cash_impulse"] == 0.0
    assert math.isnan(features.iloc[185]["cash_rank126"])
    assert features.iloc[186]["cash_rank126"] == 0.5


def test_substantive_parent_or_label_change_resets_category_history() -> None:
    source = _synthetic_source(
        parent_for={(60, "deposit"): "New Parent"},
        label_for={(60, "withdrawal"): "New Withdrawal Category"},
    )
    features = builder.build_report_features(source)
    row = features.iloc[60]
    assert row["deposit_eligible_categories"] == 1
    assert row["withdrawal_eligible_categories"] == 1
    assert row["deposit_breadth"] == 0.0
    assert row["withdrawal_breadth"] == 0.0


def test_side_breadth_cash_debt_impulses_and_total_cash_rank() -> None:
    amounts: dict[tuple[int, str], str | None] = {
        (60, "deposit"): "200",
        (60, "withdrawal"): "0",
        (60, "issue"): "0",
        (60, "redemption"): "200",
    }
    source = _synthetic_source(amount_for=amounts)
    features = builder.build_report_features(source)
    row = features.iloc[60]
    assert row["deposit_breadth"] == 1.0
    assert row["withdrawal_breadth"] == 0.0
    assert row["issue_breadth"] == 0.0
    assert row["redemption_breadth"] == 1.0
    assert row["cash_impulse"] == -1.0
    assert row["debt_impulse"] == 1.0
    assert row["total_net_cash"] == -100
    assert features.iloc[126]["total_net_cash_rank126"] == 0.5


def test_primary_events_have_directions_24h_nonoverlap_and_window_containment() -> None:
    features = _features(
        [
            {
                "record_date": "2021-01-04",
                "decision_time_utc": "2021-01-04T20:55:00Z",
                "entry_time_utc": "2021-01-04T21:00:00Z",
                "exit_time_utc": "2021-01-05T21:00:00Z",
                "cash_rank126": 1.0,
                "debt_rank126": 1.0,
            },
            {
                "record_date": "2021-01-05",
                "decision_time_utc": "2021-01-05T08:55:00Z",
                "entry_time_utc": "2021-01-05T09:00:00Z",
                "exit_time_utc": "2021-01-06T09:00:00Z",
                "cash_rank126": 0.0,
                "debt_rank126": 0.0,
            },
            {
                "record_date": "2021-01-06",
                "decision_time_utc": "2021-01-06T21:00:00Z",
                "entry_time_utc": "2021-01-06T21:00:00Z",
                "exit_time_utc": "2021-01-07T21:00:00Z",
                "cash_rank126": 0.0,
                "debt_rank126": 0.0,
            },
            {
                "record_date": "2022-12-31",
                "decision_time_utc": "2022-12-31T12:00:00Z",
                "entry_time_utc": "2022-12-31T12:00:00Z",
                "exit_time_utc": "2023-01-01T12:00:00Z",
                "cash_rank126": 1.0,
                "debt_rank126": 1.0,
            },
        ]
    )
    clock = builder.build_clock(features, mode="primary", clock="primary")
    assert clock["signal_record_date"].tolist() == ["2021-01-04", "2021-01-06"]
    assert clock["side"].tolist() == [1, -1]
    assert (clock["exit_time_utc"] - clock["entry_time_utc"]).tolist() == [
        pd.Timedelta(hours=24),
        pd.Timedelta(hours=24),
    ]


def test_all_six_controls_include_delay_flip_component_and_random_side() -> None:
    features = _features(
        [
            {
                "record_date": "2021-01-04",
                "decision_time_utc": "2021-01-04T20:55:00Z",
                "entry_time_utc": "2021-01-04T21:00:00Z",
                "exit_time_utc": "2021-01-05T21:00:00Z",
                "cash_rank126": 1.0,
                "debt_rank126": 1.0,
                "total_net_cash_rank126": 1.0,
            },
            {
                "record_date": "2021-01-05",
                "decision_time_utc": "2021-01-05T20:55:00Z",
                "entry_time_utc": "2021-01-05T21:00:00Z",
                "exit_time_utc": "2021-01-06T21:00:00Z",
                "cash_rank126": 0.0,
                "debt_rank126": 0.0,
                "total_net_cash_rank126": 0.0,
            },
            {
                "record_date": "2021-01-06",
                "decision_time_utc": "2021-01-06T20:55:00Z",
                "entry_time_utc": "2021-01-06T21:00:00Z",
                "exit_time_utc": "2021-01-07T21:00:00Z",
                "cash_rank126": 0.5,
                "debt_rank126": 0.5,
                "total_net_cash_rank126": 0.5,
            },
        ]
    )
    primary = builder.build_clock(features, mode="primary", clock="primary")
    controls = builder.build_control_clocks(features, primary)
    assert set(controls["clock"]) == set(builder.CONTROL_NAMES)
    flipped = controls[controls["clock"] == "direction_flip"]
    assert flipped["side"].tolist() == [-1, 1]
    delayed = controls[controls["clock"] == "one_report_delay"]
    assert delayed["execution_record_date"].tolist() == ["2021-01-05", "2021-01-06"]
    random = controls[controls["clock"] == "deterministic_random_side"]
    assert random["side"].tolist() == [
        builder._random_side(value) for value in primary["entry_time_utc"]
    ]
    for name in builder.COMPONENT_CONTROL_NAMES:
        assert len(controls[controls["clock"] == name]) == 2


def test_support_floors_and_month_concentration_gate_pass_and_fail() -> None:
    specs: list[tuple[str, int]] = []
    for year in (2021, 2022):
        for month in range(1, 13):
            side = 1 if month % 2 else -1
            specs.append((f"{year}-{month:02d}-02T00:00:00", side))
    for month in range(1, 13):
        side = 1 if month % 2 else -1
        specs.append((f"2023-{month:02d}-02T00:00:00", side))
    passed = builder.support_gate_summary(_clock_rows(specs))
    assert passed["passed"] is True

    concentrated = _clock_rows(
        [(f"2021-01-{day:02d}T00:00:00", 1 if day % 2 else -1) for day in range(2, 26)]
    )
    failed = builder.support_gate_summary(concentrated)
    assert failed["passed"] is False
    assert failed["checks"]["train_maximum_month_share"] is False


def test_novelty_jaccard_and_one_federal_business_day_boundaries() -> None:
    candidate = frozenset({date(2023, 1, 17), date(2023, 2, 1)})
    comparator = frozenset({date(2023, 1, 13), date(2023, 2, 1)})
    metrics = builder.novelty_metrics(candidate, comparator)
    assert metrics["decision_date_jaccard"] == pytest.approx(1 / 3)
    assert metrics["within_one_us_business_day_count"] == 2
    assert metrics["passed"] is False

    distant = builder.novelty_metrics(
        frozenset({date(2023, 3, 1), date(2023, 4, 3)}),
        frozenset({date(2023, 5, 1), date(2023, 6, 1)}),
    )
    assert distant["passed"] is True


def _intervals(specs: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_time_utc": pd.Timestamp(start, tz="UTC"),
                "exit_time_utc": pd.Timestamp(end, tz="UTC"),
                "side": side,
            }
            for start, end, side in specs
        ]
    )


def test_exposure_correlation_pass_fail_zero_variance_and_nonoverlap() -> None:
    primary = _intervals(
        [
            ("2021-01-01T00:00:00", "2021-01-01T00:05:00", 1),
            ("2021-01-01T00:05:00", "2021-01-01T00:10:00", -1),
            ("2021-01-01T00:10:00", "2021-01-01T00:15:00", 1),
            ("2021-01-01T00:15:00", "2021-01-01T00:20:00", -1),
        ]
    )
    orthogonal = _intervals(
        [
            ("2021-01-01T00:00:00", "2021-01-01T00:05:00", 1),
            ("2021-01-01T00:10:00", "2021-01-01T00:15:00", -1),
        ]
    )
    passed = builder.occupied_exposure_correlation(primary, orthogonal, "cmp")
    assert passed["passed"] is True

    failed = builder.occupied_exposure_correlation(primary, primary.copy(), "cmp")
    assert failed["passed"] is False
    assert failed["absolute_pearson"] == pytest.approx(1.0)

    zero = builder.occupied_exposure_correlation(
        _intervals([("2021-01-01T00:00:00", "2021-01-01T00:10:00", 1)]),
        _intervals([("2021-01-01T00:00:00", "2021-01-01T00:05:00", 1)]),
        "cmp",
    )
    assert zero == {"passed": False, "reason": "empty grid or zero variance"}

    overlapping = _intervals(
        [
            ("2021-01-01T00:00:00", "2021-01-01T00:10:00", 1),
            ("2021-01-01T00:05:00", "2021-01-01T00:15:00", -1),
        ]
    )
    with pytest.raises(RuntimeError, match="intervals overlap"):
        builder.occupied_exposure_correlation(primary, overlapping, "cmp")


def test_config_immutability_and_path_guards(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    good = builder.Config(
        preregistration=str(tmp_path / "prereg.json"),
        output=str(root / "support.json"),
        primary_clock=str(root / "primary.csv.gz"),
        control_clocks=str(root / "controls.csv.gz"),
        artifact_root=str(root),
    )
    builder._validate_config(good)

    with pytest.raises(ValueError, match="must be JSON"):
        builder._validate_config(
            builder.Config(**{**good.__dict__, "output": str(root / "x.txt")})
        )
    with pytest.raises(ValueError, match="distinct"):
        builder._validate_config(
            builder.Config(**{**good.__dict__, "control_clocks": good.primary_clock})
        )
    with pytest.raises(ValueError, match="artifact root"):
        builder._validate_config(
            builder.Config(**{**good.__dict__, "output": str(tmp_path / "x.json")})
        )
    (root / "support.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        builder._validate_config(good)


def test_frozen_preregistration_validation_rejects_tamper_without_real_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg_path = tmp_path / "prereg.json"
    source_path = tmp_path / "source.py"
    source_builder_path = tmp_path / "source_builder.py"
    prereg_path.write_text("{}", encoding="utf-8")
    source_path.write_text("# synthetic\n", encoding="utf-8")
    source_builder_path.write_text("# synthetic calendar\n", encoding="utf-8")
    monkeypatch.setattr(builder, "DEFAULT_PREREGISTRATION", prereg_path)
    monkeypatch.setattr(builder.prereg, "PREREGISTRATION_SOURCE", source_path)
    monkeypatch.setattr(builder.source_builder, "__file__", str(source_builder_path))
    monkeypatch.setattr(builder, "_regular_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(builder, "_repository_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(builder, "sha256_file", lambda path: "file-sha")
    monkeypatch.setattr(builder, "EXPECTED_PREREGISTRATION_FILE_SHA256", "file-sha")
    monkeypatch.setattr(builder, "EXPECTED_PREREGISTRATION_MANIFEST_HASH", "manifest")
    monkeypatch.setattr(builder, "EXPECTED_POLICY_HASH", "policy")
    monkeypatch.setattr(builder, "EXPECTED_PREREGISTRATION_SOURCE_SHA256", "file-sha")
    monkeypatch.setattr(builder, "canonical_hash", lambda payload: "policy")
    monkeypatch.setattr(builder.prereg, "policy", lambda: {"frozen": True})

    valid = {
        "manifest_hash": "manifest",
        "policy_hash": "policy",
        "policy": {"frozen": True},
        "outcomes_opened": False,
        "incidence_or_support_results": None,
        "preregistration_source": {"path": str(source_path), "sha256": "file-sha"},
        "source_binding": {
            "source_builder": {
                "path": str(source_builder_path),
                "sha256": "file-sha",
            }
        },
    }
    monkeypatch.setattr(builder.prereg, "load_preregistration", lambda path: valid)
    assert builder.validate_frozen_preregistration(prereg_path) == valid

    tampered = {**valid, "outcomes_opened": True}
    monkeypatch.setattr(builder.prereg, "load_preregistration", lambda path: tampered)
    with pytest.raises(RuntimeError, match="opened outcomes"):
        builder.validate_frozen_preregistration(prereg_path)


def test_frozen_source_builder_calendar_hash_rejects_drift_before_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg_path = tmp_path / "prereg.json"
    prereg_source = tmp_path / "preregister.py"
    source_builder_path = tmp_path / "source_builder.py"
    for path in (prereg_path, prereg_source, source_builder_path):
        path.write_text("# synthetic\n", encoding="utf-8")
    monkeypatch.setattr(builder, "DEFAULT_PREREGISTRATION", prereg_path)
    monkeypatch.setattr(builder.prereg, "PREREGISTRATION_SOURCE", prereg_source)
    monkeypatch.setattr(builder.source_builder, "__file__", str(source_builder_path))
    monkeypatch.setattr(builder, "_regular_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(builder, "_repository_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(builder, "sha256_file", lambda path: "current-sha")
    monkeypatch.setattr(builder, "EXPECTED_PREREGISTRATION_FILE_SHA256", "current-sha")
    monkeypatch.setattr(builder, "EXPECTED_PREREGISTRATION_MANIFEST_HASH", "manifest")
    monkeypatch.setattr(builder, "EXPECTED_POLICY_HASH", "policy")
    monkeypatch.setattr(
        builder, "EXPECTED_PREREGISTRATION_SOURCE_SHA256", "current-sha"
    )
    monkeypatch.setattr(builder, "canonical_hash", lambda payload: "policy")
    monkeypatch.setattr(builder.prereg, "policy", lambda: {"frozen": True})
    registration = {
        "manifest_hash": "manifest",
        "policy_hash": "policy",
        "policy": {"frozen": True},
        "outcomes_opened": False,
        "incidence_or_support_results": None,
        "preregistration_source": {
            "path": str(prereg_source),
            "sha256": "current-sha",
        },
        "source_binding": {
            "source_builder": {
                "path": str(source_builder_path),
                "sha256": "frozen-sha",
            }
        },
    }
    monkeypatch.setattr(
        builder.prereg, "load_preregistration", lambda path: registration
    )

    with pytest.raises(RuntimeError, match="source-builder SHA drift"):
        builder.validate_frozen_preregistration(prereg_path)


@pytest.mark.parametrize(
    "missing_field", ["reopening", "securityType", "originalSecurityTerm"]
)
def test_auction_raw_rows_require_every_allowlisted_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_field: str
) -> None:
    allowed_fields = {
        "auctionDate",
        "issueDate",
        "cusip",
        "securityType",
        "originalSecurityTerm",
        "reopening",
    }
    raw_row = {
        "auctionDate": "2022-01-03T00:00:00",
        "issueDate": "2022-01-05T00:00:00",
        "cusip": "SYNTHETIC",
        "securityType": "Note",
        "originalSecurityTerm": "2-Year",
        "reopening": "No",
    }
    raw_row.pop(missing_field)
    raw_path = tmp_path / "auction.json.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump({"securityList": [raw_row]}, handle)
    panel = pd.DataFrame([{"auction_date": "2022-01-03", "cusip": "SYNTHETIC"}])
    monkeypatch.setattr(builder.prereg, "AUCTION_PANEL", tmp_path / "panel.csv.gz")
    monkeypatch.setattr(builder, "_read_bound_csv", lambda **kwargs: panel.copy())
    registration = {
        "comparator_binding": {
            "official_auction_settlement_calendar": {
                "normalized_panel": {
                    "sha256": "synthetic",
                    "header": list(panel.columns),
                    "allowed_columns": list(panel.columns),
                },
                "raw_allowed_fields": sorted(allowed_fields),
                "raw_pages": [
                    {
                        "path": str(raw_path),
                        "sha256": builder.sha256_file(raw_path),
                    }
                ],
            }
        }
    }

    with pytest.raises(RuntimeError, match="missing allowlisted fields"):
        builder.load_auction_settlement_calendar(registration)
