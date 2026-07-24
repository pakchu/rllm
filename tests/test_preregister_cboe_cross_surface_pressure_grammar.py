from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
import gzip
import json

import pytest

from training import preregister_cboe_cross_surface_pressure_grammar as p


def _ordered(values: tuple[float, float, float]) -> OrderedDict[str, float]:
    return OrderedDict(zip(p.SURFACES, values, strict=True))


def _valid_tokens() -> OrderedDict[str, str]:
    return OrderedDict(
        (
            ("term_level", "LOW"),
            ("tail_level", "MID"),
            ("option_level", "HIGH"),
            ("term_change", "DOWN"),
            ("tail_change", "SAME"),
            ("option_change", "UP"),
            ("stress_leader", "OPTION"),
            ("relief_leader", "TERM"),
            ("dispersion", "FRACTURED"),
            ("agreement", "POLARIZED"),
            ("topology_transition", "ROTATING"),
            ("pressure_breadth", "BALANCED"),
        )
    )


def test_manifest_is_outcome_blind_and_binds_candidate_level_eval() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "CSPG-288"
    assert payload["research_history_boundary"]["global_pristine_holdout_claimed"] is False
    assert payload["research_history_boundary"]["claim_scope"] == (
        "candidate-level frozen 2023 outcome window"
    )
    assert payload["temporal_roles"]["initial_fit"] == [
        "2020-01-01T00:00:00Z",
        "2021-01-01T00:00:00Z",
    ]
    assert payload["temporal_roles"]["selection"] == [
        "2022-01-01T00:00:00Z",
        "2023-01-01T00:00:00Z",
    ]
    assert payload["temporal_roles"]["untouched_eval"] == [
        "2023-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ]
    assert payload["temporal_roles"]["adaptation"] is None
    assert all(
        value == 0
        for key, value in payload["outcome_boundary"].items()
        if key not in {"source_artifact_bytes_hashed", "source_headers_read"}
    )


def test_source_rank_and_token_contracts_are_frozen() -> None:
    payload = p.build_manifest()
    source = payload["source_contracts"]
    assert source["term"]["allowlist"] == list(p.TERM_ALLOWLIST)
    assert source["tail"]["allowlist"] == list(p.TAIL_ALLOWLIST)
    assert source["option"]["allowlist"] == list(p.OPTION_ALLOWLIST)
    assert all(
        contract["loader"] == (
            "pandas.read_csv(usecols=allowlist); no load-and-drop"
        )
        for contract in (source["term"], source["tail"], source["option"])
    )
    assert source["missing_policy"].startswith("no fill")
    rank = payload["rank_contract"]
    assert rank["lookback"] == 252
    assert rank["minimum"] == 126
    assert rank["current_excluded"] is True
    assert rank["source_histories_independent_before_join"] is True
    token = payload["token_contract"]
    assert token["count"] == 12
    assert [item["name"] for item in token["ordered_schema"]] == list(
        p.TOKEN_COLUMNS
    )
    assert token["position"] == "deterministic guard, not a token"
    assert "date_time_row_or_source_identity" in token["forbidden"]
    assert "btc_price_return_funding_premium_oi_kimchi_dxy" in token[
        "forbidden"
    ]


def test_model_contract_is_one_causal_lm_with_train_only_prior_correction() -> None:
    contract = p.build_manifest()["rllm_contract"]
    assert contract["model"] == "google/gemma-2-2b-it"
    assert contract["revision"] == p.MODEL_REVISION
    assert contract["loader"] == "AutoModelForCausalLM"
    assert contract["trust_remote_code"] is False
    assert contract["tasks"] == ["ADMISSION", "DIRECTION"]
    assert contract["neutral_codes"] == ["Q1", "Q2"]
    assert contract["code_orders"] == [["Q1", "Q2"], ["Q2", "Q1"]]
    assert contract["mapping"]["ADMISSION"] == {
        "Q1": "ABSTAIN",
        "Q2": "TRADE",
    }
    assert contract["mapping"]["DIRECTION"] == {
        "Q1": "LONG",
        "Q2": "SHORT",
    }
    assert contract["generation"] is False
    assert "adapted-base" in contract["prior_correction"]
    assert "2020-2021" in contract["prior_correction"]
    assert contract["offset_reuse"].startswith("hash-frozen")
    assert contract["ties_or_errors"] == "ABSTAIN"
    assert contract["memory_gib"]["inference_reserved_max"] == 6.5


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


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (0.0, "LOW"),
        (1.0 / 3.0, "MID"),
        (0.5, "MID"),
        (2.0 / 3.0, "MID"),
        (1.0, "HIGH"),
    ),
)
def test_pressure_level_exact_boundaries(value: float, expected: str) -> None:
    assert p.pressure_level(value) == expected


def test_change_extremes_dispersion_and_agreement_are_exact() -> None:
    assert p.change_token("LOW", "HIGH") == "DOWN"
    assert p.change_token("MID", "MID") == "SAME"
    assert p.change_token("HIGH", "LOW") == "UP"

    assert p.extreme_leader(_ordered((0.1, 0.2, 0.3)), highest=True) == "OPTION"
    assert p.extreme_leader(_ordered((0.1, 0.2, 0.3)), highest=False) == "TERM"
    assert p.extreme_leader(_ordered((0.3, 0.3, 0.1)), highest=True) == "TIE"

    assert p.dispersion_token(_ordered((0.0, 0.0, 1.0 / 6.0 - 1e-12))) == (
        "COMPRESSED"
    )
    assert p.dispersion_token(_ordered((0.0, 0.0, 1.0 / 6.0))) == "SEPARATED"
    assert p.dispersion_token(_ordered((0.0, 0.0, 1.0 / 3.0))) == "FRACTURED"

    assert p.agreement_token(
        OrderedDict((("TERM", "LOW"), ("TAIL", "LOW"), ("OPTION", "LOW")))
    ) == "UNISON"
    assert p.agreement_token(
        OrderedDict((("TERM", "LOW"), ("TAIL", "MID"), ("OPTION", "LOW")))
    ) == "ADJACENT"
    assert p.agreement_token(
        OrderedDict((("TERM", "LOW"), ("TAIL", "MID"), ("OPTION", "HIGH")))
    ) == "POLARIZED"


@pytest.mark.parametrize(
    ("changed", "expected"),
    ((0, "STABLE"), (1, "STABLE"), (2, "ROTATING"), (3, "ROTATING"),
     (4, "RESET"), (5, "RESET")),
)
def test_topology_transition_exact_change_bands(changed: int, expected: str) -> None:
    fields = (
        "term_level",
        "tail_level",
        "option_level",
        "stress_leader",
        "relief_leader",
    )
    previous = dict.fromkeys(fields, "A")
    current = {
        field: ("B" if index < changed else "A")
        for index, field in enumerate(fields)
    }
    assert p.topology_transition(current, previous) == expected


def test_breadth_token_validation_and_neutral_orders_fail_closed() -> None:
    assert p.pressure_breadth(
        OrderedDict((("TERM", "DOWN"), ("TAIL", "DOWN"), ("OPTION", "UP")))
    ) == "FALLING"
    assert p.pressure_breadth(
        OrderedDict((("TERM", "DOWN"), ("TAIL", "SAME"), ("OPTION", "UP")))
    ) == "BALANCED"
    assert p.pressure_breadth(
        OrderedDict((("TERM", "DOWN"), ("TAIL", "UP"), ("OPTION", "UP")))
    ) == "RISING"
    assert p.validate_tokens(_valid_tokens()) == dict(_valid_tokens())
    reversed_tokens = OrderedDict(reversed(tuple(_valid_tokens().items())))
    with pytest.raises(ValueError, match="order or schema"):
        p.validate_tokens(reversed_tokens)
    bad = _valid_tokens()
    bad["dispersion"] = "UNKNOWN"
    with pytest.raises(ValueError, match="token level is invalid"):
        p.validate_tokens(bad)
    assert p.neutral_code_orders() == (("Q1", "Q2"), ("Q2", "Q1"))


def test_calendar_clock_allows_weekends_and_tracks_dst() -> None:
    weekend = p.opportunity_times(date(2023, 1, 6))
    assert weekend["signal_available"] == datetime(
        2023, 1, 7, 14, 30, tzinfo=timezone.utc
    )
    assert weekend["entry"] == datetime(
        2023, 1, 7, 14, 35, tzinfo=timezone.utc
    )
    assert weekend["exit"] - weekend["entry"] == timedelta(hours=24)

    spring = p.opportunity_times("2023-03-11")
    assert spring["signal_available"] == datetime(
        2023, 3, 12, 13, 30, tzinfo=timezone.utc
    )
    assert spring["entry"] == datetime(
        2023, 3, 12, 13, 35, tzinfo=timezone.utc
    )

    fall = p.opportunity_times("2023-11-04")
    assert fall["signal_available"] == datetime(
        2023, 11, 5, 14, 30, tzinfo=timezone.utc
    )
    assert fall["entry"] == datetime(
        2023, 11, 5, 14, 35, tzinfo=timezone.utc
    )
    assert p.opportunity_times("2023-03-11") == spring


def test_global_reservation_is_action_independent_and_half_open() -> None:
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "id": "overlap",
            "entry": start + timedelta(hours=23),
            "exit": start + timedelta(hours=47),
            "action": "LONG",
        },
        {
            "id": "first",
            "entry": start,
            "exit": start + timedelta(hours=24),
            "action": "ABSTAIN",
        },
        {
            "id": "touching",
            "entry": start + timedelta(hours=24),
            "exit": start + timedelta(hours=48),
            "action": "SHORT",
        },
    ]
    first = p.reserve_intervals(rows)
    second = p.reserve_intervals(
        [{**row, "action": "SHORT" if row["action"] != "SHORT" else "LONG"}
         for row in rows]
    )
    assert [(row["id"], row["reserved"]) for row in first] == [
        ("first", True),
        ("overlap", False),
        ("touching", True),
    ]
    assert [row["reserved"] for row in first] == [
        row["reserved"] for row in second
    ]


def test_frozen_hashes_headers_and_forbidden_dependencies_stay_closed() -> None:
    p.validate_frozen_dependencies()
    assert p.sha256_csv_header(p.TERM_SOURCE) == p.TERM_HEADER_SHA256
    assert p.sha256_csv_header(p.TAIL_SOURCE) == p.TAIL_HEADER_SHA256
    assert p.sha256_csv_header(p.OPTION_SOURCE) == p.OPTION_HEADER_SHA256
    for path, allowlist in (
        (p.TERM_SOURCE, p.TERM_ALLOWLIST),
        (p.TAIL_SOURCE, p.TAIL_ALLOWLIST),
        (p.OPTION_SOURCE, p.OPTION_ALLOWLIST),
    ):
        header = p.csv_header(path)
        assert [column for column in header if column in allowlist] == list(
            allowlist
        )
    assert not set(p.FORBIDDEN_COMPARATOR_PATHS) & set(p.frozen_dependencies())


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
    stored["policy"]["hold_bars"] = 287
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(stored)


def test_validate_manifest_rejects_self_rehashed_drift() -> None:
    payload = p.build_manifest()
    payload["eval_2023_gate"]["ratio_min"] = 2.99
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(payload)
