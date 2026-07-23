from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training import build_dollar_collateral_liquidity_bank_relay_support as s


def _utc_string(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _h41_raw(
    rows: int = 6,
    *,
    start: str = "2018-01-04",
    equal_log_deltas: bool = False,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=rows, freq="7D")
    if equal_log_deltas:
        levels = 1_000_000.0 * np.exp(np.arange(rows) * 0.01)
    else:
        phase = np.arange(rows, dtype=float)
        levels = 1_000_000.0 * np.exp(
            np.cumsum(0.002 + 0.01 * np.sin(phase / 3.0))
        )
    return pd.DataFrame(
        {
            "release_date": dates.strftime("%Y-%m-%d"),
            "observation_date": (dates - pd.Timedelta(days=7)).strftime(
                "%Y-%m-%d"
            ),
            "available_at_utc": [
                _utc_string(
                    pd.Timestamp(f"{date:%Y-%m-%d} 16:35:00", tz=s.NEW_YORK)
                )
                for date in dates
            ],
            "net_liquidity_usd_millions": [
                format(value, ".12g") for value in levels
            ],
        },
        columns=s.prereg.H41_ALLOWLIST,
    )


def _h8_raw(
    rows: int = 8,
    *,
    start: str = "2018-01-04",
    constant_components: bool = False,
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=rows, freq="7D")
    phase = np.arange(rows, dtype=float)
    output: dict[str, list[str]] = {
        "release_date": dates.strftime("%Y-%m-%d").tolist(),
        "release_time_utc": [
            _utc_string(
                pd.Timestamp(f"{date:%Y-%m-%d} 16:15:00", tz=s.NEW_YORK)
            )
            for date in dates
        ],
        "release_weekday": [date.day_name() for date in dates],
    }
    for adjustment, offset in (("sa", 0.0), ("nsa", 0.7)):
        if constant_components:
            migration = np.full(rows, 2.0 + offset)
            borrowings = np.full(rows, 1.0 + offset)
            cash_stress = np.full(rows, -1.5 - offset)
        else:
            migration = 4.0 * np.sin(phase / 2.3 + offset)
            borrowings = 3.0 * np.cos(phase / 3.1 + offset)
            cash_stress = 2.5 * np.sin(phase / 1.7 + 0.4 + offset)
        large_change = migration / 2.0
        small_change = -migration / 2.0
        cash_change = -cash_stress
        definitions = {
            "large_other_deposits": (1_000.0, large_change),
            "small_other_deposits": (800.0, small_change),
            "small_borrowings": (120.0, borrowings),
            "small_cash_assets": (200.0, cash_change),
        }
        for name, (prior, change) in definitions.items():
            output[f"{adjustment}_{name}_prior"] = [
                format(prior + index * 0.01, ".12g")
                for index in range(rows)
            ]
            output[f"{adjustment}_{name}_latest"] = [
                format(
                    (prior + index * 0.01) * np.exp(change[index] / 10_000.0),
                    ".12g",
                )
                for index in range(rows)
            ]
    return pd.DataFrame(output, columns=s.prereg.H8_ALLOWLIST)


def _rrp_for_h8(
    h8_raw: pd.DataFrame,
    *,
    rows_per_interval: int = 5,
    quarantine_interval: int | None = None,
) -> pd.DataFrame:
    decisions = [
        s._decision_time(str(value)) for value in h8_raw["release_date"]
    ]
    records: list[dict[str, str]] = []
    operation_index = 0
    for interval_index in range(1, len(decisions)):
        previous = decisions[interval_index - 1]
        for offset in range(1, rows_per_interval + 1):
            available = previous + pd.Timedelta(days=offset, hours=-2)
            operation_index += 1
            quarantined = (
                quarantine_interval == interval_index and offset == 2
            )
            records.append(
                {
                    "operation_date": available.strftime("%Y-%m-%d"),
                    "result_available_at_utc": _utc_string(available),
                    "total_amount_accepted_usd": (
                        ""
                        if quarantined
                        else str(
                            int(
                                50_000_000_000
                                + interval_index * 1_000_000_000
                                + offset * 10_000_000
                            )
                        )
                    ),
                    "source_complete": "false" if quarantined else "true",
                    "quarantine_reason": (
                        "late_update" if quarantined else ""
                    ),
                }
            )
    return pd.DataFrame(records, columns=s.prereg.RRP_ALLOWLIST)


def test_loaders_pass_exact_allowlists_and_string_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    frame = _h41_raw()

    def fake_read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
        observed["path"] = path
        observed.update(kwargs)
        return frame.copy()

    monkeypatch.setattr(s.pd, "read_csv", fake_read_csv)
    loaded = s.load_h41_source()
    assert observed["usecols"] == list(s.prereg.H41_ALLOWLIST)
    assert observed["dtype"] == "string"
    assert observed["keep_default_na"] is False
    assert observed["na_filter"] is False
    assert list(loaded.columns) == list(s.prereg.H41_ALLOWLIST)


def test_source_validation_is_fail_closed_and_preserves_valid_zero_rrp() -> None:
    h41 = s.validate_h41_source(_h41_raw())
    assert h41["available_at_utc"].dt.tz is not None
    bad_h41 = _h41_raw()
    bad_h41.loc[2, "net_liquidity_usd_millions"] = "0"
    with pytest.raises(RuntimeError, match="primitive invalid"):
        s.validate_h41_source(bad_h41)

    h8_raw = _h8_raw()
    rrp_raw = _rrp_for_h8(h8_raw, quarantine_interval=3)
    first_complete = rrp_raw.index[rrp_raw["source_complete"].eq("true")][0]
    rrp_raw.loc[first_complete, "total_amount_accepted_usd"] = "0"
    validated = s.validate_rrp_source(rrp_raw)
    assert validated.loc[first_complete, "total_amount_accepted_usd"] == 0.0
    quarantined = validated.loc[~validated["source_complete"]]
    assert quarantined["total_amount_accepted_usd"].isna().all()

    bad_rrp = rrp_raw.copy()
    bad_rrp.loc[bad_rrp["source_complete"].eq("false"), "quarantine_reason"] = ""
    with pytest.raises(RuntimeError, match="lacks quarantine"):
        s.validate_rrp_source(bad_rrp)

    h8 = s.validate_h8_source(h8_raw)
    assert h8["release_time_utc"].dt.tz is not None
    bad_h8 = h8_raw.copy()
    bad_h8.loc[0, "release_weekday"] = "Friday"
    with pytest.raises(RuntimeError, match="weekday mismatch"):
        s.validate_h8_source(bad_h8)
    bad_nsa = h8_raw.copy()
    bad_nsa.loc[0, "nsa_small_cash_assets_latest"] = "0"
    with pytest.raises(RuntimeError, match="primitive invalid"):
        s.validate_h8_source(bad_nsa)


def test_h41_midrank_warmup_appends_and_uses_tie_midpoint() -> None:
    raw = _h41_raw(rows=5, equal_log_deltas=True)
    raw["net_liquidity_usd_millions"] = [
        "100",
        "200",
        "400",
        "800",
        "1600",
    ]
    frame = s.validate_h41_source(raw)
    features = s.build_h41_features(frame, prior_deltas=2)
    assert features["rank_num"].iloc[:3].isna().all()
    assert features.iloc[3]["rank_num"] == 2
    assert features.iloc[3]["center_num"] == 0
    assert features.iloc[4]["previous_emitted_rank_num"] == 2


def test_rrp_interval_rank_reset_never_bridges_quarantine() -> None:
    h8 = s.validate_h8_source(_h8_raw(rows=10))
    rrp = s.validate_rrp_source(
        _rrp_for_h8(_h8_raw(rows=10), quarantine_interval=5)
    )
    features = s.build_rrp_interval_features(
        rrp,
        h8,
        prior_deltas=2,
    )
    assert features.iloc[4]["rank_num"] is not None
    assert bool(features.iloc[5]["complete"]) is False
    assert features.iloc[5]["rank_num"] is None or pd.isna(
        features.iloc[5]["rank_num"]
    )
    assert features.iloc[6]["delta"] is None or pd.isna(
        features.iloc[6]["delta"]
    )
    assert features.iloc[7]["rank_num"] is None or pd.isna(
        features.iloc[7]["rank_num"]
    )
    assert features.iloc[5]["segment"] > features.iloc[4]["segment"]


def test_h8_robust_state_is_strict_prior_and_sa_nsa_separate() -> None:
    h8 = s.validate_h8_source(_h8_raw(rows=8))
    sa = s.build_h8_features(h8, adjustment="sa", prior_observations=3)
    nsa = s.build_h8_features(h8, adjustment="nsa", prior_observations=3)
    assert sa["zscores"].iloc[:3].apply(
        lambda values: all(value is None for value in values)
    ).all()
    assert any(value is not None for value in sa.iloc[3]["zscores"])
    assert sa.iloc[4]["components"] != nsa.iloc[4]["components"]

    constant_raw = _h8_raw(rows=6, constant_components=True)
    for adjustment in ("sa", "nsa"):
        for name in (
            "large_other_deposits",
            "small_other_deposits",
            "small_borrowings",
            "small_cash_assets",
        ):
            constant_raw[f"{adjustment}_{name}_latest"] = constant_raw[
                f"{adjustment}_{name}_prior"
            ]
    constant = s.validate_h8_source(constant_raw)
    invalid = s.build_h8_features(
        constant,
        adjustment="sa",
        prior_observations=3,
    )
    assert not invalid["valid"].any()


def test_joint_state_asof_join_never_uses_post_decision_h41() -> None:
    h8_raw = _h8_raw(rows=125)
    h41 = s.validate_h41_source(_h41_raw(rows=125))
    h8 = s.validate_h8_source(h8_raw)
    rrp = s.validate_rrp_source(_rrp_for_h8(h8_raw))
    states, funnel = s.build_joint_states(h41, rrp, h8)
    assert len(states) == 125
    assert funnel["h41_rank_complete_rows"] == 20
    assert funnel["rrp_rank_complete_intervals"] > 0
    assert states["common_valid"].any()
    assert states.loc[states["common_valid"], "signal_available_time"].le(
        states.loc[states["common_valid"], "decision_time"]
    ).all()

    late_raw = _h41_raw(rows=125)
    last_date = late_raw.iloc[-1]["release_date"]
    late_raw.loc[len(late_raw) - 1, "available_at_utc"] = _utc_string(
        pd.Timestamp(f"{last_date} 17:30:00", tz=s.NEW_YORK)
    )
    late_h41 = s.validate_h41_source(late_raw)
    late_states, _ = s.build_joint_states(late_h41, rrp, h8)
    assert bool(late_states.iloc[-1]["common_valid"]) is False


def _manual_states(rows: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2022-01-06", periods=rows, freq="7D")
    states = []
    h8_rows = []
    previous_side: int | None = None
    patterns = ((50, 3), (-50, -3), (60, -2), (-60, 2))
    for index, date in enumerate(dates):
        release_date = date.strftime("%Y-%m-%d")
        decision = s._decision_time(release_date)
        entry = s._entry_time(release_date)
        h41_center, rrp_center = patterns[index % len(patterns)]
        macro = s._macro_state(h41_center, rrp_center)
        side = int(macro["side_sign"])
        prior_transition = (
            "NO_PRIOR"
            if previous_side is None
            else "PERSIST"
            if previous_side == side
            else "FLIP"
        )
        previous_side = side
        relief = side if index % 2 == 0 else -side
        states.append(
            {
                "h8_index": index,
                "release_date": release_date,
                "release_time": decision - pd.Timedelta(minutes=45),
                "decision_time": decision,
                "entry_time": entry,
                "exit_time": entry + s.HOLD,
                "signal_available_time": decision - pd.Timedelta(minutes=5),
                "excluded": False,
                "common_valid": True,
                "primary_eligible": True,
                "h41_num": h41_center + 104,
                "h41_center": h41_center,
                "h41_stale_num": 104 - h41_center,
                "h41_direction": s._direction(int(np.sign(h41_center))),
                "h41_transition": "PERSIST" if index else "NO_PRIOR",
                "h41_age_bucket": "SAME_DAY",
                "rrp_num": rrp_center + 13,
                "rrp_center": rrp_center,
                "rrp_stale_num": 13 - rrp_center,
                "rrp_segment": 1,
                "rrp_direction": s._direction(int(np.sign(-rrp_center))),
                "rrp_transition": "PERSIST" if index else "NO_PRIOR",
                "rrp_count_bucket": "FIVE",
                "macro_integer": macro["macro_integer"],
                "side_sign": side,
                "macro_relation": macro["macro_relation"],
                "macro_strength": macro["macro_strength"],
                "sa_valid": True,
                "sa_relief_sign": relief,
                "sa_agreement": 2 if index % 2 else 3,
                "nsa_valid": True,
                "nsa_relief_sign": -relief,
                "nsa_agreement": 2,
                "prior_side_transition": prior_transition,
            }
        )
        h8_rows.append(
            {
                "release_date": release_date,
                "release_time_utc": decision - pd.Timedelta(minutes=45),
            }
        )
    return pd.DataFrame(states), pd.DataFrame(h8_rows)


def test_controls_freeze_stale_nsa_random_flip_delay_and_reservation() -> None:
    states, h8 = _manual_states()
    controls, raw_counts, reserved_counts = s.build_controls(states, h8)
    assert tuple(controls) == s.prereg.CONTROL_ORDER
    assert raw_counts["primary"] == len(states)
    primary = controls["primary"]
    flipped = controls["exact_direction_flip"]
    random_side = controls["deterministic_random_side"]
    assert len(primary) == reserved_counts["primary"]
    assert flipped["entry_time"].equals(primary["entry_time"])
    assert s._same_clock_side_control_integrity(
        primary, flipped, mode="flip"
    )
    assert s._same_clock_side_control_integrity(
        primary, random_side, mode="random"
    )
    assert not controls["stale_h41_one_release"].empty
    assert not controls["stale_rrp_one_interval"].empty
    assert not controls["nsa_h8"].empty
    delayed = controls["one_h8_release_execution_delay"]
    assert delayed.iloc[0]["entry_time"] == s._entry_time(
        str(h8.iloc[1]["release_date"])
    )
    assert delayed.iloc[0]["exit_time"] - delayed.iloc[0]["entry_time"] == s.HOLD


def test_control_only_exact_macro_balance_uses_amended_token() -> None:
    macro = s._macro_state(8, 1)
    assert macro["macro_integer"] == 0
    assert macro["side_sign"] == 0
    assert macro["macro_relation"] == "MACRO_BALANCED_OPPOSITION"
    amendment = s.prereg.build_manifest()["source_algebra"]["macro"][
        "control_only_balanced_relation"
    ]
    assert amendment["token"] == macro["macro_relation"]
    assert amendment["primary_eligible"] is False


def test_delayed_control_may_execute_on_excluded_host_release() -> None:
    states, h8 = _manual_states(rows=2)
    states.loc[0, "release_date"] = "2023-03-23"
    states.loc[0, "decision_time"] = s._decision_time("2023-03-23")
    states.loc[0, "entry_time"] = s._entry_time("2023-03-23")
    states.loc[0, "exit_time"] = s._entry_time("2023-03-23") + s.HOLD
    states.loc[0, "signal_available_time"] = (
        s._decision_time("2023-03-23") - pd.Timedelta(minutes=5)
    )
    h8.loc[0, "release_date"] = "2023-03-23"
    h8.loc[0, "release_time_utc"] = (
        s._decision_time("2023-03-23") - pd.Timedelta(minutes=45)
    )
    h8.loc[1, "release_date"] = "2023-03-31"
    h8.loc[1, "release_time_utc"] = (
        s._decision_time("2023-03-31") - pd.Timedelta(minutes=45)
    )
    controls, _, _ = s.build_controls(states.iloc[[0]], h8)
    delayed = controls["one_h8_release_execution_delay"]
    assert len(delayed) == 1
    assert delayed.iloc[0]["entry_time"] == s._entry_time("2023-03-31")


def test_reservation_accepts_entry_equal_previous_exit_and_dst_is_elapsed() -> None:
    state, _ = _manual_states(rows=1)
    first = s._event_from_state("primary", state.iloc[0], side_sign=1)
    second = dict(first)
    second["entry_time"] = first["exit_time"]
    second["decision_time"] = second["entry_time"] - pd.Timedelta(minutes=5)
    second["signal_available_time"] = second["decision_time"]
    second["exit_time"] = second["entry_time"] + s.HOLD
    second["signal_id"] = ""
    second["signal_id"] = s._event_signal_id(second)
    reserved = s.reserve_nonoverlap([first, second])
    assert len(reserved) == 2

    entry = s._entry_time("2023-03-09")
    exit_time = entry + s.HOLD
    assert exit_time - entry == pd.Timedelta(minutes=4_320)
    assert entry.tz_convert(s.NEW_YORK).hour == 17
    assert exit_time.tz_convert(s.NEW_YORK).hour == 18


def _fabricated_controls() -> dict[str, pd.DataFrame]:
    primary_rows = []
    index = 0
    for year in (2020, 2021, 2022, 2023):
        for month in range(1, 13):
            for day in (2, 12, 22):
                local = pd.Timestamp(
                    f"{year}-{month:02d}-{day:02d} 17:05:00",
                    tz=s.NEW_YORK,
                )
                decision = local - pd.Timedelta(minutes=5)
                side = "LONG" if index % 2 == 0 else "SHORT"
                event = {
                    "control": "primary",
                    "signal_id": "",
                    "signal_available_time": decision - pd.Timedelta(minutes=1),
                    "decision_time": decision,
                    "entry_time": local.tz_convert("UTC"),
                    "exit_time": local.tz_convert("UTC") + s.HOLD,
                    "side": side,
                    "h41_direction": "RELIEF" if index % 2 == 0 else "STRESS",
                    "h41_transition": "FLIP" if index else "NO_PRIOR",
                    "rrp_direction": "RELIEF" if index % 3 else "STRESS",
                    "rrp_transition": "PERSIST",
                    "macro_relation": (
                        "MACRO_CONCORDANT"
                        if index % 2 == 0
                        else "MACRO_DISCORDANT_H41_DOMINANT"
                    ),
                    "macro_strength": "WEAK" if index % 2 == 0 else "STRONG",
                    "h8_relief": "RELIEF" if index % 2 == 0 else "STRESS",
                    "h8_agreement": (
                        "TWO_OF_THREE"
                        if index % 2 == 0
                        else "THREE_OF_THREE"
                    ),
                    "bank_relation": (
                        "BANK_SUPPORTS"
                        if index % 2 == 0
                        else "BANK_OPPOSES"
                    ),
                    "h41_age_bucket": "SAME_DAY",
                    "rrp_count_bucket": "FIVE",
                    "prior_side_transition": "FLIP" if index else "NO_PRIOR",
                }
                event["signal_id"] = s._event_signal_id(event)
                primary_rows.append(event)
                index += 1
    primary = s._event_frame(primary_rows)
    controls: dict[str, pd.DataFrame] = {"primary": primary}
    for control in s.prereg.CONTROL_ORDER:
        if control == "primary":
            continue
        rows = []
        for row_index, row in enumerate(primary.to_dict("records")):
            side = row["side"]
            if control == "exact_direction_flip":
                side = "SHORT" if side == "LONG" else "LONG"
            elif control == "deterministic_random_side":
                digest = hashlib.sha256(
                    f"DCLB-864|{s._format_time(row['entry_time'])}".encode(
                        "utf-8"
                    )
                ).digest()
                side = "LONG" if digest[0] < 128 else "SHORT"
            elif control == "h41_only" and row_index % 4 == 0:
                side = "SHORT" if side == "LONG" else "LONG"
            elif control == "rrp_interval_only" and row_index % 5 == 0:
                side = "SHORT" if side == "LONG" else "LONG"
            elif control == "stale_h41_one_release" and row_index % 3 == 0:
                side = "SHORT" if side == "LONG" else "LONG"
            elif control == "stale_rrp_one_interval" and row_index % 5 == 0:
                side = "SHORT" if side == "LONG" else "LONG"
            rows.append(s._reidentity(row, control=control, side=side))
        controls[control] = s._event_frame(rows)
    return controls


def test_dense_balanced_clocks_pass_source_and_composition_gates() -> None:
    controls = _fabricated_controls()
    statistics, source_checks, composition, composition_checks = (
        s.support_checks(controls)
    )
    assert statistics["train"]["events"] == 108
    assert statistics["selection"]["events"] == 36
    assert all(source_checks.values()), {
        name: value for name, value in source_checks.items() if not value
    }
    assert all(composition_checks.values()), {
        name: value for name, value in composition_checks.items() if not value
    }
    assert composition["selection"]["bank_supports_share"] == 0.5


def test_required_control_absence_is_source_support_failure() -> None:
    controls = _fabricated_controls()
    controls["nsa_h8"] = controls["nsa_h8"].loc[
        controls["nsa_h8"]["entry_time"].lt(s.TRAIN_END)
    ].copy()
    _, source_checks, _, composition_checks = s.support_checks(controls)
    key = "selection:required_control:nsa_h8"
    assert source_checks[key] is False
    assert key not in composition_checks
    stage, check = s.first_failure(
        source_checks,
        composition_checks,
        {},
        artifact_eligible=True,
    )
    assert stage == "source_support"
    assert check == key


def test_maximum_gap_uses_new_york_calendar_days_across_dst() -> None:
    def stats_for(first: str, second: str) -> dict[str, object]:
        entries = pd.Series(
            [
                pd.Timestamp(first, tz=s.NEW_YORK).tz_convert("UTC"),
                pd.Timestamp(second, tz=s.NEW_YORK).tz_convert("UTC"),
            ]
        )
        return s.clock_stats(
            pd.DataFrame(
                {
                    "entry_time": entries,
                    "signal_id": ["a", "b"],
                    "side": ["LONG", "SHORT"],
                }
            )
        )

    fall = stats_for(
        "2021-09-08 17:05:00",
        "2021-11-07 17:05:00",
    )
    spring = stats_for(
        "2021-02-01 17:05:00",
        "2021-04-02 17:05:00",
    )
    assert fall["maximum_gap_days"] == 60.0
    assert spring["maximum_gap_days"] == 60.0


def test_first_failure_is_stage_ordered() -> None:
    assert s.first_failure(
        {"support": False},
        {"composition": False},
        {},
        artifact_eligible=True,
    ) == ("source_support", "support")
    assert s.first_failure(
        {"support": True},
        {"composition": False},
        {},
        artifact_eligible=True,
    ) == ("relational_composition", "composition")
    assert s.first_failure(
        {"support": True},
        {"composition": True},
        {},
        artifact_eligible=False,
    ) == ("artifact_eligibility", "synthetic_or_injected_build")


def test_source_failure_short_circuits_comparator_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = _fabricated_controls()
    source_checks = {"forced_source_failure": False}
    monkeypatch.setattr(
        s,
        "support_checks",
        lambda _: ({}, source_checks, {}, {}),
    )
    monkeypatch.setattr(
        s,
        "evaluate_novelty",
        lambda *args, **kwargs: pytest.fail("comparator access"),
    )
    clock_bytes = s.deterministic_clock_bytes(controls)
    report = s._core_payload(
        pd.DataFrame(),
        {},
        controls,
        {name: len(controls[name]) for name in s.prereg.CONTROL_ORDER},
        {name: len(controls[name]) for name in s.prereg.CONTROL_ORDER},
        {
            "h41_rows": 1,
            "rrp_rows": 1,
            "h8_rows": 1,
            "bindings": {},
        },
        s.prereg.build_manifest(),
        clock_bytes,
        clock_path=s.DEFAULT_CLOCK_OUTPUT,
        artifact_eligible=True,
        protocol_git_subprocess_calls=2,
    )
    assert report["comparator_rows_decoded"] == 0
    assert report["first_failing_stage"] == "source_support"


def test_comparator_decoder_uses_exact_parser_and_full_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "comparator.csv.gz"
    frame = pd.DataFrame(
        {
            "group": ["primary"] * 4,
            "entry_time": [
                "2019-01-01T00:00:00Z",
                "2021-01-01T00:00:00Z",
                "2023-12-30T00:00:00Z",
                "2024-02-01T00:00:00Z",
            ],
            "exit_time": [
                "2019-01-02T00:00:00Z",
                "2021-01-02T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-02-02T00:00:00Z",
            ],
            "side": ["LONG", "SHORT", "LONG", "SHORT"],
            "forbidden_outcome": [1.0, 2.0, 3.0, 4.0],
        }
    )
    frame.to_csv(path, index=False, compression="gzip", lineterminator="\n")
    contract = {
        "id": "TEST",
        "path": str(path),
        "sha256": s.sha256_file(path),
        "header_sha256": s.sha256_csv_header(path),
        "usecols": ["group", "entry_time", "exit_time", "side"],
        "parser": s.prereg._comparator_parser(group_columns=["group"]),
        "groups": [
            {
                "filter": {"group": "primary"},
                "minimum_contained_rows": 1,
            }
        ],
        "clock_family": "asynchronous",
    }
    observed: list[object] = []
    original = s.pd.read_csv

    def spy(*args: object, **kwargs: object) -> pd.DataFrame:
        observed.append(kwargs.get("usecols"))
        return original(*args, **kwargs)

    monkeypatch.setattr(s.pd, "read_csv", spy)
    groups, decoded = s._read_comparator_groups(
        {"novelty_contract": {"comparators": [contract]}}
    )
    assert decoded == 4
    item = next(iter(groups.values()))
    assert item["counts"] == {
        "raw_selected_rows": 4,
        "fully_contained_rows": 1,
        "before_window_rows": 1,
        "after_window_rows": 1,
        "boundary_crossing_rows": 1,
    }
    assert observed == [["group", "entry_time", "exit_time", "side"]]

    malformed = frame.copy()
    malformed.loc[3, "group"] = "unselected"
    malformed.loc[3, "side"] = "INVALID"
    malformed.to_csv(
        path,
        index=False,
        compression="gzip",
        lineterminator="\n",
    )
    contract["sha256"] = s.sha256_file(path)
    contract["header_sha256"] = s.sha256_csv_header(path)
    with pytest.raises(
        s.ComparatorContractFailure,
        match="before filtering",
    ) as caught:
        s._read_comparator_groups(
            {"novelty_contract": {"comparators": [contract]}}
        )
    assert caught.value.code == "comparator_artifact_contract"
    assert caught.value.rows_decoded == 4


def test_comparator_contract_failure_is_serialized_before_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = _fabricated_controls()
    monkeypatch.setattr(
        s,
        "support_checks",
        lambda _: (
            {},
            {"source_support": True},
            {},
            {"relational_composition": True},
        ),
    )

    def fail_novelty(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise s.ComparatorContractFailure(
            "comparator_artifact_contract",
            37,
            "invalid comparator row",
        )

    monkeypatch.setattr(s, "evaluate_novelty", fail_novelty)
    clock_bytes = s.deterministic_clock_bytes(controls)
    report = s._core_payload(
        pd.DataFrame(),
        {},
        controls,
        {name: len(controls[name]) for name in s.prereg.CONTROL_ORDER},
        {name: len(controls[name]) for name in s.prereg.CONTROL_ORDER},
        {
            "h41_rows": 1,
            "rrp_rows": 1,
            "h8_rows": 1,
            "bindings": {},
        },
        s.prereg.build_manifest(),
        clock_bytes,
        clock_path=s.DEFAULT_CLOCK_OUTPUT,
        artifact_eligible=True,
        protocol_git_subprocess_calls=2,
    )

    assert report["comparator_rows_decoded"] == 37
    assert report["comparator_status"].endswith("then_failed_closed")
    assert report["novelty"]["contract_failure"]["code"] == (
        "comparator_artifact_contract"
    )
    assert report["first_failing_stage"] == "comparator_novelty"
    assert report["first_failing_check"] == (
        "comparator_contract:comparator_artifact_contract"
    )
    assert report["decision"] == "retire_DCLB_864_unchanged_before_outcomes"
    s.validate_report(report)


def test_occupancy_and_tolerant_matching_are_fail_closed() -> None:
    start = s.COMMON_START
    overlap = pd.DataFrame(
        {
            "entry_time": [start, start + pd.Timedelta(hours=12)],
            "exit_time": [
                start + pd.Timedelta(days=1),
                start + pd.Timedelta(days=2),
            ],
            "side_sign": [1, -1],
        }
    )
    with pytest.raises(RuntimeError, match="overlaps itself"):
        s._signed_occupancy(overlap, s.COMMON_START, s.COMMON_END)
    assert s.maximum_tolerant_matches(
        [start, start + pd.Timedelta(hours=10)],
        [start + pd.Timedelta(hours=5)],
        pd.Timedelta(hours=6),
    ) == 1


def test_deterministic_clock_is_canonical_and_has_no_raw_columns() -> None:
    controls = _fabricated_controls()
    first = s.deterministic_clock_bytes(controls)
    second = s.deterministic_clock_bytes(controls)
    assert first == second
    assert first[4:8] == b"\x00\x00\x00\x00"
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as handle:
        text = handle.read().decode("utf-8")
    header = text.splitlines()[0].split(",")
    assert header == list(s.CLOCK_COLUMNS)
    assert not any(
        token in column.lower()
        for column in header
        for token in s.FORBIDDEN_CLOCK_TOKENS
    )


def test_synthetic_report_never_opens_comparators_or_outcomes() -> None:
    h8_raw = _h8_raw(rows=125)
    report, _ = s.build_support_from_frames(
        _h41_raw(rows=125),
        _rrp_for_h8(h8_raw),
        h8_raw,
    )
    assert report["artifact_eligible"] is False
    assert report["comparator_rows_decoded"] == 0
    assert report["outcomes_opened"] is False
    assert report["funding_loaded"] is False
    assert report["outcome_boundary"]["btc_market_rows_loaded"] == 0
    assert report["outcome_boundary"]["protocol_git_subprocess_calls"] == 0
    s.validate_report(report)


def test_write_once_is_repository_confined_durable_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    assert s._write_once("artifact.bin", b"alpha") == "created"
    assert s._write_once("artifact.bin", b"alpha") == "verified_existing"
    with pytest.raises(RuntimeError, match="noncanonical"):
        s._write_once("artifact.bin", b"beta")
    with pytest.raises(RuntimeError, match="repository-relative"):
        s._write_once("../escape.bin", b"alpha")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent missing or symlinked"):
        s._write_once("linked/artifact.bin", b"alpha")


def test_write_once_same_content_and_drift_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)

    def same_winner(*args: object, **kwargs: object) -> None:
        del args, kwargs
        (tmp_path / "race.bin").write_bytes(b"alpha")
        raise FileExistsError

    monkeypatch.setattr(s.os, "link", same_winner)
    assert s._write_once("race.bin", b"alpha") == "verified_existing"
    assert not list(tmp_path.glob(".race.bin.*.tmp"))

    (tmp_path / "race.bin").unlink()

    def drift_winner(*args: object, **kwargs: object) -> None:
        del args, kwargs
        (tmp_path / "race.bin").write_bytes(b"beta")
        raise FileExistsError

    monkeypatch.setattr(s.os, "link", drift_winner)
    with pytest.raises(RuntimeError, match="artifact race drift"):
        s._write_once("race.bin", b"alpha")
    assert not list(tmp_path.glob(".race.bin.*.tmp"))


def test_run_checks_commit_before_any_source_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        s,
        "_assert_protocol_committed",
        lambda: (_ for _ in ()).throw(RuntimeError("not committed")),
    )
    monkeypatch.setattr(
        s,
        "load_h41_source",
        lambda: pytest.fail("source opened before commit proof"),
    )
    with pytest.raises(RuntimeError, match="not committed"):
        s.run()


def test_dirty_preregistration_builder_blocks_protocol_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> SimpleNamespace:
        calls.append(args)
        if args[0] == "ls-files":
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(s, "_git_check", fake_git)
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        s._assert_protocol_committed()
    flattened = " ".join(" ".join(call) for call in calls)
    assert str(s.PREREGISTRATION_BUILDER) in flattened
    assert str(s.BASE_PREREGISTRATION_BUILDER) in flattened


def test_contract_hash_and_preregistration_are_bound() -> None:
    assert s.sha256_file(s.IMPLEMENTATION_CONTRACT) == (
        s.IMPLEMENTATION_CONTRACT_SHA256
    )
    payload = s.validate_preregistration()
    assert payload["manifest_hash"] == s.PREREGISTRATION_MANIFEST_HASH
    assert payload["outcomes_opened"] is False
    assert s.sha256_file(s.PREREGISTRATION_BUILDER) == (
        s.PREREGISTRATION_BUILDER_SHA256
    )
