from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from training import evaluate_ethereum_settlement_demand_impulse_economics as e


def _market(
    start: str,
    end: str,
    *,
    overrides: dict[str, tuple[float, float, float]] | None = None,
) -> pd.DataFrame:
    times = pd.date_range(start, end, freq="5min", tz="UTC")
    rows = {
        timestamp: (100.0, 100.0, 100.0) for timestamp in times
    }
    for raw_time, values in (overrides or {}).items():
        rows[pd.Timestamp(raw_time)] = values
    return pd.DataFrame(
        {
            "timestamp": times,
            "open": [rows[timestamp][0] for timestamp in times],
            "high": [rows[timestamp][1] for timestamp in times],
            "low": [rows[timestamp][2] for timestamp in times],
        },
        columns=e.MARKET_COLUMNS,
    )


def _funding(
    rows: list[tuple[str, float, float]] | None = None,
) -> pd.DataFrame:
    rows = rows or []
    return pd.DataFrame(
        {
            "funding_time": [pd.Timestamp(row[0]) for row in rows],
            "funding_rate": [row[1] for row in rows],
            "settlement_mark": [row[2] for row in rows],
        },
        columns=e.FUNDING_COLUMNS,
    )


def _clock(
    rows: list[tuple[str, str, int]],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": [pd.Timestamp(row[0]) for row in rows],
            "exit_time": [pd.Timestamp(row[1]) for row in rows],
            "side": [row[2] for row in rows],
        },
        columns=e.CLOCK_COLUMNS,
    )


def _registration_and_context() -> tuple[dict[str, Any], e.ValidationContext]:
    registration = e.load_bound_preregistration()
    authority = registration["gross9"]["authority"]
    declared: dict[str, str] = {}

    def add(binding: dict[str, Any]) -> None:
        declared[str(binding["path"])] = str(binding["sha256"])

    add(authority["portfolio"])
    add(authority["base_portfolio"])
    add(authority["transitive_source_manifest"])
    add(authority["pre2025_anchor"])
    for binding in authority["runtime"].values():
        add(binding)
    for sleeve in authority["sleeves"].values():
        add(sleeve["config"])
        if "bundle_manifest" in sleeve:
            add(sleeve["bundle_manifest"])
    source_manifest = {
        "schema_version": 1,
        "sources": [
            {
                "name": "synthetic",
                "path": "synthetic/source.bin",
                "sha256": "a" * 64,
            }
        ],
    }
    rank7_root = Path(
        authority["sleeves"]["frozen_annual_rank7"]["bundle_manifest"]["path"]
    ).parent
    rank7_manifest_core = {
        "models": [
            {
                "path": f"models/seed_{index}.npz",
                "sha256": character * 64,
            }
            for index, character in enumerate(("b", "c", "d"), start=1)
        ],
        "hourly_history": {
            "path": "history/hourly.csv.gz",
            "sha256": "e" * 64,
        },
    }
    rank7_manifest = {
        **rank7_manifest_core,
        "bundle_manifest_hash": e.canonical_hash(rank7_manifest_core),
    }
    rank7_bundle_hashes = {
        str(rank7_root / row["path"]): row["sha256"]
        for row in rank7_manifest_core["models"]
    }
    rank7_bundle_hashes[
        str(rank7_root / rank7_manifest_core["hourly_history"]["path"])
    ] = rank7_manifest_core["hourly_history"]["sha256"]
    context = e.ValidationContext(
        runtime_environment=copy.deepcopy(
            authority["runtime_code_closure"]["exact_runtime_environment"]
        ),
        file_hashes=declared,
        discovered_runtime_closure=tuple(
            path
            for path in authority["runtime_code_closure"]["paths"]
            if path
            not in authority["runtime_code_closure"]["environment_lock_paths"]
        ),
        source_manifest=source_manifest,
        source_hashes={"synthetic/source.bin": "a" * 64},
        rank7_bundle_manifest=rank7_manifest,
        rank7_bundle_hashes=rank7_bundle_hashes,
        repository_sha256=copy.deepcopy(
            registration["frozen_preregistration"]["repository_identity"]["sha256"]
        ),
        repository_git_blobs=copy.deepcopy(
            registration["frozen_preregistration"]["repository_identity"]["git_blobs"]
        ),
    )
    return registration, context


def test_bound_preregistration_hashes_are_exact_and_outcome_blind() -> None:
    registration = e.load_bound_preregistration()

    assert (
        registration["manifest_hash"]
        == "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
    )
    assert e.PREREGISTRATION_ARTIFACT_SHA256 == (
        "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
    )
    assert registration["outcomes_opened"] is False
    assert registration["btc_market_rows_opened"] is False
    assert registration["funding_rows_opened"] is False


def test_frozen_validation_fails_on_environment_dependency_closure_and_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration, context = _registration_and_context()
    assert e.validate_frozen_contract(
        registration, context=context, synthetic=True
    )["validated"]

    unobserved_contract_drift = copy.deepcopy(registration)
    unobserved_contract_drift["source"]["request_chunk_blocks"] += 1
    with pytest.raises(RuntimeError, match="canonical manifest drift"):
        e.validate_frozen_contract(
            unobserved_contract_drift,
            context=context,
            synthetic=True,
        )

    bad_environment = copy.deepcopy(context.runtime_environment)
    bad_environment["python"]["version"] = [9, 9, 9]
    with pytest.raises(RuntimeError, match="runtime environment"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{**context.__dict__, "runtime_environment": bad_environment}
            ),
            synthetic=True,
        )

    bad_hashes = dict(context.file_hashes)
    bad_hashes[next(iter(bad_hashes))] = "0" * 64
    with pytest.raises(RuntimeError, match="dependency SHA drift"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{**context.__dict__, "file_hashes": bad_hashes}
            ),
            synthetic=True,
        )

    extra_hashes = {**context.file_hashes, "synthetic/extra.bin": "f" * 64}
    with pytest.raises(RuntimeError, match="hash inventory"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{**context.__dict__, "file_hashes": extra_hashes}
            ),
            synthetic=True,
        )

    with pytest.raises(RuntimeError, match="import closure"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{
                    **context.__dict__,
                    "discovered_runtime_closure": tuple(
                        context.discovered_runtime_closure[:-1]
                    ),
                }
            ),
            synthetic=True,
        )

    adapter_closure = e._discover_gross9_adapter_closure()
    required_extra = e.source_builder.GROSS9_ADAPTER_EXTRA_CLOSURE_PATHS[-1]
    with monkeypatch.context() as patch:
        patch.setattr(
            e,
            "_discover_gross9_adapter_closure",
            lambda: tuple(
                path for path in adapter_closure if path != required_extra
            ),
        )
        with pytest.raises(RuntimeError, match="adapter import closure"):
            e.validate_frozen_contract(
                registration,
                context=context,
                synthetic=True,
            )
    with monkeypatch.context() as patch:
        patch.setattr(
            e,
            "_discover_gross9_adapter_closure",
            lambda: (
                *adapter_closure,
                Path("training/preregister_ethereum_settlement_demand_impulse.py"),
            ),
        )
        with pytest.raises(RuntimeError, match="adapter import closure"):
            e.validate_frozen_contract(
                registration,
                context=context,
                synthetic=True,
            )

    with pytest.raises(RuntimeError, match="source SHA drift"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{
                    **context.__dict__,
                    "source_hashes": {"synthetic/source.bin": "b" * 64},
                }
            ),
            synthetic=True,
        )


def test_production_gross9_artifact_load_checks_support_before_frozen_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_called = False

    def reject_support(_path: Any) -> dict[str, Any]:
        raise RuntimeError("source support did not pass")

    def forbidden_frozen(_registration: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal frozen_called
        frozen_called = True
        return {}

    monkeypatch.setattr(e, "_load_passed_source_support", reject_support)
    monkeypatch.setattr(e, "validate_frozen_contract", forbidden_frozen)
    with pytest.raises(RuntimeError, match="source support"):
        e.load_gross9_clock_artifact(e.GROSS9_CLOCK_ARTIFACT)
    assert frozen_called is False


def test_frozen_validation_hashes_every_rank7_bundle_dependency() -> None:
    registration, context = _registration_and_context()
    validated = e.validate_frozen_contract(
        registration, context=context, synthetic=True
    )
    assert validated["rank7_bundle_hashes"] == dict(
        sorted(context.rank7_bundle_hashes.items())
    )

    missing = dict(context.rank7_bundle_hashes)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="Rank7 transitive hash inventory"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{**context.__dict__, "rank7_bundle_hashes": missing}
            ),
            synthetic=True,
        )

    drifted = dict(context.rank7_bundle_hashes)
    drifted[next(iter(drifted))] = "0" * 64
    with pytest.raises(RuntimeError, match="Rank7 transitive SHA drift"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{**context.__dict__, "rank7_bundle_hashes": drifted}
            ),
            synthetic=True,
        )

    escaped_manifest = copy.deepcopy(context.rank7_bundle_manifest)
    escaped_manifest["models"][0]["path"] = "../escaped.npz"
    escaped_core = {
        key: value
        for key, value in escaped_manifest.items()
        if key != "bundle_manifest_hash"
    }
    escaped_manifest["bundle_manifest_hash"] = e.canonical_hash(escaped_core)
    with pytest.raises(RuntimeError, match="Rank7 transitive path escapes"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{
                    **context.__dict__,
                    "rank7_bundle_manifest": escaped_manifest,
                }
            ),
            synthetic=True,
        )


def test_reconstruction_requires_validation_and_exact_five_signed_clocks() -> None:
    registration, context = _registration_and_context()
    clocks = {
        sleeve: _clock(
            [("2023-06-01T00:00:00Z", "2023-06-01T00:05:00Z", 1)]
        )
        for sleeve in e.GROSS9_SLEEVES
    }
    reconstructed = e.reconstruct_gross9_sleeve_clocks(
        registration,
        context=context,
        injected_clocks=clocks,
        synthetic=True,
    )
    assert tuple(reconstructed) == e.GROSS9_SLEEVES
    assert all(frame["side"].tolist() == [1] for frame in reconstructed.values())

    with pytest.raises(RuntimeError, match="synthetic-only"):
        e.reconstruct_gross9_sleeve_clocks(
            registration,
            context=context,
            injected_clocks=clocks,
        )
    with pytest.raises(RuntimeError, match="missing, extra, or reordered"):
        e.reconstruct_gross9_sleeve_clocks(
            registration,
            context=context,
            injected_clocks={name: clocks[name] for name in e.GROSS9_SLEEVES[:-1]},
            synthetic=True,
        )
    broken = dict(clocks)
    broken[e.GROSS9_SLEEVES[0]] = _clock(
        [("2023-06-01T00:00:00Z", "2023-06-01T00:05:00Z", 0)]
    )
    with pytest.raises(RuntimeError, match="side is not signed"):
        e.reconstruct_gross9_sleeve_clocks(
            registration,
            context=context,
            injected_clocks=broken,
            synthetic=True,
        )
    broken[e.GROSS9_SLEEVES[0]] = _clock(
        [("2023-06-01T00:00:00Z", "2023-06-01T00:05:00Z", 1.5)]
    )
    with pytest.raises(RuntimeError, match="side is not signed"):
        e.reconstruct_gross9_sleeve_clocks(
            registration,
            context=context,
            injected_clocks=broken,
            synthetic=True,
        )


def test_runtime_trade_replay_verifies_exact_side_and_exit_geometry() -> None:
    class Trade:
        def __init__(
            self,
            signal_position: int,
            entry_position: int,
            exit_position: int,
            side: int,
        ) -> None:
            self.signal_position = signal_position
            self.entry_position = entry_position
            self.exit_position = exit_position
            self.side = side

    observed = Trade(10, 11, 25, 1)
    exact = Trade(10, 11, 25, 1)
    e._verify_exact_trade_replays(
        [observed],
        lambda _trade: exact,
        sleeve="synthetic",
    )

    wrong_exit = Trade(10, 11, 24, 1)
    with pytest.raises(RuntimeError, match="side/exit replay drift"):
        e._verify_exact_trade_replays(
            [observed],
            lambda _trade: wrong_exit,
            sleeve="synthetic",
        )

    wrong_side = Trade(10, 11, 25, -1)
    with pytest.raises(RuntimeError, match="side/exit replay drift"):
        e._verify_exact_trade_replays(
            [observed],
            lambda _trade: wrong_side,
            sleeve="synthetic",
        )


def _runtime_market_rows(
    count: int = 400,
    *,
    start: str = "2023-01-01T00:00:00Z",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                start,
                periods=count,
                freq="5min",
            )
        }
    )


def _event_path_with_exit(count: int, exit_position: int) -> tuple[np.ndarray]:
    event_return = np.zeros(count, dtype=np.float64)
    event_return[exit_position] = 1.0
    return (event_return,)


def test_concrete_mask_clock_adapter_replays_geometry_and_source_counts() -> None:
    market = _runtime_market_rows()
    masks = {"train": np.ones(len(market), dtype=bool)}
    active = np.zeros(len(market), dtype=bool)
    active[143] = True
    calls: list[tuple[int, str, int]] = []

    def event_path(
        _market: pd.DataFrame,
        position: int,
        *,
        side: str,
        hold: int,
        **_kwargs: Any,
    ) -> tuple[np.ndarray]:
        calls.append((position, side, hold))
        return _event_path_with_exit(len(market), position + hold + 1)

    runtime = SimpleNamespace(
        portfolio=SimpleNamespace(
            new_alpha=SimpleNamespace(_event_path=event_path)
        )
    )
    rows, counts = e._reconstruct_mask_long_clock(
        runtime,
        market,
        masks,
        active,
        hold=10,
        stride=12,
    )
    assert counts == {"train": 1}
    assert calls == [(143, "long", 10)]
    assert rows == [
        (
            pd.Timestamp(market["date"].iloc[144]),
            pd.Timestamp(market["date"].iloc[154]),
            1,
        )
    ]
    dates = e._utc_market_dates(market)
    e._verify_fixed_clock_geometry(
        rows,
        dates,
        sleeve="mask",
        hold_bars=10,
        allowed_sides={1},
    )


def test_concrete_rex_taker_adapter_replays_signed_fixed_exit(
    tmp_path: Path,
) -> None:
    market = _runtime_market_rows()
    dates = e._utc_market_dates(market)
    source = {
        "signal_pos": 143,
        "date": dates[143].isoformat(),
        "action": {"side": "long"},
    }
    source_path = tmp_path / "rex.jsonl"
    source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")

    def event_path(
        _market: pd.DataFrame,
        position: int,
        **_kwargs: Any,
    ) -> tuple[np.ndarray]:
        return _event_path_with_exit(len(market), position + 145)

    runtime = SimpleNamespace(
        portfolio=SimpleNamespace(
            resolve_existing=lambda _path: source_path,
            rex_gate_match=lambda _row, _gates: True,
            REX_GATES=("frozen",),
            new_alpha=SimpleNamespace(_event_path=event_path),
        )
    )
    rows, counts = e._reconstruct_rex_taker_clock(
        runtime,
        market,
        {"train": np.ones(len(market), dtype=bool)},
    )
    assert counts == {"train": 1}
    assert rows == [(dates[144], dates[288], 1)]
    e._verify_fixed_clock_geometry(
        rows,
        dates,
        sleeve="rex-taker",
        hold_bars=144,
        allowed_sides={-1, 1},
    )


def test_concrete_rex_veto_adapter_replays_gate_side_and_event_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(e, "REPOSITORY_ROOT", tmp_path)
    market = _runtime_market_rows()
    dates = e._utc_market_dates(market)
    source_path = (
        tmp_path / "data/rex_event_reasoning_policy_sft_20260712.jsonl"
    )
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        json.dumps(
            {
                "signal_pos": 143,
                "date": dates[143].isoformat(),
                "base_event": {"base_side": "short"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def event_path(
        _market: pd.DataFrame,
        position: int,
        **_kwargs: Any,
    ) -> tuple[np.ndarray]:
        return _event_path_with_exit(len(market), position + 145)

    legacy = SimpleNamespace(
        SCAN_FILES={"rex_veto": "frozen-report.json"},
        load_json=lambda _path: {"rows": []},
        _build_light_rex_features=lambda _market: {"frozen": True},
        _rex_row_matches=lambda _gates, _features, _source: True,
    )
    runtime = SimpleNamespace(
        FROZEN_REX_ROW_INDEX=0,
        _unique_rex_rows=lambda _report, _limit: [{"gates": ["frozen"]}],
        legacy_all=legacy,
        portfolio=SimpleNamespace(
            new_alpha=SimpleNamespace(_event_path=event_path)
        ),
    )
    masks = {"train": np.ones(len(market), dtype=bool)}
    rows, counts = e._reconstruct_rex_veto_clock(
        runtime,
        market,
        masks,
    )
    assert counts == {"train": 1}
    assert rows == [(dates[144], dates[288], -1)]
    events = [
        {
            "split": "train",
            "sleeve": "cand_rex_veto_7",
            "entry_positions": [144],
        }
    ]
    e._verify_event_entry_positions(
        events,
        rows,
        dates,
        masks,
        sleeve="cand_rex_veto_7",
    )
    with pytest.raises(RuntimeError, match="reconstructed entries drift"):
        e._verify_event_entry_positions(
            [{**events[0], "entry_positions": [145]}],
            rows,
            dates,
            masks,
            sleeve="cand_rex_veto_7",
        )


def test_concrete_gross9_runtime_adapter_assembles_all_five_signed_clocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _runtime_market_rows(
        900,
        start="2023-06-01T00:00:00Z",
    )
    dates = e._utc_market_dates(market)
    masks = {"train": np.ones(len(market), dtype=bool)}
    helper_rows = {
        "markov": [(dates[1], dates[577], 1)],
        "taker": [(dates[2], dates[146], -1)],
        "veto": [(dates[3], dates[147], 1)],
    }
    fresh_trade = SimpleNamespace(
        signal_position=4,
        entry_position=5,
        exit_position=9,
        side=-1,
    )
    rank7_trade = SimpleNamespace(
        signal_position=5,
        entry_position=6,
        exit_position=10,
        side=1,
    )
    events = [
        {
            "split": "train",
            "sleeve": "markov_transition_long",
            "entry_positions": [1],
        },
        {
            "split": "train",
            "sleeve": "rex_taker_low_range_position",
            "entry_positions": [2],
        },
        {
            "split": "train",
            "sleeve": "cand_rex_veto_7",
            "entry_positions": [3],
            "trade_count": 1,
        },
        {
            "split": "train",
            "sleeve": "fresh_kimchi_fx",
            "entry_positions": [5],
        },
        {
            "split": "train",
            "sleeve": "frozen_annual_rank7",
            "entry_positions": [6],
        },
    ]
    source_meta = {
        "markov_counts": {"train": 1},
        "rex_counts": {"train": 1},
        "path_counts": {
            "fresh_kimchi_fx": {"train": 1},
            "frozen_annual_rank7": {"train": 1},
        },
    }
    replay_calls: list[tuple[str, int, int, tuple[int, ...]]] = []

    class Engine:
        def __init__(self, name: str, expected: Any) -> None:
            self.name = name
            self.expected = expected

        def trade_at(
            self,
            signal: int,
            side: int,
            *action: int,
        ) -> Any:
            replay_calls.append((self.name, signal, side, action))
            return self.expected

    fresh = {
        "market": market,
        "long_active": np.zeros(len(market), dtype=bool),
        "short_active": np.zeros(len(market), dtype=bool),
        "engine": Engine("fresh", fresh_trade),
    }
    fresh["short_active"][fresh_trade.signal_position] = True
    rank7 = {
        "base": {
            "context": {
                "market": market,
                "funding_leg": np.zeros(len(market), dtype=bool),
            },
            "engine": Engine("rank7", rank7_trade),
        }
    }

    class Config:
        market_csv = "market.csv"
        funding_csv = "funding.csv"
        premium_csv = "premium.csv"

    class FreshAuditConfig:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    portfolio = SimpleNamespace(
        feature_frame=lambda _market: pd.DataFrame(index=range(len(market))),
        markov_active=lambda _market, _features: np.ones(
            len(market), dtype=bool
        ),
        FreshAuditConfig=FreshAuditConfig,
        resolve_existing=lambda path: Path("/frozen") / Path(path).name,
        FULL_CUTOFF="2026-06-01T00:00:00Z",
        build_candidate_context=lambda _cfg: fresh,
        build_rank7_context=lambda _cfg: rank7,
        CANDIDATE_SPEC={
            "hold_bars": 4,
            "take_bps": 10,
            "stop_bps": 10,
        },
        rank7_action_spec=lambda _funding_leg: (4, 10, 10),
        SPLIT_BOUNDS={"train": ("2023-06-01", "2023-06-04")},
        candidate_schedule=lambda _context, **_bounds: [fresh_trade],
        rank7_schedule=lambda _context, **_bounds: [rank7_trade],
    )
    runtime = ModuleType(
        "training.audit_gross9_pullback_premium_overheat_marginal"
    )
    runtime.Config = Config
    runtime.portfolio = portfolio
    runtime.build_full_context = lambda _cfg: (
        market,
        masks,
        events,
        source_meta,
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(
        e,
        "_reconstruct_mask_long_clock",
        lambda *_args, **_kwargs: (helper_rows["markov"], {"train": 1}),
    )
    monkeypatch.setattr(
        e,
        "_reconstruct_rex_taker_clock",
        lambda *_args, **_kwargs: (helper_rows["taker"], {"train": 1}),
    )
    monkeypatch.setattr(
        e,
        "_reconstruct_rex_veto_clock",
        lambda *_args, **_kwargs: (helper_rows["veto"], {"train": 1}),
    )

    clocks = e._reconstruct_gross9_runtime_clocks()
    assert tuple(clocks) == e.GROSS9_SLEEVES
    expected = {
        "cand_rex_veto_7": helper_rows["veto"][0],
        "fresh_kimchi_fx": (dates[5], dates[9], -1),
        "frozen_annual_rank7": (dates[6], dates[10], 1),
        "markov_transition_long": helper_rows["markov"][0],
        "rex_taker_low_range_position": helper_rows["taker"][0],
    }
    for sleeve, row in expected.items():
        observed = clocks[sleeve].iloc[0]
        assert (
            observed["entry_time"],
            observed["exit_time"],
            int(observed["side"]),
        ) == row
        assert len(clocks[sleeve]) == 1
    assert replay_calls == [
        ("fresh", 4, -1, (4, 10, 10)),
        ("rank7", 5, 1, (4, 10, 10)),
    ]


def test_strict_mdd_orders_favorable_before_adverse_and_uses_preentry_hwm() -> None:
    market = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:10:00Z",
        overrides={
            "2023-01-01T00:00:00Z": (100.0, 120.0, 80.0),
            "2023-01-01T00:10:00Z": (110.0, 110.0, 110.0),
        },
    )
    result = e.simulate_portfolio(
        market,
        _funding(),
        {
            "esdi": _clock(
                [("2023-01-01T00:00:00Z", "2023-01-01T00:10:00Z", 1)]
            )
        },
        {"esdi": 1.0},
        start="2023-01-01T00:00:00Z",
        end="2023-01-01T00:10:00Z",
        cost_rate=e.BASE_COST_RATE,
        synthetic=True,
    )

    entry_cost = 0.5 * e.BASE_COST_RATE
    favorable_equity = 1.0 - entry_cost + 0.5 * 0.20
    liquidation_cost = (0.5 / 100.0) * 80.0 * e.BASE_COST_RATE
    adverse_equity = (
        1.0 - entry_cost - 0.5 * 0.20 - liquidation_cost
    )
    expected = 1.0 - adverse_equity / favorable_equity
    assert result["strict_mdd"] == pytest.approx(expected)
    assert result["path"][0]["hwm"] == pytest.approx(favorable_equity)
    assert result["path"][0]["strict_lower_equity"] == pytest.approx(
        adverse_equity
    )


def test_short_strict_mdd_uses_low_for_hwm_high_for_adverse_and_liquidation() -> None:
    market = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:10:00Z",
        overrides={
            "2023-01-01T00:00:00Z": (100.0, 120.0, 80.0),
            "2023-01-01T00:10:00Z": (90.0, 90.0, 90.0),
        },
    )
    result = e.simulate_portfolio(
        market,
        _funding(),
        {
            "esdi": _clock(
                [("2023-01-01T00:00:00Z", "2023-01-01T00:10:00Z", -1)]
            )
        },
        {"esdi": 1.0},
        start="2023-01-01T00:00:00Z",
        end="2023-01-01T00:10:00Z",
        cost_rate=e.BASE_COST_RATE,
        synthetic=True,
    )

    quantity = 0.5 / 100.0
    entry_cost = 0.5 * e.BASE_COST_RATE
    favorable_equity = 1.0 - entry_cost + quantity * (100.0 - 80.0)
    liquidation_cost = quantity * 120.0 * e.BASE_COST_RATE
    adverse_equity = (
        1.0 - entry_cost - quantity * (120.0 - 100.0) - liquidation_cost
    )
    expected = 1.0 - adverse_equity / favorable_equity
    first_path = result["path"][0]
    assert first_path["net_signed_btc_quantity"] == pytest.approx(-quantity)
    assert first_path["hypothetical_liquidation_cost"] == pytest.approx(
        liquidation_cost
    )
    assert first_path["hwm"] == pytest.approx(favorable_equity)
    assert first_path["strict_lower_equity"] == pytest.approx(adverse_equity)
    assert result["strict_mdd"] == pytest.approx(expected)


def test_exact_per_notional_side_costs_and_funding_entry_inclusive_exit_exclusive() -> None:
    market = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:10:00Z",
        overrides={"2023-01-01T00:10:00Z": (110.0, 110.0, 110.0)},
    )
    funding = _funding(
        [
            ("2023-01-01T00:00:00Z", 0.010, 100.0),
            ("2023-01-01T00:05:00Z", -0.010, 110.0),
            ("2023-01-01T00:10:00Z", 9.999, 110.0),
        ]
    )
    result = e.simulate_portfolio(
        market,
        funding,
        {
            "esdi": _clock(
                [("2023-01-01T00:00:00Z", "2023-01-01T00:10:00Z", 1)]
            )
        },
        {"esdi": 1.0},
        start="2023-01-01T00:00:00Z",
        end="2023-01-01T00:10:00Z",
        cost_rate=e.BASE_COST_RATE,
        synthetic=True,
    )
    trade = result["trade_records"][0]

    quantity = 0.5 / 100.0
    assert trade["entry_cost"] == pytest.approx(quantity * 100 * 0.0006)
    assert trade["exit_cost"] == pytest.approx(quantity * 110 * 0.0006)
    assert trade["funding_cash"] == pytest.approx(
        -quantity * 0.010 * 100 + quantity * 0.010 * 110
    )
    assert trade["net_return_on_allocated_equity"] == pytest.approx(
        quantity * 10
        + trade["funding_cash"]
        - trade["entry_cost"]
        - trade["exit_cost"]
    )
    assert result["mean_gross_underlying_bp"] == pytest.approx(1_000.0)


def test_same_open_rollover_sizes_new_trade_after_prior_exit_cost() -> None:
    market = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:10:00Z",
        overrides={
            "2023-01-01T00:05:00Z": (110.0, 110.0, 110.0),
            "2023-01-01T00:10:00Z": (110.0, 110.0, 110.0),
        },
    )
    result = e.simulate_portfolio(
        market,
        _funding(),
        {
            "esdi": _clock(
                [
                    ("2023-01-01T00:00:00Z", "2023-01-01T00:05:00Z", 1),
                    ("2023-01-01T00:05:00Z", "2023-01-01T00:10:00Z", 1),
                ]
            )
        },
        {"esdi": 1.0},
        start="2023-01-01T00:00:00Z",
        end="2023-01-01T00:10:00Z",
        cost_rate=e.BASE_COST_RATE,
        synthetic=True,
    )
    first, second = result["trade_records"]
    rollover_equity = (
        1.0
        + 0.5 * 0.10
        - first["entry_cost"]
        - first["exit_cost"]
    )
    assert second["entry_cost"] == pytest.approx(
        rollover_equity * e.LEVERAGE * e.BASE_COST_RATE
    )


def test_full_calendar_cagr_includes_idle_time() -> None:
    clock = _clock(
        [("2023-01-01T00:00:00Z", "2023-01-01T00:05:00Z", 1)]
    )
    one_day = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-02T00:00:00Z",
        overrides={"2023-01-01T00:05:00Z": (102.0, 102.0, 102.0)},
    )
    two_days = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-03T00:00:00Z",
        overrides={"2023-01-01T00:05:00Z": (102.0, 102.0, 102.0)},
    )

    first = e.simulate_portfolio(
        one_day,
        _funding(),
        {"esdi": clock},
        {"esdi": 1.0},
        start="2023-01-01",
        end="2023-01-02",
        cost_rate=0.0,
        synthetic=True,
    )
    second = e.simulate_portfolio(
        two_days,
        _funding(),
        {"esdi": clock},
        {"esdi": 1.0},
        start="2023-01-01",
        end="2023-01-03",
        cost_rate=0.0,
        synthetic=True,
    )
    assert first["absolute_return"] == pytest.approx(second["absolute_return"])
    assert first["cagr"] > second["cagr"]
    assert second["cagr"] == pytest.approx(
        (1.0 + second["absolute_return"]) ** (365.25 / 2.0) - 1.0
    )
    assert e._calendar_years(*e.PERIODS["full"]) == 3.0
    assert e._full_calendar_cagr(1.21, 3.0) == pytest.approx(
        1.21 ** (1.0 / 3.0) - 1.0
    )


def test_calendar_month_clustered_signflip_is_seeded_and_month_not_week() -> None:
    trades = [
        {
            "entry_time": f"2023-{month:02d}-01T00:00:00Z",
            "net_return_on_allocated_equity": value,
        }
        for month, value in enumerate((0.03, 0.02, -0.005, 0.01), start=1)
    ]
    first = e.calendar_month_clustered_signflip(trades, seed=77, samples=5000)
    second = e.calendar_month_clustered_signflip(trades, seed=77, samples=5000)
    combined_same_month = e.calendar_month_clustered_signflip(
        [
            {
                "entry_time": "2023-01-01T00:00:00Z",
                "net_return_on_allocated_equity": 0.01,
            },
            {
                "entry_time": "2023-01-29T00:00:00Z",
                "net_return_on_allocated_equity": 0.02,
            },
        ],
        seed=77,
        samples=5000,
    )
    monte_carlo_trades = [
        {
            "entry_time": (
                pd.Timestamp("2021-01-01T00:00:00Z")
                + pd.DateOffset(months=index)
            ),
            "net_return_on_allocated_equity": (
                -0.02 if index % 3 == 0 else 0.03
            ),
        }
        for index in range(21)
    ]
    monte_first = e.calendar_month_clustered_signflip(
        monte_carlo_trades, seed=77, samples=5000
    )
    monte_second = e.calendar_month_clustered_signflip(
        monte_carlo_trades, seed=77, samples=5000
    )

    assert first == second
    assert first["cluster_count"] == 4
    assert first["method"] == "exact"
    assert first["p_value_one_sided"] == pytest.approx(2.0 / 16.0)
    assert combined_same_month["cluster_count"] == 1
    assert monte_first == monte_second
    assert monte_first["method"] == "monte_carlo"
    assert monte_first["cluster_count"] == 21


def test_same_gross_arithmetic_scales_every_sleeve_and_preserves_nine() -> None:
    treatment = e.same_gross_weights(0.75)
    scale = (9.0 - 0.75) / 9.0

    assert treatment["esdi"] == 0.75
    assert sum(treatment.values()) == pytest.approx(9.0)
    for sleeve, baseline in e.GROSS9_WEIGHTS.items():
        assert treatment[sleeve] == pytest.approx(baseline * scale)


def test_same_gross_identical_clocks_match_unscaled_gross9_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:10:00Z",
        overrides={
            "2023-01-01T00:00:00Z": (100.0, 101.0, 99.0),
            "2023-01-01T00:10:00Z": (101.0, 101.0, 101.0),
        },
    )
    clock = _clock(
        [("2023-01-01T00:00:00Z", "2023-01-01T00:10:00Z", 1)]
    )
    gross9 = {sleeve: clock.copy() for sleeve in e.GROSS9_SLEEVES}
    funding = _funding(
        [("2023-01-01T00:05:00Z", -0.001, 100.5)]
    )
    baseline = e.simulate_portfolio(
        market,
        funding,
        gross9,
        e.GROSS9_WEIGHTS,
        start="2023-01-01T00:00:00Z",
        end="2023-01-01T00:10:00Z",
        cost_rate=e.STRESS_COST_RATE,
        synthetic=True,
    )
    treatment = e.simulate_portfolio(
        market,
        funding,
        {**gross9, "esdi": clock.copy()},
        e.same_gross_weights(0.5),
        start="2023-01-01T00:00:00Z",
        end="2023-01-01T00:10:00Z",
        cost_rate=e.STRESS_COST_RATE,
        synthetic=True,
    )
    for key in (
        "absolute_return",
        "cagr",
        "strict_mdd",
        "cagr_to_strict_mdd",
        "final_equity",
    ):
        assert treatment[key] == pytest.approx(baseline[key])

    def fake_simulation(
        _market_frame: pd.DataFrame,
        _funding_frame: pd.DataFrame,
        _clocks: Mapping[str, pd.DataFrame],
        weights: Mapping[str, float],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        gross = sum(weights.values())
        return {
            "absolute_return": gross / 100.0,
            "cagr_to_strict_mdd": gross,
            "strict_mdd": 0.1,
            "liquidation_safe": True,
            "final_equity": 1.0 + gross / 100.0,
        }

    monkeypatch.setattr(e, "_simulate_portfolio", fake_simulation)
    row = e.evaluate_same_gross_weight(
        market,
        _funding([("2023-01-01T00:00:00Z", 0.0001, 100.0)]),
        gross9,
        clock,
        0.5,
        periods=e.SELECTION_PERIODS,
        synthetic=True,
    )

    for cost in ("base", "stress"):
        for period in e.SELECTION_PERIODS:
            metrics = row["periods"][period][cost]
            for key in ("absolute_return", "strict_mdd", "final_equity"):
                assert metrics["treatment"][key] == pytest.approx(
                    metrics["unscaled_gross9"][key]
                )


def test_rank_uses_maximum_minimum_improvement_then_lower_weight() -> None:
    def row(weight: float, improvement: float, passing: bool) -> dict[str, Any]:
        return {
            "candidate_weight": weight,
            "treatment_weights": e.same_gross_weights(weight),
            "baseline_weights": dict(e.GROSS9_WEIGHTS),
            "fresh_evaluation": True,
            "period_order": ["2023H2", "2024"],
            "periods": {
                period: {
                    cost: {
                        "treatment": {
                            "absolute_return": 0.5 if passing else -0.1,
                            "cagr_to_strict_mdd": 1.0 + improvement,
                            "strict_mdd": 0.09,
                            "liquidation_safe": True,
                        },
                        "unscaled_gross9": {
                            "absolute_return": 0.5,
                            "cagr_to_strict_mdd": 1.0,
                            "strict_mdd": 0.1,
                            "liquidation_safe": True,
                        },
                    }
                    for cost in ("base", "stress")
                }
                for period in ("2023H2", "2024")
            },
        }

    rows = [
        row(0.25, 0.10, True),
        row(0.50, 0.20, True),
        row(0.75, 0.20, True),
        row(1.00, 0.30, True),
    ]
    ranked = e.rank_same_gross_treatments(rows)

    assert [row["candidate_weight"] for row in ranked] == [1.0, 0.5, 0.75, 0.25]
    assert ranked[0]["rank"] == 1
    assert ranked[0]["frozen"] is True


def test_failed_maximum_improvement_rank_one_is_terminal_not_replaced() -> None:
    def row(weight: float, improvement: float, passing: bool) -> dict[str, Any]:
        return {
            "candidate_weight": weight,
            "treatment_weights": e.same_gross_weights(weight),
            "baseline_weights": dict(e.GROSS9_WEIGHTS),
            "fresh_evaluation": True,
            "period_order": ["2023H2", "2024"],
            "periods": {
                period: {
                    cost: {
                        "treatment": {
                            "absolute_return": 0.5 if passing else -0.1,
                            "cagr_to_strict_mdd": 1.0 + improvement,
                            "strict_mdd": 0.09,
                            "liquidation_safe": True,
                        },
                        "unscaled_gross9": {
                            "absolute_return": 0.5,
                            "cagr_to_strict_mdd": 1.0,
                            "strict_mdd": 0.1,
                            "liquidation_safe": True,
                        },
                    }
                    for cost in ("base", "stress")
                }
                for period in ("2023H2", "2024")
            },
        }

    rows = [
        row(0.25, 0.10, True),
        row(0.50, 0.20, True),
        row(0.75, 0.20, True),
        row(1.00, 0.30, False),
    ]
    ranked = e._rank_same_gross_treatments(
        rows,
        require_passing_freeze=False,
    )
    assert [item["candidate_weight"] for item in ranked] == [
        1.0,
        0.5,
        0.75,
        0.25,
    ]
    assert ranked[0]["passes"] is False
    assert ranked[0]["frozen"] is False
    with pytest.raises(RuntimeError, match="no passing"):
        e.rank_same_gross_treatments(rows)


def test_future_is_veto_only_and_cannot_rerank_or_change_weight() -> None:
    frozen = {
        "candidate_weight": 0.5,
        "minimum_improvement": 0.2,
        "passes": True,
        "rank": 1,
        "frozen": True,
    }
    result = e.future_veto(
        frozen,
        {
            "future25": {"candidate_weight": 0.5, "passes": True},
            "future26": {"candidate_weight": 0.5, "passes": False},
        },
        synthetic=True,
    )
    assert result == {
        "frozen_weight": 0.5,
        "checks": {"future25": True, "future26": False},
        "passes": False,
        "reranked": False,
    }

    with pytest.raises(RuntimeError, match="rerank or change weight"):
        e.future_veto(
            frozen,
            {
                "future25": {"candidate_weight": 0.75, "passes": True},
                "future26": {"candidate_weight": 0.5, "passes": True},
            },
            synthetic=True,
        )


def _standalone_stub(ratio: float, passes: bool) -> dict[str, Any]:
    row = {"metrics": {"cagr_to_strict_mdd": ratio}}
    return {"base": copy.deepcopy(row), "stress": copy.deepcopy(row), "passes": passes}


def test_primary_strict_superiority_and_control_disqualification() -> None:
    controls = {
        "base_fee_one_epoch_stale": _standalone_stub(2.5, True),
        "gas_utilization_only": _standalone_stub(3.0, True),
        "base_fee_no_tail": _standalone_stub(4.0, True),
        "exact_direction_flip": _standalone_stub(0.0, False),
        "deterministic_random_side": _standalone_stub(0.0, False),
        "constant_long": _standalone_stub(0.0, False),
        "constant_short": _standalone_stub(0.0, False),
        "one_bar_delayed_entry": _standalone_stub(4.0, True),
    }
    assert e.evaluate_primary_superiority(
        _standalone_stub(3.1, True), controls
    )["passes"]

    controls["constant_long"]["passes"] = True
    assert not e.evaluate_primary_superiority(
        _standalone_stub(3.1, True), controls
    )["passes"]
    controls["constant_long"]["passes"] = False
    assert not e.evaluate_primary_superiority(
        _standalone_stub(3.0, True), controls
    )["passes"]


def test_period_suite_evaluates_all_frozen_controls_and_both_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    primary = _clock([])
    primary.attrs["name"] = "primary"
    clocks: dict[str, pd.DataFrame] = {}
    for name in e.CONTROL_NAMES:
        frame = _clock([])
        frame.attrs["name"] = name
        clocks[name] = frame

    def fake_period(
        _market_frame: pd.DataFrame,
        _funding_frame: pd.DataFrame,
        clock: pd.DataFrame,
        *,
        start: Any,
        end: Any,
        synthetic: bool,
    ) -> dict[str, Any]:
        del start, end
        assert synthetic is True
        name = str(clock.attrs["name"])
        calls.append(name)
        disqualified = name in {
            "exact_direction_flip",
            "deterministic_random_side",
            "constant_long",
            "constant_short",
        }
        ratio = 10.0 if name == "primary" else 9.0
        return _standalone_stub(ratio, not disqualified)

    monkeypatch.setattr(e, "evaluate_standalone_period", fake_period)
    result = e.evaluate_standalone_period_with_controls(
        _market("2023-01-01T00:00:00Z", "2023-01-01T00:05:00Z"),
        _funding(),
        primary,
        clocks,
        start="2023-01-01",
        end="2023-01-02",
        synthetic=True,
    )

    assert calls == ["primary", *e.CONTROL_NAMES]
    assert tuple(result["controls"]) == e.CONTROL_NAMES
    assert result["passes"] is True


def test_standalone_gate_requires_every_frozen_condition() -> None:
    metrics = {
        "absolute_return": 0.01,
        "cagr_to_strict_mdd": 3.0,
        "strict_mdd": 0.15,
        "mean_gross_underlying_bp": 20.0,
        "calendar_month_clustered_signflip": {"p_value_one_sided": 0.10},
        "liquidation_safe": True,
    }
    assert all(e.standalone_gate_checks(metrics).values())
    metrics["absolute_return"] = 0.0
    assert not all(e.standalone_gate_checks(metrics).values())


def test_liquidation_unsafe_envelope_fails_closed() -> None:
    market = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:05:00Z",
        overrides={"2023-01-01T00:00:00Z": (100.0, 100.0, 1.0)},
    )
    clock = _clock(
        [("2023-01-01T00:00:00Z", "2023-01-01T00:05:00Z", 1)]
    )
    with pytest.raises(RuntimeError, match="liquidation safe"):
        e.simulate_portfolio(
            market,
            _funding(),
            {"levered": clock},
            {"levered": 3.0},
            start="2023-01-01T00:00:00Z",
            end="2023-01-01T00:05:00Z",
            cost_rate=0.0,
            synthetic=True,
        )


def test_write_once_result_is_canonical_and_refuses_mutation(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    payload = {"z": 1, "a": {"timestamp": pd.Timestamp("2023-01-01", tz="UTC")}}

    assert e.write_once_result(target, payload) == "created"
    assert e.write_once_result(target, payload) == "verified_existing"
    assert target.read_bytes() == (
        '{"a":{"timestamp":"2023-01-01T00:00:00+00:00"},"z":1}\n'
    ).encode()
    assert target.stat().st_mode & 0o777 == 0o444
    with pytest.raises(RuntimeError, match="already differs"):
        e.write_once_result(target, {"z": 2})


def test_cli_exposes_only_gross9_then_staged_economics_commands(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        e,
        "reconstruct_production_gross9_clocks",
        lambda: (
            calls.append("gross9")
            or {"manifest_hash": "a" * 64}
        ),
    )
    monkeypatch.setattr(
        e,
        "run_staged_economics",
        lambda: calls.append("economics") or {"passed": True},
    )
    assert e.main(["gross9-clocks"]) == 0
    gross9_output = json.loads(capsys.readouterr().out)
    assert gross9_output == {
        "manifest_hash": "a" * 64,
        "output": str(e.GROSS9_CLOCK_ARTIFACT),
        "status": "gross9_clocks_reconstructed",
    }
    assert e.main(["economics"]) == 0
    assert json.loads(capsys.readouterr().out) == {"passed": True}
    assert calls == ["gross9", "economics"]


def test_evaluator_self_test_gate_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def passing(command: list[str], **_kwargs: Any) -> Result:
        calls.append(command)
        return Result(0)

    monkeypatch.setattr(e.subprocess, "run", passing)
    evidence = e._run_evaluator_synthetic_tests()
    assert evidence == {
        "command": [
            "python",
            "-m",
            "pytest",
            "-q",
            str(e.EVALUATOR_TEST_PATH),
        ],
        "passed": True,
    }
    assert calls[0][-1] == str(e.EVALUATOR_TEST_PATH)

    monkeypatch.setattr(
        e.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(1),
    )
    with pytest.raises(RuntimeError, match="synthetic test suite failed"):
        e._run_evaluator_synthetic_tests()


def test_injected_validation_reconstructors_and_frames_are_synthetic_only() -> None:
    registration, context = _registration_and_context()
    market = _market(
        "2023-01-01T00:00:00Z", "2023-01-01T00:05:00Z"
    )
    funding = _funding()
    empty_clock = _clock([])
    with pytest.raises(RuntimeError, match="synthetic-only"):
        e.validate_frozen_contract(registration, context=context)

    reconstructors = {
        sleeve: (lambda _binding, _receipt: _clock([]))
        for sleeve in e.GROSS9_SLEEVES
    }
    with pytest.raises(RuntimeError, match="synthetic-only"):
        e.reconstruct_gross9_sleeve_clocks(
            registration,
            context=context,
            reconstructors=reconstructors,
        )

    with pytest.raises(RuntimeError, match="synthetic-only"):
        e.simulate_portfolio(
            market,
            funding,
            {"esdi": empty_clock},
            {"esdi": 1.0},
            start="2023-01-01T00:00:00Z",
            end="2023-01-01T00:05:00Z",
            cost_rate=0.0,
        )
    with pytest.raises(RuntimeError, match="synthetic-only"):
        e.evaluate_standalone_period(
            market,
            funding,
            empty_clock,
            start="2023-01-01T00:00:00Z",
            end="2023-01-01T00:05:00Z",
        )
    with pytest.raises(RuntimeError, match="synthetic-only"):
        e.evaluate_same_gross_weight(
            market,
            funding,
            {
                sleeve: empty_clock.copy()
                for sleeve in e.GROSS9_SLEEVES
            },
            empty_clock,
            0.25,
            periods=e.SELECTION_PERIODS,
        )
    with pytest.raises(RuntimeError, match="synthetic-only"):
        e.build_gross9_clock_artifact(
            {
                sleeve: empty_clock.copy()
                for sleeve in e.GROSS9_SLEEVES
            },
            source_support_binding={
                "path": "synthetic.json",
                "sha256": "a" * 64,
                "manifest_hash": "b" * 64,
            },
            authority=registration["gross9"]["authority"],
        )
    with pytest.raises(RuntimeError, match="synthetic-only"):
        e.run_staged_economics(
            novelty={"passed": True},
            stage_loader=lambda _stage, _cutoff: {},
            stage_evaluator=lambda _stage, _inputs, _state: {
                "passed": False
            },
        )


def test_validation_covers_all_77_repository_bindings_and_pre2025_anchor() -> None:
    registration, context = _registration_and_context()
    assert len(context.repository_sha256) == 77
    assert len(context.repository_git_blobs) == 77

    missing_sha = dict(context.repository_sha256)
    missing_sha.pop(next(iter(missing_sha)))
    with pytest.raises(RuntimeError, match="77 repository"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{**context.__dict__, "repository_sha256": missing_sha}
            ),
            synthetic=True,
        )

    missing_blob = dict(context.repository_git_blobs)
    missing_blob.pop(next(iter(missing_blob)))
    with pytest.raises(RuntimeError, match="77 repository"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{**context.__dict__, "repository_git_blobs": missing_blob}
            ),
            synthetic=True,
        )

    anchor_path = registration["gross9"]["authority"]["pre2025_anchor"]["path"]
    assert anchor_path in context.file_hashes
    bad_authority = dict(context.file_hashes)
    bad_authority[anchor_path] = "0" * 64
    with pytest.raises(RuntimeError, match="dependency SHA drift"):
        e.validate_frozen_contract(
            registration,
            context=e.ValidationContext(
                **{**context.__dict__, "file_hashes": bad_authority}
            ),
            synthetic=True,
        )

    drifted_registration = copy.deepcopy(registration)
    drifted_registration["gross9"]["authority"]["pre2025_anchor"][
        "metadata_only_until_economic_stage"
    ] = False
    with pytest.raises(RuntimeError, match="authority inventory"):
        e.validate_frozen_contract(
            drifted_registration,
            context=context,
            synthetic=True,
        )


def test_net_signed_quantity_offsets_shared_bar_mdd() -> None:
    market = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:05:00Z",
        overrides={"2023-01-01T00:00:00Z": (100.0, 150.0, 50.0)},
    )
    clock = _clock(
        [("2023-01-01T00:00:00Z", "2023-01-01T00:05:00Z", 1)]
    )
    short_clock = clock.copy()
    short_clock["side"] = -1
    result = e.simulate_portfolio(
        market,
        _funding(),
        {"long": clock, "short": short_clock},
        {"long": 1.0, "short": 1.0},
        start="2023-01-01T00:00:00Z",
        end="2023-01-01T00:05:00Z",
        cost_rate=0.0,
        synthetic=True,
    )

    assert result["strict_mdd"] == 0.0
    assert result["path"][0]["net_signed_btc_quantity"] == pytest.approx(0.0)
    assert result["liquidation_safe"] is True


def test_exit_cost_is_applied_after_shared_bar_favorable_envelope() -> None:
    market = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:10:00Z",
        overrides={"2023-01-01T00:05:00Z": (100.0, 120.0, 100.0)},
    )
    result = e.simulate_portfolio(
        market,
        _funding(),
        {
            "exiting": _clock(
                [("2023-01-01T00:00:00Z", "2023-01-01T00:05:00Z", 1)]
            ),
            "held": _clock(
                [("2023-01-01T00:00:00Z", "2023-01-01T00:10:00Z", 1)]
            ),
        },
        {"exiting": 1.0, "held": 1.0},
        start="2023-01-01T00:00:00Z",
        end="2023-01-01T00:10:00Z",
        cost_rate=0.01,
        synthetic=True,
    )

    # Both 0.5-notional entry costs precede the bar.  The exiting 0.5-notional
    # side cost is charged only after the held sleeve can establish its high.
    assert result["path"][1]["hwm"] == pytest.approx(0.99 + 0.5 * 0.20)


def test_split_crossers_are_skipped_and_counted_not_rejected() -> None:
    market = _market(
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:15:00Z",
        overrides={"2023-01-01T00:15:00Z": (101.0, 101.0, 101.0)},
    )
    clock = _clock(
        [
            ("2022-12-31T23:55:00Z", "2023-01-01T00:05:00Z", 1),
            ("2023-01-01T00:10:00Z", "2023-01-01T00:15:00Z", 1),
        ]
    )
    result = e.simulate_portfolio(
        market,
        _funding(),
        {"esdi": clock},
        {"esdi": 1.0},
        start="2023-01-01T00:00:00Z",
        end="2023-01-01T00:15:00Z",
        cost_rate=0.0,
        synthetic=True,
    )

    assert result["trades"] == 1
    assert result["skipped_split_crossers"] == {"esdi": 1}
    assert result["split_crossers_truncated"] == 0


def test_same_gross_rejects_nonexact_or_reordered_selection_periods() -> None:
    exact = {
        "2023H2": e.PERIODS["2023H2"],
        "2024": e.PERIODS["2024"],
    }
    reversed_periods = {
        "2024": e.PERIODS["2024"],
        "2023H2": e.PERIODS["2023H2"],
    }
    gross9 = {sleeve: _clock([]) for sleeve in e.GROSS9_SLEEVES}
    with pytest.raises(RuntimeError, match="exact ordered"):
        e.evaluate_same_gross_weight(
            _market("2023-01-01T00:00:00Z", "2023-01-01T00:05:00Z"),
            _funding(),
            gross9,
            _clock([]),
            0.25,
            periods=reversed_periods,
            synthetic=True,
        )
    drifted = dict(exact)
    drifted["2024"] = (
        e.PERIODS["2024"][0],
        e.PERIODS["2024"][1] - pd.Timedelta(minutes=5),
    )
    with pytest.raises(RuntimeError, match="exact ordered"):
        e.evaluate_same_gross_weight(
            _market("2023-01-01T00:00:00Z", "2023-01-01T00:05:00Z"),
            _funding(),
            gross9,
            _clock([]),
            0.25,
            periods=drifted,
            synthetic=True,
        )


def test_ranking_rejects_forged_fields_and_failed_freeze() -> None:
    rows = [
        {
            "candidate_weight": weight,
            "treatment_weights": e.same_gross_weights(weight),
            "baseline_weights": dict(e.GROSS9_WEIGHTS),
            "fresh_evaluation": True,
            "period_order": ["2023H2", "2024"],
            "periods": {
                period: {
                    cost: {
                        "treatment": {
                            "absolute_return": 0.5,
                            "cagr_to_strict_mdd": 1.0,
                            "strict_mdd": 0.2,
                            "liquidation_safe": True,
                        },
                        "unscaled_gross9": {
                            "absolute_return": 0.5,
                            "cagr_to_strict_mdd": 2.0,
                            "strict_mdd": 0.1,
                            "liquidation_safe": True,
                        },
                    }
                    for cost in ("base", "stress")
                }
                for period in ("2023H2", "2024")
            },
            "minimum_improvement": 999.0,
            "passes": True,
        }
        for weight in e.CANDIDATE_WEIGHTS
    ]
    with pytest.raises(RuntimeError, match="derived fields"):
        e.rank_same_gross_treatments(rows)

    for row in rows:
        row.pop("minimum_improvement")
        row.pop("passes")
    with pytest.raises(RuntimeError, match="no passing"):
        e.rank_same_gross_treatments(rows)

    forged_weights = copy.deepcopy(rows)
    forged_weights[0]["treatment_weights"]["esdi"] = 1.0
    with pytest.raises(RuntimeError, match="weight binding"):
        e.rank_same_gross_treatments(forged_weights)


def test_future26_requires_hash_bound_future25_pass() -> None:
    frozen = {
        "candidate_weight": 0.5,
        "rank": 1,
        "frozen": True,
        "passes": True,
        "selection_receipt_sha256": "1" * 64,
    }
    future_row = {"candidate_weight": 0.5, "passes": True}
    with pytest.raises(RuntimeError, match="future25"):
        e.authorize_future_period(
            frozen,
            "future26",
            future_row,
            synthetic=True,
        )

    receipt_core = {
        "protocol_version": e.ECONOMIC_RECEIPT_PROTOCOL,
        "stage": "future25",
        "passed": True,
        "frozen_weight": 0.5,
        "selection_receipt_sha256": "1" * 64,
    }
    receipt = {
        **receipt_core,
        "manifest_hash": e.canonical_hash(receipt_core),
    }
    authorized = e.authorize_future_period(
        frozen,
        "future26",
        future_row,
        future25_receipt=receipt,
        synthetic=True,
    )
    assert authorized["passed"] is True
    assert authorized["reranked"] is False


def test_gross9_clock_artifact_self_hashes_and_rejects_tamper(tmp_path: Path) -> None:
    from training import (
        evaluate_ethereum_settlement_demand_impulse_novelty as novelty,
    )

    registration, _ = _registration_and_context()
    clocks = {
        sleeve: _clock(
            [("2023-06-01T00:00:00Z", "2023-06-01T00:05:00Z", 1)]
        )
        for sleeve in e.GROSS9_SLEEVES
    }
    support = {
        "path": "synthetic/source-support.json",
        "sha256": "2" * 64,
        "manifest_hash": "3" * 64,
    }
    artifact = e.build_gross9_clock_artifact(
        clocks,
        source_support_binding=support,
        authority=registration["gross9"]["authority"],
        synthetic=True,
    )
    output = tmp_path / "gross9-clocks.json"
    assert e.write_once_result(output, artifact) == "created"
    loaded = e.load_gross9_clock_artifact(
        output,
        source_support_binding=support,
        authority=registration["gross9"]["authority"],
        synthetic=True,
    )
    assert loaded["protocol_version"] == (
        "ethereum_settlement_demand_impulse_gross9_clocks_v1"
    )
    assert artifact["frozen_contract_validation"] == (
        novelty.gross9_frozen_contract_validation(registration)
    )

    malformed_support = {**support, "sha256": "z" * 64}
    with pytest.raises(RuntimeError, match="binding is malformed"):
        e.build_gross9_clock_artifact(
            clocks,
            source_support_binding=malformed_support,
            authority=registration["gross9"]["authority"],
            synthetic=True,
        )

    tampered = copy.deepcopy(artifact)
    tampered["clocks"][e.GROSS9_SLEEVES[0]]["intervals"][0]["side"] = "SHORT"
    tampered_output = tmp_path / "tampered-gross9-clocks.json"
    tampered_output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest|sleeve hash"):
        e.load_gross9_clock_artifact(
            tampered_output,
            source_support_binding=support,
            authority=registration["gross9"]["authority"],
            synthetic=True,
        )


def test_production_reconstruction_checks_support_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_called = False

    def reject_support(_path: Any) -> dict[str, Any]:
        raise RuntimeError("source support did not pass")

    def forbidden_runtime() -> dict[str, pd.DataFrame]:
        nonlocal runtime_called
        runtime_called = True
        return {}

    monkeypatch.setattr(e, "_load_passed_source_support", reject_support)
    monkeypatch.setattr(e, "_reconstruct_gross9_runtime_clocks", forbidden_runtime)
    with pytest.raises(RuntimeError, match="source support"):
        e.reconstruct_production_gross9_clocks()
    assert runtime_called is False


def test_gross9_attempt_claim_precedes_frozen_inputs_and_forbids_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_path = tmp_path / e.SOURCE_SUPPORT_ARTIFACT
    support_path.parent.mkdir(parents=True, exist_ok=True)
    support_path.write_bytes(b"immutable-support")
    support = {"manifest_hash": "a" * 64}
    calls: list[str] = []

    monkeypatch.setattr(e, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        e,
        "_load_passed_source_support",
        lambda _path: support,
    )
    monkeypatch.setattr(
        e,
        "_validate_evaluator_source_identity",
        lambda: {"manifest_hash": "b" * 64},
    )

    def fail_after_claim() -> dict[str, Any]:
        calls.append("frozen")
        assert (tmp_path / e.GROSS9_ATTEMPT_CLAIM).is_file()
        raise RuntimeError("synthetic frozen-contract failure")

    monkeypatch.setattr(e, "load_bound_preregistration", fail_after_claim)
    monkeypatch.setattr(
        e,
        "_reconstruct_gross9_runtime_clocks",
        lambda: (_ for _ in ()).throw(
            AssertionError("Gross9 rows must remain unopened")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic frozen-contract failure"):
        e.reconstruct_production_gross9_clocks()
    assert calls == ["frozen"]
    assert (tmp_path / e.GROSS9_ATTEMPT_CLAIM).is_file()
    assert not (tmp_path / e.GROSS9_CLOCK_ARTIFACT).exists()

    with pytest.raises(RuntimeError, match="lacks its completion artifact"):
        e.reconstruct_production_gross9_clocks()
    assert calls == ["frozen"]


def test_prefix_loaders_stop_before_future_numeric_rows_and_keep_settlement_mark(
    tmp_path: Path,
) -> None:
    market = tmp_path / "market.csv"
    market.write_text(
        "date,open,high,low\n"
        "2023-01-01T00:00:00Z,100,101,99\n"
        "2023-01-01T00:05:00Z,101,102,100\n"
        "2023-01-01T00:10:00Z,DO_NOT_PARSE,DO_NOT_PARSE,DO_NOT_PARSE\n",
        encoding="utf-8",
    )
    funding = tmp_path / "funding.csv"
    funding.write_text(
        "date,funding_rate,mark_price\n"
        "2023-01-01T00:00:00Z,0.001,100\n"
        "2023-01-01T00:05:00Z,DO_NOT_PARSE,DO_NOT_PARSE\n",
        encoding="utf-8",
    )

    market_frame = e._load_market_prefix(
        market, pd.Timestamp("2023-01-01T00:05:00Z")
    )
    funding_frame = e._load_funding_prefix(
        funding, pd.Timestamp("2023-01-01T00:05:00Z")
    )
    assert market_frame["open"].tolist() == [100, 101]
    assert funding_frame["funding_rate"].tolist() == [0.001]
    assert funding_frame["settlement_mark"].tolist() == [100]


def _novelty_pass() -> dict[str, Any]:
    core = {
        "protocol_version": "ethereum_settlement_demand_impulse_novelty_v1",
        "policy_id": e.POLICY_ID,
        "passed": True,
    }
    return {**core, "manifest_hash": e.canonical_hash(core)}


def test_exact_production_novelty_schema_rejects_synthetic_alias() -> None:
    with pytest.raises(RuntimeError, match="exact schema"):
        e._validate_passed_novelty(_novelty_pass(), exact=True)


def test_exact_production_novelty_schema_records_truthful_clock_evidence() -> None:
    core = {
        "protocol_version": e.NOVELTY_PROTOCOL_VERSION,
        "policy_id": e.POLICY_ID,
        "preregistration": {},
        "attempt_claim": {
            "path": (
                "results/"
                "ethereum_settlement_demand_impulse_"
                "novelty_attempt_claim_2026-07-30.json"
            ),
            "sha256": "a" * 64,
            "claim_hash": "b" * 64,
        },
        "source_support": {},
        "gross9_clock_artifact": {},
        "registry_artifacts": 0,
        "registry_comparator_groups": 0,
        "novelty": {
            "prior_source_comparators": [],
            "gross9_sleeves": [
                {
                    "sleeve": sleeve,
                    "weight": e.GROSS9_WEIGHTS[sleeve],
                    "checks": {"structural_novelty": True},
                    "passed": True,
                }
                for sleeve in e.GROSS9_SLEEVES
            ],
            "passed": True,
            "terminal": False,
            "failed_checks": [],
        },
        "evidence_boundary": dict(e.NOVELTY_EVIDENCE_BOUNDARY),
    }
    payload = {**core, "manifest_hash": e.canonical_hash(core)}
    e._validate_passed_novelty(payload, exact=True)

    drifted_core = copy.deepcopy(core)
    drifted_core["evidence_boundary"][
        "future_rows_used_for_economic_weight_ranking"
    ] = True
    drifted = {
        **drifted_core,
        "manifest_hash": e.canonical_hash(drifted_core),
    }
    with pytest.raises(RuntimeError, match="evidence boundary"):
        e._validate_passed_novelty(drifted, exact=True)


def _selection_stage_pass(weight: float = 0.5) -> dict[str, Any]:
    ranked_weights = [
        weight,
        *[candidate for candidate in e.CANDIDATE_WEIGHTS if candidate != weight],
    ]
    return {
        "passed": True,
        "frozen_weight": weight,
        "frozen_rank": 1,
        "ranking": [
            {
                "candidate_weight": candidate,
                "rank": rank,
                "frozen": rank == 1,
                "passes": True,
            }
            for rank, candidate in enumerate(ranked_weights, start=1)
        ],
    }


def test_staged_runner_stops_before_later_loader_after_failure(tmp_path: Path) -> None:
    calls: list[str] = []

    def loader(stage: str, cutoff: pd.Timestamp) -> dict[str, Any]:
        calls.append(stage)
        return {"stage": stage, "cutoff": cutoff}

    def evaluator(
        stage: str, _inputs: Mapping[str, Any], _state: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {"passed": stage != "2024"}

    report = e.run_staged_economics(
        synthetic=True,
        novelty=_novelty_pass(),
        stage_loader=loader,
        stage_evaluator=evaluator,
        receipt_root=tmp_path,
    )
    assert calls == ["2023H2", "2024"]
    assert report["stopped_at"] == "2024"
    assert not (tmp_path / e.STAGE_RECEIPT_NAMES["selection"]).exists()
    assert not (tmp_path / e.STAGE_RECEIPT_NAMES["future25"]).exists()


def test_synthetic_runner_cannot_write_canonical_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(e, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="cannot use canonical results"):
        e.run_staged_economics(
            synthetic=True,
            novelty=_novelty_pass(),
            stage_loader=lambda _stage, _cutoff: {},
            stage_evaluator=lambda _stage, _inputs, _state: {
                "passed": False
            },
            receipt_root=tmp_path / "results",
        )


def test_staged_runner_exact_end_to_end_sequence_and_receipts(tmp_path: Path) -> None:
    calls: list[tuple[str, pd.Timestamp]] = []

    def loader(stage: str, cutoff: pd.Timestamp) -> dict[str, Any]:
        calls.append((stage, cutoff))
        return {"stage": stage}

    def evaluator(
        stage: str, _inputs: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"passed": True}
        if stage == "same_gross":
            return _selection_stage_pass()
        elif stage in {"future25", "future26", "full"}:
            row["frozen_weight"] = state["frozen_weight"]
        return row

    report = e.run_staged_economics(
        synthetic=True,
        novelty=_novelty_pass(),
        stage_loader=loader,
        stage_evaluator=evaluator,
        receipt_root=tmp_path,
    )
    assert [stage for stage, _ in calls] == list(e.ECONOMIC_STAGE_ORDER)
    assert report["passed"] is True
    assert report["frozen_weight"] == 0.5
    for stage in e.ECONOMIC_STAGE_ORDER:
        receipt_path = tmp_path / e.STAGE_RECEIPT_NAMES[stage]
        payload = json.loads(receipt_path.read_text())
        core = {key: value for key, value in payload.items() if key != "manifest_hash"}
        assert payload["manifest_hash"] == e.canonical_hash(core)


def test_future25_veto_never_calls_future26_loader(tmp_path: Path) -> None:
    calls: list[str] = []

    def loader(stage: str, _cutoff: pd.Timestamp) -> dict[str, Any]:
        calls.append(stage)
        return {"stage": stage}

    def evaluator(
        stage: str, _inputs: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        if stage == "same_gross":
            return _selection_stage_pass()
        if stage == "future25":
            return {
                "passed": False,
                "frozen_weight": state["frozen_weight"],
            }
        return {"passed": True}

    report = e.run_staged_economics(
        synthetic=True,
        novelty=_novelty_pass(),
        stage_loader=loader,
        stage_evaluator=evaluator,
        receipt_root=tmp_path,
    )
    assert report["stopped_at"] == "future25"
    assert "future26" not in calls
    assert "full" not in calls


def test_staged_runner_rejects_failed_or_non_rank1_freeze_before_future_loader(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def loader(stage: str, _cutoff: pd.Timestamp) -> dict[str, Any]:
        calls.append(stage)
        return {"stage": stage}

    def evaluator(
        stage: str, _inputs: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        if stage == "same_gross":
            forged = _selection_stage_pass()
            forged["frozen_rank"] = 2
            forged["ranking"][0]["passes"] = False
            return forged
        if stage in {"future25", "future26"}:
            return {
                "passed": True,
                "frozen_weight": state["frozen_weight"],
            }
        return {"passed": True}

    with pytest.raises(RuntimeError, match="passed frozen rank one"):
        e.run_staged_economics(
            synthetic=True,
            novelty=_novelty_pass(),
            stage_loader=loader,
            stage_evaluator=evaluator,
            receipt_root=tmp_path,
        )
    assert calls == ["2023H2", "2024", "selection", "same_gross"]


def test_production_stage_evaluator_normalizes_pass_and_stitches_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standalone_calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    same_gross_calls: list[tuple[float, pd.Timestamp, pd.Timestamp]] = []

    def standalone(
        _market_frame: pd.DataFrame,
        _funding_frame: pd.DataFrame,
        _primary: pd.DataFrame,
        _controls: Mapping[str, pd.DataFrame],
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, Any]:
        standalone_calls.append((start, end))
        return {"passes": True, "kind": "standalone"}

    def same_gross(
        _market_frame: pd.DataFrame,
        _funding_frame: pd.DataFrame,
        _gross9: Mapping[str, pd.DataFrame],
        _primary: pd.DataFrame,
        weight: float,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, Any]:
        same_gross_calls.append((weight, start, end))
        return {"passes": True, "kind": "same_gross"}

    monkeypatch.setattr(
        e, "_evaluate_standalone_period_with_controls", standalone
    )
    monkeypatch.setattr(e, "_evaluate_same_gross_future_period", same_gross)
    inputs = {
        "market": _market(
            "2023-01-01T00:00:00Z", "2023-01-01T00:05:00Z"
        ),
        "funding": _funding(),
        "primary_clock": _clock([]),
        "control_clocks": {
            name: _clock([]) for name in e.CONTROL_NAMES
        },
        "gross9_clocks": {
            name: _clock([]) for name in e.GROSS9_SLEEVES
        },
    }

    selection = e._production_stage_evaluator("2023H2", inputs, {})
    assert selection["passed"] is True
    assert selection["standalone"]["passes"] is True

    full = e._production_stage_evaluator(
        "full", inputs, {"frozen_weight": 0.5}
    )
    assert full["passed"] is True
    assert full["standalone"]["passes"] is True
    assert full["same_gross"]["passes"] is True
    assert standalone_calls == [e.PERIODS["2023H2"], e.PERIODS["full"]]
    assert same_gross_calls == [(0.5, *e.PERIODS["full"])]


def test_production_runner_revalidates_prerequisites_before_first_row_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def novelty_loader(_path: Path) -> dict[str, Any]:
        calls.append("novelty")
        return _novelty_pass()

    def prerequisites() -> dict[str, Any]:
        calls.append("prerequisites")
        return {
            "frozen_contract_validation_hash": "a" * 64,
            "evaluator_source": {"manifest_hash": "c" * 64},
        }

    def loader(
        stage: str,
        _cutoff: pd.Timestamp,
        *,
        novelty: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert novelty["passed"] is True
        calls.append(f"loader:{stage}")
        return {}

    def evaluator(
        stage: str,
        _inputs: Mapping[str, Any],
        _state: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {"passed": stage != "2023H2"}

    monkeypatch.setattr(e, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(e, "_load_passed_novelty", novelty_loader)
    monkeypatch.setattr(
        e, "_validate_production_economic_prerequisites", prerequisites
    )
    monkeypatch.setattr(e, "_production_stage_loader", loader)
    monkeypatch.setattr(e, "_production_stage_evaluator", evaluator)
    monkeypatch.setattr(
        e,
        "_validate_pre2025_anchor",
        lambda _: (_ for _ in ()).throw(
            AssertionError("2024 anchor must remain unopened")
        ),
    )

    report = e.run_staged_economics(receipt_root=tmp_path / "results")
    assert report["stopped_at"] == "2023H2"
    assert calls == ["novelty", "prerequisites", "loader:2023H2"]


def test_production_stage_claim_precedes_rows_and_forbids_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    calls: list[str] = []
    monkeypatch.setattr(e, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(e, "_load_passed_novelty", lambda _path: _novelty_pass())
    monkeypatch.setattr(
        e,
        "_validate_production_economic_prerequisites",
        lambda: {
            "frozen_contract_validation_hash": "a" * 64,
            "evaluator_source": {"manifest_hash": "b" * 64},
        },
    )

    def fail_loader(
        stage: str,
        _cutoff: pd.Timestamp,
        *,
        novelty: Mapping[str, Any],
    ) -> dict[str, Any]:
        calls.append(stage)
        assert novelty["passed"] is True
        assert (
            results / e.STAGE_ATTEMPT_CLAIM_NAMES[stage]
        ).is_file()
        raise RuntimeError("synthetic stage-row failure")

    monkeypatch.setattr(e, "_production_stage_loader", fail_loader)
    with pytest.raises(RuntimeError, match="synthetic stage-row failure"):
        e.run_staged_economics(receipt_root=results)

    assert calls == ["2023H2"]
    assert (
        results / e.STAGE_ATTEMPT_CLAIM_NAMES["2023H2"]
    ).is_file()
    assert not (results / e.STAGE_RECEIPT_NAMES["2023H2"]).exists()
    with pytest.raises(RuntimeError, match="claim/completion is incomplete"):
        e.run_staged_economics(receipt_root=results)
    assert calls == ["2023H2"]


def test_completed_production_stage_resume_skips_rows_and_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    calls: list[str] = []
    monkeypatch.setattr(e, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(e, "_load_passed_novelty", lambda _path: _novelty_pass())
    monkeypatch.setattr(
        e,
        "_validate_production_economic_prerequisites",
        lambda: {
            "frozen_contract_validation_hash": "a" * 64,
            "evaluator_source": {"manifest_hash": "b" * 64},
        },
    )

    def loader(
        stage: str,
        _cutoff: pd.Timestamp,
        *,
        novelty: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert novelty["passed"] is True
        calls.append(f"loader:{stage}")
        return {}

    def evaluator(
        stage: str,
        _inputs: Mapping[str, Any],
        _state: Mapping[str, Any],
    ) -> dict[str, Any]:
        calls.append(f"evaluator:{stage}")
        return {"passed": stage != "2024"}

    monkeypatch.setattr(e, "_production_stage_loader", loader)
    monkeypatch.setattr(e, "_production_stage_evaluator", evaluator)
    first = e.run_staged_economics(receipt_root=results)
    assert first["stopped_at"] == "2024"
    assert calls == [
        "loader:2023H2",
        "evaluator:2023H2",
        "loader:2024",
        "evaluator:2024",
    ]

    calls.clear()
    second = e.run_staged_economics(receipt_root=results)
    assert second["stopped_at"] == "2024"
    assert calls == []


def test_nonreproducing_novelty_stops_before_economic_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training import (
        evaluate_ethereum_settlement_demand_impulse_novelty as novelty_module,
    )

    called = False
    monkeypatch.setattr(e, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        novelty_module,
        "load_reproduced_novelty_for_economics",
        lambda _path: (_ for _ in ()).throw(
            novelty_module.NoveltyTerminalError(
                "self-hashed novelty did not reproduce"
            )
        ),
    )

    def forbidden_loader(
        _stage: str,
        _cutoff: pd.Timestamp,
        *,
        novelty: Mapping[str, Any],
    ) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(e, "_production_stage_loader", forbidden_loader)
    with pytest.raises(RuntimeError, match="authenticate and reproduce"):
        e.run_staged_economics(receipt_root=tmp_path / "results")
    assert called is False


def test_production_anchor_opens_only_after_all_pre_same_gross_pass_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    results = tmp_path / "results"

    monkeypatch.setattr(e, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(e, "_load_passed_novelty", lambda _path: _novelty_pass())
    monkeypatch.setattr(
        e,
        "_validate_production_economic_prerequisites",
        lambda: {
            "frozen_contract_validation_hash": "a" * 64,
            "evaluator_source": {"manifest_hash": "c" * 64},
        },
    )
    monkeypatch.setattr(
        e,
        "load_bound_preregistration",
        lambda: {"frozen": "registration"},
    )

    def anchor(registration: Mapping[str, Any]) -> dict[str, Any]:
        assert registration == {"frozen": "registration"}
        calls.append("anchor")
        assert all(
            (results / e.STAGE_RECEIPT_NAMES[stage]).is_file()
            for stage in ("2023H2", "2024", "selection")
        )
        assert not (results / e.STAGE_RECEIPT_NAMES["same_gross"]).exists()
        return {"sha256": "d" * 64}

    def loader(
        stage: str,
        _cutoff: pd.Timestamp,
        *,
        novelty: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert novelty["passed"] is True
        calls.append(f"loader:{stage}")
        return {}

    def evaluator(
        stage: str,
        _inputs: Mapping[str, Any],
        _state: Mapping[str, Any],
    ) -> dict[str, Any]:
        calls.append(f"evaluator:{stage}")
        return {"passed": stage != "same_gross"}

    monkeypatch.setattr(e, "_validate_pre2025_anchor", anchor)
    monkeypatch.setattr(e, "_production_stage_loader", loader)
    monkeypatch.setattr(e, "_production_stage_evaluator", evaluator)

    report = e.run_staged_economics(receipt_root=results)
    assert report["stopped_at"] == "same_gross"
    assert calls == [
        "loader:2023H2",
        "evaluator:2023H2",
        "loader:2024",
        "evaluator:2024",
        "loader:selection",
        "evaluator:selection",
        "anchor",
        "loader:same_gross",
        "evaluator:same_gross",
    ]
    same_gross = json.loads(
        (results / e.STAGE_RECEIPT_NAMES["same_gross"]).read_text()
    )
    assert same_gross["pre2025_anchor_sha256"] == "d" * 64
