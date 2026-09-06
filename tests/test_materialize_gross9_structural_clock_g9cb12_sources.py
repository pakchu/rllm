from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import stat
import subprocess
from dataclasses import fields, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import pytest

from training import materialize_gross9_structural_clock_g9cb12_sources as subject


A12 = "a533ec5ec6bb01d0eeed8ab66a37a3a10f1dba5d"
T11 = "87c9d32df28f4b8c157d78e2d88145d6bfbb92c0"
S11 = "646fccbf6568bcf39fab12a47873f72da880ca01"

G11_SENTINEL = (
    "results/gross9_structural_clock_bundle_g9cb11_"
    "source_support_attempt_consumed_2026-07-31.json"
)
G11_ABSENCES = (
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb11_complete.csv.gz",
    "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb11_complete.csv.gz",
    "configs/shadow/gross9_structural_clock_bundle_g9cb11_sources_2026-07-31.json",
    "results/gross9_structural_clock_bundle_g9cb11_source_support_2026-07-31.json",
)
G12_OUTPUTS = (
    "results/gross9_structural_clock_bundle_g9cb12_source_support_attempt_consumed_2026-07-31.json",
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb12_complete.csv.gz",
    "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb12_complete.csv.gz",
    "configs/shadow/gross9_structural_clock_bundle_g9cb12_sources_2026-07-31.json",
    "results/gross9_structural_clock_bundle_g9cb12_source_support_2026-07-31.json",
)
S12_FILES = (
    "training/materialize_gross9_structural_clock_g9cb12_sources.py",
    "tests/test_materialize_gross9_structural_clock_g9cb12_sources.py",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def _canonical_json_bytes(value: object, *, trailing_lf: bool = True) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if trailing_lf else b"")


def _object_hash(value: Mapping[str, object], field: str) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {key: member for key, member in value.items() if key != field},
            trailing_lf=False,
        )
    ).hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame, *, compressed: bool) -> None:
    raw = frame.to_csv(index=False, lineterminator="\n").encode()
    if compressed:
        with path.open("wb") as destination:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=destination, compresslevel=9, mtime=0
            ) as handle:
                handle.write(raw)
    else:
        path.write_bytes(raw)
    path.chmod(0o644)


def _market(dates: pd.DatetimeIndex) -> pd.DataFrame:
    values: dict[str, object] = {"date": dates}
    for index, column in enumerate(subject.MARKET_SCHEMA[1:], start=1):
        values[column] = np.arange(len(dates), dtype=float) + index
    return pd.DataFrame(values, columns=subject.MARKET_SCHEMA)


def _manifest() -> dict[str, object]:
    names = (
        "market_5m",
        "funding",
        "premium",
        "open_interest",
        "rex_taker_train",
        "rex_taker_test",
        "rex_taker_eval",
        "rex_veto_source",
    )
    return {
        "schema_version": 1,
        "as_of": "2026-07-16",
        "sources": [
            {
                "name": name,
                "path": f"synthetic/{name}",
                "sha256": f"{index:064x}",
                **({"rows": index} if name.startswith("rex_") else {}),
            }
            for index, name in enumerate(names, start=1)
        ],
    }


def _replace_frame(
    frames: dict[str, tuple[pd.DataFrame, bool]], name: str, frame: pd.DataFrame
) -> None:
    frames[name] = (frame, frames[name][1])


def synthetic_config(
    tmp_path: Path,
    mutate: Callable[[dict[str, tuple[pd.DataFrame, bool]]], None] | None = None,
) -> subject.MaterializationConfig:
    for directory in ("inputs", "results", "data", "configs/shadow"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    old_dates = pd.date_range("2026-01-01", periods=20, freq="5min")
    complete_dates = pd.date_range(old_dates[0], periods=23, freq="5min")
    old_market = _market(old_dates)
    replacement = _market(complete_dates)
    old_oi = pd.DataFrame(
        {"date": old_dates, "open_interest": np.arange(20, dtype=float) + 100}
    )
    metrics_dates = pd.date_range(
        old_dates[-1] - pd.Timedelta(minutes=10), complete_dates[-1], freq="5min"
    )
    old_oi_by_date = dict(zip(old_dates, old_oi.open_interest, strict=True))
    metrics = pd.DataFrame(
        {
            "create_time": metrics_dates,
            "symbol": "BTCUSDT",
            "sum_open_interest": [
                old_oi_by_date.get(date, 200.0 + index)
                for index, date in enumerate(metrics_dates)
            ],
            "sum_open_interest_value": 1.0,
            "count_toptrader_long_short_ratio": 1.0,
            "sum_toptrader_long_short_ratio": 1.0,
            "count_long_short_ratio": 1.0,
            "sum_taker_long_short_vol_ratio": 1.0,
        },
        columns=subject.METRICS_SCHEMA,
    )
    funding = pd.DataFrame(
        {
            "date": [complete_dates[0], old_dates[-1]],
            "funding_rate": [0.001, 0.002],
        }
    )
    premium = pd.DataFrame(
        {
            "date": [complete_dates[0], old_dates[-1]],
            "premium_index": [0.1, 0.2],
        }
    )
    required = complete_dates
    rank_dates = [
        complete_dates[0] - pd.Timedelta(minutes=10),
        complete_dates[0] - pd.Timedelta(minutes=5),
        *required,
        complete_dates[-1] + pd.Timedelta(minutes=5),
        complete_dates[-1] + pd.Timedelta(minutes=10),
    ]
    rank = pd.DataFrame(
        {
            "date": rank_dates,
            "spot_close": np.arange(len(rank_dates), dtype=float) + 1_000,
            "spot_rows": 5.0,
            "premium_index_1m_close": np.arange(len(rank_dates), dtype=float) / 100,
            "premium_rows": 5.0,
            "unused": "bound-but-not-projected",
        }
    )
    frames = {
        "old_market": (old_market, True),
        "replacement_market": (replacement, True),
        "funding": (funding, True),
        "premium": (premium, True),
        "old_open_interest": (old_oi, False),
        "binance_metrics_open_interest": (metrics, True),
        "rank7_spot_premium_5m": (rank, True),
    }
    if mutate is not None:
        mutate(frames)
    bindings = []
    for name, (frame, compressed) in frames.items():
        path = tmp_path / "inputs" / f"{name}.csv{'.gz' if compressed else ''}"
        _write_csv(path, frame, compressed=compressed)
        raw = path.read_bytes()
        bindings.append(
            subject.InputBinding(
                name, str(path), hashlib.sha256(raw).hexdigest(), len(raw), compressed
            )
        )
    kwargs: dict[str, object] = {
        "root": tmp_path,
        "inputs": tuple(bindings),
        "attempt_path": "results/attempt.json",
        "market_output_path": "data/market.csv.gz",
        "oi_output_path": "data/oi.csv.gz",
        "manifest_output_path": "configs/shadow/sources.json",
        "support_output_path": "results/support.json",
        "old_last": old_dates[-1],
        "domain_end": complete_dates[-1] + pd.Timedelta(minutes=5),
        "expected_old_rows": 20,
        "expected_complete_rows": 23,
        "expected_append_rows": 3,
        "splice_rows": 3,
        "inherited_manifest": _manifest(),
    }
    accepted = {field.name for field in fields(subject.MaterializationConfig)}
    return subject.MaterializationConfig(**{key: value for key, value in kwargs.items() if key in accepted})


def _is_absent(path: Path) -> bool:
    return not path.exists() and not path.is_symlink()


def _nested_values(value: object, key: str) -> list[object]:
    found: list[object] = []
    if isinstance(value, Mapping):
        for member, nested in value.items():
            if member == key:
                found.append(nested)
            found.extend(_nested_values(nested, key))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_nested_values(nested, key))
    return found


def _assert_fails(
    config: subject.MaterializationConfig, match: str, failure_reason: str
) -> None:
    with pytest.raises(subject.SourceSupportFailure, match=match) as error:
        subject.materialize(config)
    assert error.value.failure_reason == failure_reason


def test_preflight_inventory_sentinel_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    assert all(_is_absent(root / relative) for relative in G12_OUTPUTS)
    sentinel = root / G11_SENTINEL
    info = os.lstat(sentinel)
    raw = sentinel.read_bytes()
    payload = json.loads(raw)
    assert stat.S_ISREG(info.st_mode)
    assert (stat.S_IMODE(info.st_mode), info.st_nlink, info.st_dev, info.st_ino, len(raw)) == (
        0o444,
        1,
        2096,
        934842,
        3056,
    )
    assert hashlib.sha256(raw).hexdigest() == "128ad6213785ecfa360114eae6e3587254dda3b18e94108b9dd30a0f34533e31"
    assert payload["attempt_hash"] == "6a6204b5074aee399f6a4e318d24764140cfb07aea9b6ebd01b021f7333038f1"
    assert all(_is_absent(root / relative) for relative in G11_ABSENCES)


def test_tail_predicate_exact_membership(tmp_path: Path) -> None:
    def poison_nonrequired(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        rank = frames["rank7_spot_premium_5m"][0].copy().astype(object)
        nonrequired = [0, 1, len(rank) - 2, len(rank) - 1]
        for column in rank.columns:
            if column != "date":
                rank.loc[nonrequired, column] = f"poison-{column}"
        _replace_frame(frames, "rank7_spot_premium_5m", rank)

    support = subject.materialize(synthetic_config(tmp_path / "success", poison_nonrequired))
    expected = {
        "rank7_required_tail_rows": 23,
        "rank7_tail_exact_matches": 23,
        "rank7_pre_tail_coverage_comparison_count": 0,
        "rank7_gap_detail_disclosure_count": 0,
    }
    current = support["access_ledger"]["current_s12"]
    assert {key: current[key] for key in expected} == expected
    for key, value in expected.items():
        assert _nested_values(current, key) == [value]

    def equivalent_offsets(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
        rank = frames["rank7_spot_premium_5m"][0].copy()
        rank["date"] = [
            pd.Timestamp(value)
            .tz_localize("UTC")
            .tz_convert("Asia/Seoul")
            .isoformat()
            for value in rank["date"]
        ]
        _replace_frame(frames, "rank7_spot_premium_5m", rank)

    offset_support = subject.materialize(
        synthetic_config(tmp_path / "equivalent-offsets", equivalent_offsets)
    )
    offset_current = offset_support["access_ledger"]["current_s12"]
    assert {key: offset_current[key] for key in expected} == expected

    mutations: list[tuple[str, Callable[[dict[str, tuple[pd.DataFrame, bool]]], None], str]] = []

    def mutate_rank(label: str, action: Callable[[pd.DataFrame], pd.DataFrame], match: str) -> None:
        def mutation(frames: dict[str, tuple[pd.DataFrame, bool]]) -> None:
            _replace_frame(frames, "rank7_spot_premium_5m", action(frames["rank7_spot_premium_5m"][0].copy()))
        mutations.append((label, mutation, match))

    mutate_rank("missing", lambda frame: frame.drop(index=3), "Rank7|rank7_spot_premium_5m")
    mutate_rank(
        "duplicate",
        lambda frame: pd.concat([frame, frame.iloc[[3]]], ignore_index=True),
            "Rank7|rank7_spot_premium_5m",
    )
    for column in ("spot_close", "premium_index_1m_close", "spot_rows", "premium_rows"):
        def poison_selected(frame: pd.DataFrame, column: str = column) -> pd.DataFrame:
            frame[column] = frame[column].astype(object)
            frame.loc[4, column] = "poison"
            return frame

        mutate_rank(
            f"selected-{column}",
            poison_selected,
                "Rank7|rank7_spot_premium_5m",
        )
    for column in ("spot_close", "premium_index_1m_close"):
        def poison_latest(frame: pd.DataFrame, column: str = column) -> pd.DataFrame:
            frame[column] = frame[column].astype(object)
            frame.loc[len(frame) - 3, column] = "poison"
            return frame

        mutate_rank(
            f"latest-{column}",
            poison_latest,
                "Rank7|rank7_spot_premium_5m",
        )

    for column in ("spot_rows", "premium_rows"):
        mutate_rank(
            f"selected-{column}-wrong-count",
            lambda frame, column=column: frame.assign(
                **{column: frame[column].where(frame.index != 4, 4)}
            ),
            "Rank7|rank7_spot_premium_5m",
        )
    for value in (0, -1):
        mutate_rank(
            f"latest-spot-close-{value}",
            lambda frame, value=value: frame.assign(
                spot_close=frame.spot_close.where(
                    frame.index != len(frame) - 3, value
                )
            ),
            "Rank7|rank7_spot_premium_5m",
        )
    for column in ("spot_close", "premium_index_1m_close", "spot_rows", "premium_rows"):
        for label, value in (("nan", np.nan), ("posinf", np.inf), ("neginf", -np.inf)):
            mutate_rank(
                f"selected-{column}-{label}",
                lambda frame, column=column, value=value: frame.assign(
                    **{column: frame[column].where(frame.index != 4, value)}
                ),
                "Rank7|rank7_spot_premium_5m",
            )

    mutate_rank("schema", lambda frame: frame.rename(columns={"premium_rows": "wrong"}), "Rank7|rank7_spot_premium_5m")
    mutate_rank("invalid-date", lambda frame: frame.assign(date=frame.date.astype(object).where(frame.index != 1, "invalid")), "Rank7|rank7_spot_premium_5m")
    mutate_rank("duplicate-date", lambda frame: frame.assign(date=frame.date.where(frame.index != 1, frame.date.iloc[0])), "Rank7|rank7_spot_premium_5m")
    mutate_rank("nonmonotonic", lambda frame: frame.assign(date=[frame.date.iloc[1], frame.date.iloc[0], *frame.date.iloc[2:]]), "Rank7|rank7_spot_premium_5m")
    mutate_rank("off-grid", lambda frame: frame.assign(date=frame.date.where(frame.index != 0, frame.date.iloc[0] + pd.Timedelta(minutes=1))), "Rank7|rank7_spot_premium_5m")

    for label, mutation, match in mutations:
        reason = (
            subject.RANK7_TAIL_MEMBERSHIP_MISMATCH
            if label == "missing"
            else subject.STRUCTURAL_OR_SCHEMA_VIOLATION
        )
        _assert_fails(synthetic_config(tmp_path / label, mutation), match, reason)


def test_forbidden_ast_operations() -> None:
    source = Path(subject.__file__).read_text()
    tree = ast.parse(source)
    assert "compileall" not in source
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    transform = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_transform")
    transform_lines = source.splitlines()[transform.lineno - 1 : transform.end_lineno]
    start = next(index for index, line in enumerate(transform_lines) if "rank_raw" in line)
    stop = next(index for index, line in enumerate(transform_lines[start + 1 :], start + 1) if "funding_raw" in line)
    rank_source = "\n".join(transform_lines[start:stop])
    rank_tree = ast.parse("def scoped():\n" + rank_source)

    def local_callees(node: ast.AST) -> set[str]:
        def target_names(target: ast.AST) -> set[str]:
            if isinstance(target, ast.Name):
                return {target.id}
            if isinstance(target, (ast.Tuple, ast.List)):
                return set().union(*(target_names(item) for item in target.elts))
            return set()

        aliases: dict[str, str] = {}
        dynamic_aliases: set[str] = set()
        assigned_names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.NamedExpr):
                assigned_names.update(target_names(child.target))
                continue
            if not isinstance(child, (ast.Assign, ast.AnnAssign)):
                continue
            value = child.value
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            names = set().union(*(target_names(target) for target in targets))
            assigned_names.update(names)
            if isinstance(value, ast.Name):
                for name in names:
                    aliases[name] = value.id
            else:
                dynamic_aliases.update(names)

        def resolve(name: str) -> str:
            seen: set[str] = set()
            while name in aliases and name not in seen:
                seen.add(name)
                name = aliases[name]
            return name

        called_names = {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        assert all(
            isinstance(child.func, (ast.Name, ast.Attribute))
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        )
        assert not (called_names & assigned_names)
        assert not (called_names & dynamic_aliases)
        return {
            resolved
            for name in called_names
            if (resolved := resolve(name)) in functions
        }

    pending = {
        name for name in functions if "rank7" in name.lower()
    } | local_callees(rank_tree)
    reachable: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.update(local_callees(functions[name]) - reachable)
    calls = []
    for node in [*(functions[name] for name in sorted(reachable)), rank_tree]:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                calls.append(ast.unparse(child.func).lower())
    forbidden = ("merge", "join", "merge_asof", "fill", "interpolate", "resample", "repair", "tolerance")
    assert not [call for call in calls if any(word in call for word in forbidden)]
    assert "astype('datetime" not in rank_source.lower()
    assert 'astype("datetime' not in rank_source.lower()
    assert "to_datetime" not in rank_source.lower()
    assert "rank7_all_history" not in rank_source.lower()
    selector = functions["_select_rank7_required_tail"]
    numeric_calls = [
        child
        for child in ast.walk(selector)
        if isinstance(child, ast.Call) and ast.unparse(child.func) == "pd.to_numeric"
    ]
    assert numeric_calls
    assert all(ast.unparse(call.args[0]).startswith("selected[") for call in numeric_calls)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not {
        module
        for module in imported_modules
        if any(
            fragment in module.lower()
            for fragment in ("backtest", "candidate", "econom", "evaluator")
        )
    }
    call_names = {
        ast.unparse(node.func).lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not {
        name
        for name in call_names
        if any(
            fragment in name
            for fragment in ("cagr", "drawdown", "strict_mdd", "evaluate_candidate")
        )
    }


def test_stage_modes_and_parent_diff_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    assert getattr(subject, "A12") == A12
    assert getattr(subject, "T11") == T11
    assert _git(root, "rev-parse", f"{A12}^") == S11
    assert _git(root, "rev-parse", f"{T11}^") == A12
    assert _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", A12).splitlines() == [
        "A\tdocs/gross9-structural-clock-bundle-g9cb12-successor-authority-decision-2026-07-31.md"
    ]
    assert _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", T11).splitlines() == [
        f"A\t{G11_SENTINEL}",
        "A\tresults/gross9_structural_clock_bundle_g9cb11_source_support_terminal_failure_2026-07-31.json",
    ]
    assert all(_git(root, "ls-tree", A12, path).split()[0] == "100644" for path in ("docs/gross9-structural-clock-bundle-g9cb12-successor-authority-decision-2026-07-31.md",))
    assert all(_git(root, "ls-tree", T11, path).split()[0] == "100644" for path in (G11_SENTINEL, "results/gross9_structural_clock_bundle_g9cb11_source_support_terminal_failure_2026-07-31.json"))
    head = _git(root, "rev-parse", "HEAD")
    if head == T11:
        assert _git(root, "status", "--porcelain", "--untracked-files=all").splitlines() == [
            f"?? {path}" for path in sorted(S12_FILES)
        ]
    else:
        assert _git(root, "rev-parse", "HEAD^") == T11
        assert sorted(
            _git(
                root,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                "HEAD",
            ).splitlines()
        ) == sorted(S12_FILES)
        assert all(_git(root, "ls-tree", "HEAD", path).split()[0] == "100644" for path in S12_FILES)
        assert _git(root, "rev-parse", "HEAD") == _git(root, "rev-parse", "@{upstream}")
    for path in S12_FILES:
        assert stat.S_IMODE((root / path).stat().st_mode) == 0o644
    assert _git(root, "ls-files", "--", *G12_OUTPUTS) == ""
    ignored = subprocess.run(
        ["git", "check-ignore", "--", *G12_OUTPUTS],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    assert ignored == [G12_OUTPUTS[index] for index in (0, 1, 2, 4)]
    terminal = (root / "results/gross9_structural_clock_bundle_g9cb11_source_support_terminal_failure_2026-07-31.json").read_text()
    assert "<S12_COMMIT>" not in terminal

    official = subject.official_config(root)
    raw_paths = {
        (Path(binding.path) if Path(binding.path).is_absolute() else root / binding.path)
        .resolve(strict=False)
        for binding in official.inputs
    }
    original_open = subject._open_nofollow_components
    opened_history_paths: list[Path] = []

    def history_only_open(path: Path, flags: int) -> int:
        resolved = Path(path).resolve(strict=False)
        assert resolved not in raw_paths
        opened_history_paths.append(resolved)
        return original_open(path, flags)

    monkeypatch.setattr(subject, "_open_nofollow_components", history_only_open)
    subject._validate_history_gate(official)
    assert opened_history_paths

    authority_binding = subject._A12_AUTHORITY
    monkeypatch.setattr(
        subject,
        "_A12_AUTHORITY",
        (authority_binding[0], authority_binding[1], "0" * 64, authority_binding[3]),
    )
    with pytest.raises(subject.SourceSupportFailure, match="A12 authority binding"):
        subject._validate_history_gate(official)
    monkeypatch.setattr(subject, "_A12_AUTHORITY", authority_binding)

    original_read = subject._read_bound_json

    def mutated_terminal(*args: object, **kwargs: object) -> dict[str, object]:
        payload = original_read(*args, **kwargs)
        if args[1] == subject._T11_EVIDENCE[1][0]:
            payload["status"] = "bogus-history-status"
        return payload

    monkeypatch.setattr(subject, "_read_bound_json", mutated_terminal)
    with pytest.raises(subject.SourceSupportFailure, match="T11 terminal ledger constants"):
        subject._validate_history_gate(official)


def test_bytecode_residue_scan() -> None:
    root = Path(__file__).resolve().parents[1]
    residue = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
    )
    assert residue == []
    assert "compileall" not in Path(subject.__file__).read_text()


def test_synthetic_materialize_e2e_no_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_main() -> int:
        raise AssertionError("main and official commands are forbidden in synthetic tests")

    monkeypatch.setattr(subject, "main", forbidden_main)
    config = synthetic_config(tmp_path)
    support = subject.materialize(config)
    assert set(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()).issuperset(config.output_paths)
    assert all((tmp_path / relative).is_file() for relative in config.output_paths)
    assert all(
        stat.S_IMODE((tmp_path / relative).stat().st_mode) == 0o444
        and (tmp_path / relative).stat().st_nlink == 1
        for relative in config.output_paths
    )
    assert all(Path(binding.path).is_relative_to(tmp_path) for binding in config.inputs)
    with pytest.raises(subject.SourceSupportFailure, match="one-shot"):
        subject.materialize(config)
    access = support["access_ledger"]
    assert set(access) == {
        "access_ledger_hash",
        "current_s12",
        "historical_s11",
        "ledger_kind",
        "process_local",
        "replay_guard",
        "schema_version",
    }
    assert access["schema_version"] == 1
    assert access["ledger_kind"] == "gross9_structural_clock_bundle_g9cb12_access_v1"
    assert access["access_ledger_hash"] == _object_hash(
        access, "access_ledger_hash"
    )
    historical = access["historical_s11"]
    assert set(historical) == {
        "access",
        "attempt_sentinel",
        "authority",
        "authority_transferred",
        "execution",
        "failure",
        "identity",
        "implementation",
        "output_state",
        "permanently_absent_output_paths",
        "seal_authority",
        "status",
        "terminal_ledger",
    }
    repository_root = Path(__file__).resolve().parents[1]
    terminal_path = (
        repository_root
        / "results/gross9_structural_clock_bundle_g9cb11_source_support_terminal_failure_2026-07-31.json"
    )
    terminal_raw = terminal_path.read_bytes()
    assert hashlib.sha256(terminal_raw).hexdigest() == (
        "da943985354e4abfab87a06a16576235ab34a487bdff0a4b498ae6fb1728e045"
    )
    terminal = json.loads(terminal_raw)
    assert terminal_raw == _canonical_json_bytes(terminal)
    assert terminal["terminal_failure_hash"] == _object_hash(
        terminal, "terminal_failure_hash"
    )
    assert set(terminal) == {
        "access",
        "attempt_sentinel",
        "authority",
        "execution",
        "failure",
        "identity",
        "implementation",
        "ledger_kind",
        "output_state",
        "schema_version",
        "seal_authority",
        "status",
        "terminal_failure_hash",
    }
    assert len(terminal["access"]) == 34
    assert terminal["status"] == "terminal_rank7_all_history_coverage_failure"
    assert terminal["access"]["economic_evaluation_count"] == 0
    for key in (
        "access",
        "attempt_sentinel",
        "authority",
        "execution",
        "failure",
        "identity",
        "implementation",
        "output_state",
        "seal_authority",
        "status",
    ):
        assert historical[key] == terminal[key]
    assert historical["authority_transferred"] is False
    assert historical["attempt_sentinel"]["path"] == G11_SENTINEL
    assert historical["permanently_absent_output_paths"] == list(G11_ABSENCES)
    assert historical["access"]["rank7_spot_premium_5m_decode_count"] == 1
    assert historical["access"]["rank7_all_history_coverage_comparison_count"] == 1
    assert historical["access"]["rank7_tail_completeness_evaluation_count"] == 0
    assert historical["failure"]["rank7_gap_detail_disclosure_count"] == 0
    current = access["current_s12"]
    assert set(current) == {
        "access",
        "attempt_sentinel",
        "authority",
        "execution",
        "identity",
        "implementation",
        "rank7_gap_detail_disclosure_count",
        "rank7_pre_tail_coverage_comparison_count",
        "rank7_required_tail_rows",
        "rank7_tail_exact_matches",
        "required_tail",
        "terminal_evidence",
    }
    assert current["authority"]["commit"] == A12
    assert current["terminal_evidence"]["commit"] == T11
    assert current["implementation"]["parent_commit"] == T11
    assert current["implementation"]["paths"] == list(S12_FILES)
    assert current["required_tail"] == {
        "predicate": "market.date.tail(min(3000, len(market)))",
        "selection": "exact_date_membership_only",
        "selected_dates_equal_required_dates": True,
        "selected_required_dates_unique": True,
    }
    assert {key: current[key] for key in (
        "rank7_required_tail_rows",
        "rank7_tail_exact_matches",
        "rank7_pre_tail_coverage_comparison_count",
        "rank7_gap_detail_disclosure_count",
    )} == {
        "rank7_required_tail_rows": 23,
        "rank7_tail_exact_matches": 23,
        "rank7_pre_tail_coverage_comparison_count": 0,
        "rank7_gap_detail_disclosure_count": 0,
    }
    zero_economics = {
        "candidate_value_rows_opened",
        "comparator_value_rows_opened",
        "feature_value_rows_opened",
        "schedule_value_rows_opened",
        "signal_value_rows_opened",
        "return_value_rows_opened",
        "pnl_value_rows_opened",
        "cagr_evaluation_count",
        "mdd_evaluation_count",
        "drawdown_evaluation_count",
        "economic_value_rows_opened",
        "economic_evaluation_count",
    }
    expected_current_access = {
        "attempt_sentinel_publication_count",
        "decode_pass_count",
        "decode_passes",
        "generated_output_publication_count",
        "generated_output_readback_count",
        "global_metrics_alignment_comparison_count",
        "metrics_date_scan_count",
        "metrics_overlap_row_count",
        "metrics_selected_decode_count",
        "metrics_selected_row_count",
        "metrics_tail_row_count",
        "non_selected_metrics_non_date_semantic_evaluation_count",
        "off_grid_detail_disclosure_count",
        "rank7_spot_premium_5m_decode_count",
        "raw_file_count",
        "raw_file_open_count",
        "replacement_market_date_scan_count",
        "replacement_market_tail_decode_count",
        "replacement_market_tail_selected_row_count",
        *zero_economics,
    }
    assert set(current["access"]) == expected_current_access
    assert all(current["access"][key] == 0 for key in zero_economics)
    assert {
        key: current["access"][key]
        for key in expected_current_access - zero_economics - {"decode_passes"}
    } == {
        "attempt_sentinel_publication_count": 1,
        "decode_pass_count": 9,
        "generated_output_publication_count": 4,
        "generated_output_readback_count": 2,
        "global_metrics_alignment_comparison_count": 0,
        "metrics_date_scan_count": 1,
        "metrics_overlap_row_count": 3,
        "metrics_selected_decode_count": 1,
        "metrics_selected_row_count": 6,
        "metrics_tail_row_count": 3,
        "non_selected_metrics_non_date_semantic_evaluation_count": 0,
        "off_grid_detail_disclosure_count": 0,
        "rank7_spot_premium_5m_decode_count": 1,
        "raw_file_count": 7,
        "raw_file_open_count": 7,
        "replacement_market_date_scan_count": 1,
        "replacement_market_tail_decode_count": 1,
        "replacement_market_tail_selected_row_count": 3,
    }
    assert current["access"]["decode_passes"] == [
        "old_market",
        "replacement_market_date_scan",
        "replacement_market_tail",
        "funding",
        "premium",
        "old_open_interest",
        "binance_metrics_open_interest_date_scan",
        "binance_metrics_open_interest_selected_window",
        "rank7_spot_premium_5m",
    ]
    forbidden_count_fragments = ("physical", "raw_row", "source_row", "source_total", "rank7_row_count")
    for container_name, values in (
        ("support", support),
        ("access", access),
        ("raw_sources", support["raw_sources"]),
    ):
        assert not [
            key
            for key in _all_keys(values)
            if any(fragment in key.lower() for fragment in forbidden_count_fragments)
        ], container_name
    rank7_rows = [row for row in support["raw_sources"] if row["name"] == "rank7_spot_premium_5m"]
    assert len(rank7_rows) == 1
    assert not ({"decoded_rows", "normalized_rows", "rows", "row_count"} & set(rank7_rows[0]))
    access_keys = set(_all_keys(access))
    assert "rank7_spot_premium_5m_rows_opened" not in access_keys
    assert "source_value_rows_opened" not in access_keys
    assert support["support_hash"] == _object_hash(support, "support_hash")
    assert support["attempt_sentinel"]["attempt_hash"] == _object_hash(
        json.loads((tmp_path / config.attempt_path).read_text()), "attempt_hash"
    )
    assert (tmp_path / config.support_output_path).read_bytes() == _canonical_json_bytes(
        support
    )
    assert access["replay_guard"]["stage_order"] == ["A12", "T11", "S12"]
    assert access["replay_guard"]["t11"]["permanently_absent_output_paths"] == list(G11_ABSENCES)
    assert access["replay_guard"]["replay_guard_hash"] == _object_hash(
        access["replay_guard"], "replay_guard_hash"
    )


def _all_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_all_keys(nested))
    return keys


def test_terminal_pair_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = (
        "pre_sentinel_failure",
        "post_sentinel_pre_other_output_failure",
        "partial_publication_failure",
    )
    reasons = (
        "preflight_or_binding_failure",
        "rank7_tail_membership_mismatch",
        "structural_or_schema_violation",
        "zero_disclosure_breach",
        "publication_or_readback_failure",
        "final_reauthentication_failure",
    )
    allowed = {
        (states[0], reasons[0]),
        (states[0], reasons[4]),
        (states[1], reasons[1]),
        (states[1], reasons[2]),
        (states[1], reasons[3]),
        (states[1], reasons[4]),
        (states[2], reasons[4]),
        (states[2], reasons[5]),
    }
    validator = getattr(subject, "validate_terminal_pair")
    for state in states:
        for reason in reasons:
            if (state, reason) in allowed:
                validator(state, reason)
            else:
                with pytest.raises(subject.SourceSupportFailure):
                    validator(state, reason)

    config = synthetic_config(tmp_path / "source-config")
    for prefix_length, expected in enumerate((states[0], states[1], states[2], states[2], states[2], states[2])):
        root = tmp_path / f"prefix-{prefix_length}"
        root.mkdir()
        prefix_config = replace(config, root=root)
        for relative in prefix_config.output_paths[:prefix_length]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("immutable\n")
            path.chmod(0o444)
        assert subject.classify_terminal_publication_state(prefix_config) == expected
        for reason in reasons:
            if (expected, reason) in allowed:
                assert subject.classify_and_validate_terminal_pair(
                    prefix_config, reason
                ) == (expected, reason)
                failure = subject.SourceSupportFailure(
                    "synthetic terminal",
                    failure_reason=reason,
                )
                assert subject.classify_terminal_failure(
                    prefix_config, failure
                ) == (expected, reason)
            else:
                with pytest.raises(subject.SourceSupportFailure):
                    subject.classify_and_validate_terminal_pair(prefix_config, reason)
                with pytest.raises(subject.SourceSupportFailure):
                    subject.classify_terminal_failure(
                        prefix_config,
                        subject.SourceSupportFailure(
                            "synthetic terminal",
                            failure_reason=reason,
                        ),
                    )

    original_linkat = subject._linkat
    for failing_call in range(1, 6):
        calls = 0

        def failing_linkat(
            fd: int,
            directory_fd: int,
            leaf: str,
            *,
            failing_call: int = failing_call,
        ) -> None:
            nonlocal calls
            calls += 1
            if calls == failing_call:
                raise OSError("synthetic linkat failure")
            original_linkat(fd, directory_fd, leaf)

        monkeypatch.setattr(subject, "_linkat", failing_linkat)
        syscall_config = synthetic_config(tmp_path / f"linkat-{failing_call}")
        with pytest.raises(subject.SourceSupportFailure) as captured:
            subject.materialize(syscall_config)
        assert captured.value.failure_reason == subject.PUBLICATION_OR_READBACK_FAILURE
        expected_state = (states[0], states[1], states[2], states[2], states[2])[failing_call - 1]
        assert subject.classify_terminal_failure(
            syscall_config, captured.value
        ) == (expected_state, subject.PUBLICATION_OR_READBACK_FAILURE)
    monkeypatch.setattr(subject, "_linkat", original_linkat)

    original_fsync = subject.os.fsync
    directory_fsync_calls = 0

    def fail_after_support_link(fd: int) -> None:
        nonlocal directory_fsync_calls
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_fsync_calls += 1
            if directory_fsync_calls == 5:
                raise OSError("synthetic support-directory fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(subject.os, "fsync", fail_after_support_link)
    full_prefix_config = synthetic_config(tmp_path / "support-directory-fsync")
    with pytest.raises(subject.SourceSupportFailure) as captured:
        subject.materialize(full_prefix_config)
    assert captured.value.failure_reason == subject.PUBLICATION_OR_READBACK_FAILURE
    assert subject.classify_terminal_failure(
        full_prefix_config, captured.value
    ) == (states[2], subject.PUBLICATION_OR_READBACK_FAILURE)
    assert all((full_prefix_config.root / path).exists() for path in full_prefix_config.output_paths)
    monkeypatch.setattr(subject.os, "fsync", original_fsync)

    original_decode_csv = subject._decode_csv

    def decode_io_failure(*_args: object, **_kwargs: object) -> object:
        raise OSError("synthetic source-decode I/O failure")

    monkeypatch.setattr(subject, "_decode_csv", decode_io_failure)
    decode_config = synthetic_config(tmp_path / "decode-io")
    with pytest.raises(subject.SourceSupportFailure) as captured:
        subject.materialize(decode_config)
    assert captured.value.failure_reason == subject.STRUCTURAL_OR_SCHEMA_VIOLATION
    assert subject.classify_terminal_failure(
        decode_config, captured.value
    ) == (states[1], subject.STRUCTURAL_OR_SCHEMA_VIOLATION)
    monkeypatch.setattr(subject, "_decode_csv", original_decode_csv)

    def final_reauthentication_failure(name: str) -> None:
        if name == "final_input_reauthentication:0:before_hash":
            raise OSError("synthetic final reauthentication failure")

    final_config = synthetic_config(tmp_path / "final-reauthentication")
    with pytest.raises(subject.SourceSupportFailure) as captured:
        subject.materialize(final_config, failpoint=final_reauthentication_failure)
    assert captured.value.failure_reason == subject.FINAL_REAUTHENTICATION_FAILURE
    assert subject.classify_terminal_failure(
        final_config, captured.value
    ) == (states[2], subject.FINAL_REAUTHENTICATION_FAILURE)

    def impossible_structural_failure(name: str) -> None:
        if name == "checkpoint:4":
            raise subject.SourceSupportFailure(
                "synthetic late structural failure",
                failure_reason=subject.STRUCTURAL_OR_SCHEMA_VIOLATION,
            )

    impossible_config = synthetic_config(tmp_path / "late-structural")
    with pytest.raises(
        subject.SourceSupportFailure,
        match="invalid terminal publication-state/failure-reason pair",
    ):
        subject.materialize(impossible_config, failpoint=impossible_structural_failure)
    malformed = replace(config, root=tmp_path / "malformed")
    malformed.root.mkdir()
    for relative in (malformed.output_paths[0], malformed.output_paths[2]):
        path = malformed.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not-prefix\n")
        path.chmod(0o444)
    with pytest.raises(subject.SourceSupportFailure, match="prefix"):
        subject.classify_terminal_publication_state(malformed)
