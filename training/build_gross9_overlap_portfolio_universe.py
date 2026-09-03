"""Build the outcome-blind clock universe for G9-OVERLAP-PORT-1."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from training import build_gross9_async_active_veto_train_clocks as active_builder
from training import preregister_gross9_async_active_veto_search as active_prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

POLICY_ID = "G9-OVERLAP-PORT-1"
PROTOCOL_VERSION = "gross9_overlap_portfolio_universe_v1"
AS_OF_DATE = "2026-09-03"
RESULTS_DIR = Path("results")
OUTPUT = RESULTS_DIR / "gross9_overlap_portfolio_universe_2026-09-03.json"
HISTORICAL_INVENTORY = RESULTS_DIR / "gross9_overlap_portfolio_historical_novelty_inventory_2026-09-03.json"
ACTIVE_CLOCK_DIR = Path("data/gross9_overlap_portfolio_active_veto_clocks_2026-09-03")
EXCLUDED_POLICIES = {"G9QTR-DISTILL-8"}
SPLITS = ("train", "test", "eval", "final")
STAGE_BOUNDS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} JSON object required: {path}")
    return value


def verify_manifest(value: Mapping[str, Any], label: str) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} manifest drift: {label}")


def load_clock(path: str | Path, *, expected_sha: str | None = None, expected_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError(f"{POLICY_ID} clock hash drift: {path}")
    frame = pd.read_csv(path, compression="gzip")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise RuntimeError(f"{POLICY_ID} clock row drift: {path}")
    required = {"split", "entry_time", "exit_time", "side"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{POLICY_ID} clock schema drift: {path}")
    out = frame.copy()
    out["split"] = out["split"].astype(str)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="raise")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="raise")
    out["side"] = pd.to_numeric(out["side"], errors="raise").astype(int)
    if set(out["split"]) - set(SPLITS) or set(out["side"]) - {-1, 1}:
        raise RuntimeError(f"{POLICY_ID} split/side drift: {path}")
    for split, group in out.groupby("split"):
        start, end = STAGE_BOUNDS[split]
        ordered = group.sort_values("entry_time")
        if not (ordered.entry_time.ge(start).all() and ordered.entry_time.lt(end).all() and ordered.exit_time.le(end).all()):
            raise RuntimeError(f"{POLICY_ID} stage containment drift: {path}/{split}")
        if len(ordered) > 1 and (ordered.entry_time.iloc[1:].to_numpy() < ordered.exit_time.iloc[:-1].to_numpy()).any():
            raise RuntimeError(f"{POLICY_ID} intra-sleeve overlap: {path}/{split}")
    return out.sort_values(["split", "entry_time", "exit_time", "side"], kind="stable").reset_index(drop=True)


def clock_signature(frame: pd.DataFrame) -> str:
    rows = [
        [str(r.split), pd.Timestamp(r.entry_time).isoformat(), pd.Timestamp(r.exit_time).isoformat(), int(r.side)]
        for r in frame[["split", "entry_time", "exit_time", "side"]].itertuples(index=False)
    ]
    return canonical_hash(rows)


def discover_failed_novelty_records(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*gross9*novelty*.json")):
        value = load_json(path)
        verify_manifest(value, str(path))
        sleeves = value.get("gross9_sleeves")
        if not isinstance(sleeves, Mapping) or not sleeves:
            continue
        failed: dict[str, float] = {}
        invalid = False
        for sleeve, row in sleeves.items():
            checks = row.get("checks", {}) if isinstance(row, Mapping) else {}
            failures = {name for name, passed in checks.items() if passed is not True}
            if failures:
                if failures != {"one_to_one_6h_max_matched_share"}:
                    invalid = True
                    break
                failed[str(sleeve)] = float(row["metrics"]["one_to_one_6h_max_matched_share"])
        policy = str(value.get("policy_id", ""))
        if not invalid and failed and policy not in EXCLUDED_POLICIES:
            records.append({
                "policy_id": policy,
                "novelty_path": str(path),
                "novelty_sha256": sha256_file(path),
                "novelty_manifest_hash": value["manifest_hash"],
                "novelty": value,
                "failed_near_6h": failed,
            })
    if len(records) != 64 or len({row["policy_id"] for row in records}) != 64:
        raise RuntimeError(f"{POLICY_ID} expected 64 historical overlap-only policies")
    return sorted(records, key=lambda row: row["policy_id"])


def write_historical_inventory(
    output: Path = HISTORICAL_INVENTORY,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    discovered = discover_failed_novelty_records(results_dir)
    core = {
        "protocol_version": "gross9_overlap_historical_novelty_inventory_v1",
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "selection": "exact 64 historical Gross9 novelty artifacts failing only one_to_one_6h_max_matched_share; G9QTR-DISTILL-8 excluded",
        "records": [
            {
                "policy_id": row["policy_id"],
                "path": row["novelty_path"],
                "sha256": row["novelty_sha256"],
                "manifest_hash": row["novelty_manifest_hash"],
                "failed_near_6h": row["failed_near_6h"],
            }
            for row in discovered
        ],
        "ambient_glob_is_not_authority_after_freeze": True,
        "evidence_boundary": {
            "clock_rows_opened": False,
            "market_rows_opened": 0,
            "funding_rows_opened": 0,
            "economic_outcomes_opened": False,
        },
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def failed_novelty_records(inventory_path: Path = HISTORICAL_INVENTORY) -> list[dict[str, Any]]:
    if not inventory_path.is_file():
        raise RuntimeError(f"{POLICY_ID} missing frozen historical novelty inventory: {inventory_path}")
    inventory = load_json(inventory_path)
    verify_manifest(inventory, str(inventory_path))
    records = inventory.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise RuntimeError(f"{POLICY_ID} frozen historical novelty inventory must contain 64 records")
    out: list[dict[str, Any]] = []
    for receipt in records:
        path = Path(str(receipt.get("path", "")))
        if sha256_file(path) != receipt.get("sha256"):
            raise RuntimeError(f"{POLICY_ID} historical novelty SHA drift: {path}")
        value = load_json(path)
        verify_manifest(value, str(path))
        if value.get("policy_id") != receipt.get("policy_id") or value.get("manifest_hash") != receipt.get("manifest_hash"):
            raise RuntimeError(f"{POLICY_ID} historical novelty receipt drift: {path}")
        out.append({
            "policy_id": receipt["policy_id"],
            "novelty_path": str(path),
            "novelty_sha256": receipt["sha256"],
            "novelty_manifest_hash": receipt["manifest_hash"],
            "novelty": value,
            "failed_near_6h": receipt["failed_near_6h"],
        })
    if len({row["policy_id"] for row in out}) != 64:
        raise RuntimeError(f"{POLICY_ID} frozen historical novelty policy IDs must be unique")
    return out


def source_clock_binding(novelty: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = novelty.get("source_support", {})
    path = Path(str(receipt.get("path", "")))
    if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
        raise RuntimeError(f"{POLICY_ID} source-support receipt drift: {path}")
    source = load_json(path)
    verify_manifest(source, str(path))
    if source.get("manifest_hash") != receipt.get("manifest_hash"):
        raise RuntimeError(f"{POLICY_ID} source-support manifest binding drift: {path}")
    clock = source.get("clock")
    if not isinstance(clock, Mapping):
        clock = source.get("source_artifacts", {}).get("clock")
    if not isinstance(clock, Mapping):
        raise RuntimeError(f"{POLICY_ID} missing source clock binding: {path}")
    return dict(clock), source


def historical_sleeves(inventory_path: Path = HISTORICAL_INVENTORY) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in failed_novelty_records(inventory_path):
        clock, source = source_clock_binding(record["novelty"])
        frame = load_clock(clock["path"], expected_sha=str(clock["sha256"]), expected_rows=int(clock["rows"]))
        out.append({
            "sleeve_id": record["policy_id"],
            "clock": {"path": str(clock["path"]), "sha256": str(clock["sha256"]), "rows": int(clock["rows"])},
            "split_counts": {split: int(frame["split"].eq(split).sum()) for split in SPLITS},
            "schedule_signature": clock_signature(frame),
            "provenance": {
                "kind": "historical_gross9_near6h_only_reject",
                "novelty": {"path": record["novelty_path"], "sha256": sha256_file(record["novelty_path"]), "manifest_hash": record["novelty"]["manifest_hash"]},
                "source_support": record["novelty"]["source_support"],
                "failed_near_6h": record["failed_near_6h"],
                "source_support_passed": source.get("support_passed"),
            },
        })
    return out


def load_full_component(component: str) -> pd.DataFrame:
    binding = active_prereg.COMPONENT_ARTIFACTS[component]["clock"]
    frame = load_clock(binding["path"], expected_sha=binding["sha256"], expected_rows=binding["rows"])
    required = set(CLOCK_COLUMNS)
    raw = pd.read_csv(binding["path"], compression="gzip")
    if not required.issubset(raw.columns):
        raise RuntimeError(f"{POLICY_ID} active component schema drift: {component}")
    raw = raw[list(CLOCK_COLUMNS)].copy()
    for column in ("decision_time", "feature_available_time", "entry_time", "exit_time"):
        raw[column] = pd.to_datetime(raw[column], utc=True, errors="raise")
    raw["side"] = pd.to_numeric(raw["side"], errors="raise").astype(int)
    if raw["entry_time"].duplicated().any() or not raw["decision_time"].le(raw["entry_time"]).all() or not raw["feature_available_time"].le(raw["entry_time"]).all():
        raise RuntimeError(f"{POLICY_ID} active component integrity drift: {component}")
    return raw.sort_values(["split", "entry_time"], kind="stable").reset_index(drop=True)


def build_active_full_clock(candidate: str, base: str, veto: str, base_clock: pd.DataFrame, veto_clock: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        base_split = base_clock[base_clock["split"].eq(split)].sort_values("entry_time")
        veto_split = veto_clock[veto_clock["split"].eq(split)].sort_values("entry_time")
        candidates: list[dict[str, Any]] = []
        for event in base_split.itertuples(index=False):
            window = veto_split[veto_split["entry_time"].gt(event.entry_time - pd.Timedelta(hours=6)) & veto_split["entry_time"].le(event.entry_time)]
            veto_row = window.iloc[-1] if not window.empty else None
            if veto_row is not None and int(veto_row["side"]) == -int(event.side):
                continue
            decision = pd.Timestamp(event.decision_time)
            available = pd.Timestamp(event.feature_available_time)
            if veto_row is not None:
                decision = max(decision, pd.Timestamp(veto_row["decision_time"]))
                available = max(available, pd.Timestamp(veto_row["feature_available_time"]))
            candidates.append({
                "candidate": candidate, "control": "primary", "split": split,
                "decision_time": decision, "feature_available_time": available,
                "entry_time": pd.Timestamp(event.entry_time), "exit_time": pd.Timestamp(event.entry_time) + pd.Timedelta(hours=8), "side": int(event.side),
                "base_component_id": base, "veto_component_id": veto,
            })
        next_available: pd.Timestamp | None = None
        for row in sorted(candidates, key=lambda item: item["entry_time"]):
            if next_available is not None and row["entry_time"] < next_available:
                continue
            rows.append(row)
            next_available = row["exit_time"]
    frame = pd.DataFrame(rows)
    return frame.sort_values(["split", "entry_time"], kind="stable").reset_index(drop=True)


def active_duplicate_only_sleeves(output_dir: Path = ACTIVE_CLOCK_DIR) -> list[dict[str, Any]]:
    support = load_json(active_builder.RESULT)
    verify_manifest(support, str(active_builder.RESULT))
    eligible = [
        candidate for candidate in active_prereg.CANDIDATE_FAMILY
        if all(support["candidates"][candidate]["support_checks"].values())
        and support["candidates"][candidate]["duplicate_gate"]["rejected"] is True
    ]
    if len(eligible) != 13:
        raise RuntimeError(f"{POLICY_ID} expected 13 duplicate-only active candidates")
    groups: dict[str, list[str]] = defaultdict(list)
    for candidate in eligible:
        binding = support["candidates"][candidate]["clock"]
        train = load_clock(binding["path"], expected_sha=binding["sha256"], expected_rows=binding["rows"])
        groups[clock_signature(train)].append(candidate)
    canonical = [min(group, key=active_prereg.CANDIDATE_FAMILY.index) for group in groups.values()]
    canonical.sort(key=active_prereg.CANDIDATE_FAMILY.index)
    if len(canonical) != 7:
        raise RuntimeError(f"{POLICY_ID} expected seven canonical active schedules")
    output_dir.mkdir(parents=True, exist_ok=True)
    needed = sorted({part for candidate in canonical for part in candidate.split("__ASYNC_ACTIVE_OPPOSITE_VETO_6H__")})
    components = {component: load_full_component(component) for component in needed}
    out: list[dict[str, Any]] = []
    for candidate in canonical:
        base, veto = candidate.split("__ASYNC_ACTIVE_OPPOSITE_VETO_6H__")
        frame = build_active_full_clock(candidate, base, veto, components[base], components[veto])
        path = output_dir / f"{candidate}.csv.gz"
        _write_gzip_csv(frame, path)
        validated = load_clock(path)
        aliases = [
            alias
            for alias in groups[
                clock_signature(
                    load_clock(support["candidates"][candidate]["clock"]["path"])
                )
            ]
            if alias != candidate
        ]
        out.append({
            "sleeve_id": candidate,
            "clock": {"path": str(path), "sha256": sha256_file(path), "rows": len(validated)},
            "split_counts": {split: int(validated["split"].eq(split).sum()) for split in SPLITS},
            "schedule_signature": clock_signature(validated),
            "provenance": {"kind": "active_veto_duplicate_only_canonical", "aliases": aliases, "train_source_support": {"path": str(active_builder.RESULT), "sha256": sha256_file(active_builder.RESULT), "manifest_hash": support["manifest_hash"]}},
        })
    return out


def build(output: Path = OUTPUT, active_clock_dir: Path = ACTIVE_CLOCK_DIR) -> dict[str, Any]:
    pre = [*historical_sleeves(), *active_duplicate_only_sleeves(active_clock_dir)]
    if len(pre) != 71:
        raise RuntimeError(f"{POLICY_ID} expected 71 pre-canonical schedules")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pre:
        groups[record["schedule_signature"]].append(record)
    sleeves: list[dict[str, Any]] = []
    cross_aliases: list[dict[str, Any]] = []
    order = {record["sleeve_id"]: index for index, record in enumerate(pre)}
    for signature, group in groups.items():
        group.sort(key=lambda row: order[row["sleeve_id"]])
        canonical = dict(group[0])
        aliases = [row["sleeve_id"] for row in group[1:]]
        canonical["cross_universe_aliases"] = aliases
        sleeves.append(canonical)
        if aliases:
            cross_aliases.append({"canonical": canonical["sleeve_id"], "aliases": aliases, "schedule_signature": signature})
    sleeves.sort(key=lambda row: order[row["sleeve_id"]])
    core = {
        "protocol_version": PROTOCOL_VERSION, "policy_id": POLICY_ID, "as_of_date": AS_OF_DATE,
        "overlap_policy": {"inter_sleeve_positions_allowed": True, "near_6h_overlap_disclosure_only": True, "exact_schedule_aliases_canonicalized": True, "intra_sleeve_overlap_forbidden": True},
        "historical_novelty_inventory": {
            "path": str(HISTORICAL_INVENTORY),
            "sha256": sha256_file(HISTORICAL_INVENTORY),
            "manifest_hash": load_json(HISTORICAL_INVENTORY)["manifest_hash"],
        },
        "precanonical_schedule_count": len(pre), "canonical_sleeve_count": len(sleeves), "cross_universe_aliases": cross_aliases,
        "sleeves": sleeves,
        "evidence_boundary": {"clock_rows_opened": True, "market_rows_opened": 0, "funding_rows_opened": 0, "economic_outcomes_opened": False, "december_holdout_outcomes_opened": False, "oos_outcomes_opened": False},
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--active-clock-dir", type=Path, default=ACTIVE_CLOCK_DIR)
    parser.add_argument("--freeze-historical-inventory", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze_historical_inventory:
        result = write_historical_inventory()
        print(json.dumps({"records": len(result["records"]), "output": str(HISTORICAL_INVENTORY)}))
        return 0
    result = build(args.output, args.active_clock_dir)
    print(json.dumps({"precanonical": result["precanonical_schedule_count"], "canonical": result["canonical_sleeve_count"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
