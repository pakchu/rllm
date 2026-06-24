"""Filter feature_snapshot keys in event candidate JSONL rows."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilterCandidateFeaturesCfg:
    input_jsonl: str
    output_jsonl: str
    keep_prefixes: str = ""
    keep_features: str = ""
    drop_prefixes: str = ""


def _items(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(raw).split(",") if x.strip())


def _keep(key: str, keep_prefixes: tuple[str, ...], keep_features: set[str], drop_prefixes: tuple[str, ...]) -> bool:
    if drop_prefixes and key.startswith(drop_prefixes):
        return False
    if keep_features or keep_prefixes:
        return key in keep_features or key.startswith(keep_prefixes)
    return True


def run(cfg: FilterCandidateFeaturesCfg) -> dict[str, object]:
    keep_prefixes = _items(cfg.keep_prefixes)
    keep_features = set(_items(cfg.keep_features))
    drop_prefixes = _items(cfg.drop_prefixes)
    rows = []
    total = 0
    kept_counts = []
    for line in Path(cfg.input_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        snap = row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {}
        row["feature_snapshot"] = {k: v for k, v in snap.items() if _keep(str(k), keep_prefixes, keep_features, drop_prefixes)}
        kept_counts.append(len(row["feature_snapshot"]))
        rows.append(row)
    out = Path(cfg.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    return {"config": asdict(cfg), "rows": total, "feature_count_min": min(kept_counts) if kept_counts else 0, "feature_count_max": max(kept_counts) if kept_counts else 0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter event candidate feature_snapshot keys")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--keep-prefixes", default="")
    p.add_argument("--keep-features", default="")
    p.add_argument("--drop-prefixes", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(FilterCandidateFeaturesCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
