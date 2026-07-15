"""Materialize alpha/beta/gamma feature pool files from feature_pool.json.

The unified `feature_pool.json` is the edit source of truth.  This script writes
three tier-specific views so research can browse narrower pools without losing
one canonical registry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

POOL_DIR = Path("research/pools")
SOURCE = POOL_DIR / "feature_pool.json"
TIER_FILES = {
    "alpha_feature": POOL_DIR / "alpha_feature_pool.json",
    "beta_feature": POOL_DIR / "beta_feature_pool.json",
    "gamma_feature": POOL_DIR / "gamma_feature_pool.json",
}


def orphaned_view_entries(data: dict[str, Any]) -> dict[str, list[str]]:
    """Return tier-view entries missing from the canonical source registry.

    Tier files are generated views, but older research occasionally edited a
    view directly.  Silently rewriting such a view would delete that history.
    Fail closed until the entries have been migrated into ``feature_pool``.
    """

    source_ids = {entry.get("id") for entry in data.get("entries", [])}
    orphaned: dict[str, list[str]] = {}
    for tier, path in TIER_FILES.items():
        if not path.exists():
            continue
        current = json.loads(path.read_text())
        missing = sorted(
            entry.get("id", "<missing-id>")
            for entry in current.get("entries", [])
            if entry.get("id") not in source_ids
        )
        if missing:
            orphaned[tier] = missing
    return orphaned


def materialize() -> dict[str, Any]:
    data = json.loads(SOURCE.read_text())
    orphaned = orphaned_view_entries(data)
    if orphaned:
        details = "; ".join(f"{tier}: {', '.join(ids)}" for tier, ids in orphaned.items())
        raise ValueError(
            "tier views contain entries missing from research/pools/feature_pool.json; "
            f"migrate them before synchronization ({details})"
        )
    base = {k: v for k, v in data.items() if k != "entries"}
    base["source_pool"] = SOURCE.name
    counts = {}
    for tier, path in TIER_FILES.items():
        out = dict(base)
        out["pool_kind"] = tier
        out["entries"] = [e for e in data.get("entries", []) if e.get("feature_tier") == tier]
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        counts[tier] = len(out["entries"])
    missing = [e.get("id", "<missing-id>") for e in data.get("entries", []) if not e.get("feature_tier")]
    return {"source": str(SOURCE), "outputs": {k: str(v) for k, v in TIER_FILES.items()}, "counts": counts, "missing_feature_tier": missing}


def main() -> None:
    print(json.dumps(materialize(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
