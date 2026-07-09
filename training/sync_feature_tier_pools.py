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


def materialize() -> dict[str, Any]:
    data = json.loads(SOURCE.read_text())
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
