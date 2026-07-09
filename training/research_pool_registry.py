"""Small CLI for inspecting research pool registries.

Examples:
    python -m training.research_pool_registry summary
    python -m training.research_pool_registry list alpha --status candidate
    python -m training.research_pool_registry show alpha btc_only_vwap_funding_asia
    python -m training.research_pool_registry recipe alpha btc_only_vwap_funding_asia
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
import re
from pathlib import Path
from typing import Any

POOL_DIR = Path("research/pools")
POOL_FILES = {
    "feature": POOL_DIR / "feature_pool.json",
    "alpha_feature": POOL_DIR / "alpha_feature_pool.json",
    "beta_feature": POOL_DIR / "beta_feature_pool.json",
    "gamma_feature": POOL_DIR / "gamma_feature_pool.json",
    "alpha": POOL_DIR / "alpha_pool.json",
    "portfolio": POOL_DIR / "portfolio_pool.json",
}

RECIPE_KEYS = {
    "feature": "generation_recipe",
    "alpha_feature": "generation_recipe",
    "beta_feature": "generation_recipe",
    "gamma_feature": "generation_recipe",
    "alpha": "usage_recipe",
    "portfolio": "construction_recipe",
}

HISTORY_FILE = POOL_DIR / "history_inventory.json"


def load_history() -> dict[str, Any]:
    with HISTORY_FILE.open() as f:
        return json.load(f)


def load_pool(kind: str) -> dict[str, Any]:
    path = POOL_FILES[kind]
    with path.open() as f:
        return json.load(f)


def iter_entries(kind: str | None = None):
    kinds = [kind] if kind else list(POOL_FILES)
    for k in kinds:
        pool = load_pool(k)
        for entry in pool.get("entries", []):
            yield k, entry


def cmd_summary(_: argparse.Namespace) -> None:
    for kind in POOL_FILES:
        entries = [e for _, e in iter_entries(kind)]
        by_status = Counter(e.get("status", "unknown") for e in entries)
        by_scope = Counter(e.get("scope", "unknown") for e in entries)
        by_tier = Counter(e.get("feature_tier", "") for e in entries if e.get("feature_tier"))
        print(f"{kind}: {len(entries)} entries")
        print("  status:", ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
        print("  scope:", ", ".join(f"{k}={v}" for k, v in sorted(by_scope.items())))
        if by_tier:
            print("  feature_tier:", ", ".join(f"{k}={v}" for k, v in sorted(by_tier.items())))


def cmd_list(args: argparse.Namespace) -> None:
    for kind, entry in iter_entries(args.kind):
        if args.status and entry.get("status") != args.status:
            continue
        if args.scope and entry.get("scope") != args.scope:
            continue
        if getattr(args, "tier", None) and entry.get("feature_tier") != args.tier:
            continue
        tier = f"/{entry.get('feature_tier')}" if entry.get("feature_tier") else ""
        print(f"{kind:13s} {entry.get('status',''):10s} {entry.get('id','')}{tier} :: {entry.get('name','')}")


def find_entry(kind: str, entry_id: str) -> dict[str, Any]:
    for _, entry in iter_entries(kind):
        if entry.get("id") == entry_id:
            return entry
    raise SystemExit(f"not found: {kind}/{entry_id}")


def cmd_show(args: argparse.Namespace) -> None:
    print(json.dumps(find_entry(args.kind, args.id), indent=2, ensure_ascii=False))


def cmd_recipe(args: argparse.Namespace) -> None:
    entry = find_entry(args.kind, args.id)
    recipe_key = RECIPE_KEYS[args.kind]
    recipe = entry.get(recipe_key)
    if not recipe:
        raise SystemExit(f"recipe missing: {args.kind}/{args.id} expected {recipe_key}")
    print(json.dumps(recipe, indent=2, ensure_ascii=False))


def cmd_sources(args: argparse.Namespace) -> None:
    seen: set[str] = set()
    for kind, entry in iter_entries(args.kind):
        if args.status and entry.get("status") != args.status:
            continue
        for src in entry.get("source_artifacts", []):
            if src not in seen:
                seen.add(src)
                print(src)


def cmd_history_summary(_: argparse.Namespace) -> None:
    history = load_history()
    print(json.dumps(history.get("summary", {}), indent=2, ensure_ascii=False))


def cmd_history_search(args: argparse.Namespace) -> None:
    history = load_history()
    q = re.compile(args.query, re.I)
    rows = []
    for section in ("artifacts", "history_hits", "git_commits"):
        for row in history.get(section, []):
            blob = json.dumps(row, ensure_ascii=False)
            if q.search(blob):
                rows.append((section, row))
    for section, row in rows[: args.limit]:
        if section == "git_commits":
            print(f"git {row.get('hash')} {row.get('date')} :: {row.get('subject')}")
        elif section == "history_hits":
            print(f"history {row.get('path')}:{row.get('line')} :: {row.get('preview','')[:180]}")
        else:
            print(f"artifact {row.get('source_type')} {row.get('path')} :: {row.get('title')}")
    if len(rows) > args.limit:
        print(f"... {len(rows) - args.limit} more matches (increase --limit)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("summary", help="show counts by pool/status/scope")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("list", help="list entries")
    p.add_argument("kind", choices=POOL_FILES)
    p.add_argument("--status", choices=["live", "promoted", "candidate", "weak", "rejected", "archived"])
    p.add_argument("--scope")
    p.add_argument("--tier", choices=["alpha_feature", "beta_feature", "gamma_feature"], help="filter unified feature pool by tier")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="show one entry as JSON")
    p.add_argument("kind", choices=POOL_FILES)
    p.add_argument("id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("recipe", help="show the reproducibility recipe for one entry")
    p.add_argument("kind", choices=POOL_FILES)
    p.add_argument("id")
    p.set_defaults(func=cmd_recipe)

    p = sub.add_parser("sources", help="list source artifacts referenced by entries")
    p.add_argument("kind", choices=POOL_FILES, nargs="?")
    p.add_argument("--status", choices=["live", "promoted", "candidate", "weak", "rejected", "archived"])
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("history-summary", help="show full alpha/feature history inventory counts")
    p.set_defaults(func=cmd_history_summary)

    p = sub.add_parser("history-search", help="search full alpha/feature history inventory")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_history_search)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
