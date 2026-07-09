"""Build an exhaustive alpha/feature research history inventory.

The output is intentionally provenance-heavy: it indexes repository docs,
results, scripts, OMX session/turn history, exports, notepad, and git commit
messages so future alpha searches can start from a classified map instead of
re-reading only recent artifacts.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(".")
OUT_JSON = Path("research/pools/history_inventory.json")
OUT_MD = Path("research/pools/history_inventory.md")

KEYWORDS = [
    "alpha", "feature", "portfolio", "candidate", "scan", "backtest", "strategy",
    "regime", "policy", "selector", "rule", "signal", "setup", "edge", "gate", "veto",
    "rex", "pb30", "funding", "premium", "basis", "oi", "open_interest", "taker",
    "volume", "upbit", "kimchi", "dxy", "fx", "macro", "wave", "vpin", "alpha101",
    "price-action", "price_action", "path-shape", "path_shape", "episode", "sparse",
    "gemma", "sft", "dpo", "pairwise", "ranker", "llm", "risk", "mdd", "cagr",
]
KEY_RE = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.I)

FAMILY_PATTERNS: list[tuple[str, list[str]]] = [
    ("live_pb30_activity_flow", ["pb30", "activity_flow"]),
    ("rex_regime_rule_selector", ["rex", "regime"]),
    ("bearish_short_rex", ["bear", "short", "rex"]),
    ("oi_taker_flow", ["oi", "taker"]),
    ("funding_premium_basis", ["funding", "premium", "basis"]),
    ("kimchi_dxy_macro", ["kimchi", "dxy", "macro", "fx", "usdkrw"]),
    ("wave_structure", ["wave", "path-shape", "path_shape"]),
    ("volume_upbit", ["volume", "upbit"]),
    ("vpin_microstructure", ["vpin"]),
    ("alpha101_formulaic", ["alpha101"]),
    ("portfolio_combination", ["portfolio", "gross", "sleeve", "combo", "weight"]),
    ("dynamic_exit", ["dynamic-exit", "dynamic_exit", "exit"]),
    ("price_action_episode_sparse", ["price-action", "price_action", "episode", "sparse", "setup"]),
    ("llm_selector_policy", ["gemma", "sft", "dpo", "pairwise", "ranker", "llm", "policy"]),
    ("strict_backtest_risk", ["strict", "mdd", "cagr", "risk"]),
]

CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
    ("portfolio_pool_history", ["portfolio", "gross", "sleeve", "combo", "weight", "live"]),
    ("alpha_pool_history", ["alpha", "strategy", "backtest", "candidate", "rule", "setup", "edge"]),
    ("feature_pool_history", ["feature", "wave", "vpin", "alpha101", "volume", "oi", "taker", "funding", "premium", "kimchi", "dxy", "macro", "path-shape", "price-action"]),
    ("llm_selector_history", ["gemma", "sft", "dpo", "pairwise", "ranker", "llm", "policy", "rex-selector"]),
    ("failure_guardrail_history", ["failure", "rejected", "no-go", "negative", "collapse", "overfit", "inversion", "guardrail"]),
]

TIER_RULES = {
    "alpha_feature": ["pb30", "activity_flow", "funding", "premium", "basis", "oi", "taker", "rex"],
    "beta_feature": ["wave", "volume", "upbit", "kimchi", "dxy", "macro", "vpin", "alpha101", "path-shape", "price-action", "episode", "sparse"],
    "gamma_feature": ["failure", "rejected", "no-go", "negative", "collapse", "overfit", "inversion"],
}


def read_text(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(errors="ignore")[:limit]
    except Exception:
        return ""


def head_title(path: Path, text: str) -> str:
    for line in text.splitlines()[:30]:
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return path.stem.replace("-", " ").replace("_", " ")


def classify(blob: str) -> tuple[list[str], list[str], str]:
    low = blob.lower()
    categories = [cat for cat, pats in CATEGORY_PATTERNS if any(p in low for p in pats)]
    families = [fam for fam, pats in FAMILY_PATTERNS if any(p in low for p in pats)]
    if not categories:
        categories = ["context_history"]
    if not families:
        families = ["general_research_context"]
    # Conservative tier suggestion: gamma only if explicit failure language exists.
    if any(p in low for p in TIER_RULES["gamma_feature"]):
        suggested_tier = "gamma_usage_or_failure_review"
    elif any(p in low for p in TIER_RULES["alpha_feature"]):
        suggested_tier = "alpha_or_beta_feature_review"
    elif any(p in low for p in TIER_RULES["beta_feature"]):
        suggested_tier = "beta_feature_review"
    else:
        suggested_tier = "unclassified_review"
    return categories, families, suggested_tier


def matching_files(base: Path, glob: str, max_depth: int | None = None) -> list[Path]:
    out = []
    for p in base.glob(glob):
        if p.is_file():
            out.append(p)
    return sorted(out)


def add_artifact(entries: list[dict[str, Any]], path: Path, source_type: str) -> None:
    text = read_text(path)
    hay = f"{path.as_posix()}\n{text[:5000]}"
    if source_type in {"docs", "exports", "notepad", "plans", "wiki"}:
        relevant = bool(KEY_RE.search(hay))
    else:
        relevant = bool(KEY_RE.search(path.as_posix()))
    if not relevant:
        return
    categories, families, tier = classify(hay)
    entries.append({
        "id": re.sub(r"[^a-z0-9]+", "_", path.as_posix().lower()).strip("_"),
        "source_type": source_type,
        "path": path.as_posix(),
        "title": head_title(path, text),
        "categories": categories,
        "families": families,
        "suggested_review_tier": tier,
        "matched_keywords": sorted(set(m.group(0).lower() for m in KEY_RE.finditer(hay)))[:25],
    })


def collect_artifacts() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for p in sorted(Path("docs").glob("*.md")):
        add_artifact(entries, p, "docs")
    for p in sorted(Path("training").glob("*.py")):
        add_artifact(entries, p, "training_script")
    for p in sorted(Path("configs/live").glob("*.json")):
        add_artifact(entries, p, "live_config")
    for p in sorted(Path("research/pools").glob("*.json")):
        add_artifact(entries, p, "research_pool")
    for p in sorted(Path("results").glob("*")):
        if p.is_file() and KEY_RE.search(p.name):
            add_artifact(entries, p, "result_file")
        elif p.is_dir() and KEY_RE.search(p.name):
            # Directory-level result group; do not descend into tens of thousands of files.
            categories, families, tier = classify(p.as_posix())
            entries.append({
                "id": re.sub(r"[^a-z0-9]+", "_", p.as_posix().lower()).strip("_"),
                "source_type": "result_dir",
                "path": p.as_posix(),
                "title": p.name,
                "categories": categories,
                "families": families,
                "suggested_review_tier": tier,
                "matched_keywords": sorted(set(m.group(0).lower() for m in KEY_RE.finditer(p.as_posix()))),
            })
    for p in sorted(Path(".omx/exports").glob("*.md")):
        add_artifact(entries, p, "exports")
    for p in [Path(".omx/notepad.md"), Path(".omx/project-memory.json"), Path(".omx/plans/long-strategy-2025-eval-plan.md")]:
        if p.exists():
            add_artifact(entries, p, "omx_state")
    for p in sorted(Path("omx_wiki").glob("*.md")):
        add_artifact(entries, p, "wiki")
    return entries


def iter_history_files() -> list[Path]:
    candidates: list[Path] = []
    roots = [Path(".omx/logs"), Path(".omx/runtime"), Path(".codex")]
    for root in roots:
        if not root.exists():
            continue
        for pat in ["**/history.jsonl", "**/session-history.jsonl", "**/turns-*.jsonl", "**/omx-*.jsonl", "**/*.log"]:
            candidates.extend(p for p in root.glob(pat) if p.is_file())
    return sorted(set(candidates))


def collect_history_hits() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in iter_history_files():
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if not KEY_RE.search(line):
                continue
            categories, families, tier = classify(line)
            # Store previews only; this is an index, not a full session export copy.
            hits.append({
                "source_type": "omx_history_line",
                "path": path.as_posix(),
                "line": i,
                "categories": categories,
                "families": families,
                "suggested_review_tier": tier,
                "matched_keywords": sorted(set(m.group(0).lower() for m in KEY_RE.finditer(line)))[:20],
                "preview": line[:600],
            })
    return hits


def git_commits() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            ["git", "log", "--all", "--date=short", "--pretty=format:%h%x09%ad%x09%s"],
            text=True,
            errors="ignore",
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        if not KEY_RE.search(line):
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        h, d, s = parts
        categories, families, tier = classify(s)
        rows.append({
            "hash": h,
            "date": d,
            "subject": s,
            "categories": categories,
            "families": families,
            "suggested_review_tier": tier,
            "matched_keywords": sorted(set(m.group(0).lower() for m in KEY_RE.finditer(s))),
        })
    return rows


def summarize(entries: list[dict[str, Any]], hits: list[dict[str, Any]], commits: list[dict[str, Any]]) -> dict[str, Any]:
    def counts_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        c = Counter()
        for r in rows:
            val = r.get(key)
            if isinstance(val, list):
                c.update(val)
            elif val:
                c[val] += 1
        return dict(sorted(c.items()))

    history_files = iter_history_files()
    return {
        "artifact_count": len(entries),
        "history_file_count_scanned": len(history_files),
        "history_hit_count": len(hits),
        "git_commit_hit_count": len(commits),
        "artifacts_by_source_type": counts_by(entries, "source_type"),
        "artifacts_by_category": counts_by(entries, "categories"),
        "artifacts_by_family": counts_by(entries, "families"),
        "history_hits_by_family": counts_by(hits, "families"),
        "git_hits_by_family": counts_by(commits, "families"),
    }


def write_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    entries = payload["artifacts"]
    hits = payload["history_hits"]
    commits = payload["git_commits"]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        for fam in e["families"]:
            by_family[fam].append(e)

    lines: list[str] = []
    lines.append("# Alpha / Feature Full History Inventory")
    lines.append("")
    lines.append(f"Generated: `{payload['generated_at']}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Indexed artifacts: `{summary['artifact_count']}`")
    lines.append(f"- OMX/Codex history files scanned: `{summary['history_file_count_scanned']}`")
    lines.append(f"- OMX/Codex keyword hits: `{summary['history_hit_count']}`")
    lines.append(f"- Git commit hits: `{summary['git_commit_hit_count']}`")
    lines.append("- Sources: `docs/*.md`, `training/*.py`, top-level `results/*` files/dirs, `.omx/exports`, `.omx/notepad.md`, `.omx/project-memory.json`, `.omx/plans`, `omx_wiki`, `.omx/logs`, `.omx/runtime/**/history.jsonl`, `.codex/**/history.jsonl`, and `git log --all`.")
    lines.append("")
    lines.append("## Classification rule")
    lines.append("")
    lines.append("This file is an exhaustive provenance index, not an automatic promotion engine. `gamma_usage_or_failure_review` means the artifact contains failure/no-go/overfit language and must be reviewed under the strict gamma rule; it does **not** automatically demote a feature family to gamma. Gamma demotion still requires strong noise-only/invalid/repeated-failure evidence.")
    lines.append("")
    lines.append("## Counts by family")
    lines.append("")
    for fam, count in sorted(summary["artifacts_by_family"].items()):
        lines.append(f"- `{fam}`: {count}")
    lines.append("")
    lines.append("## Exhaustive artifact index by family")
    lines.append("")
    for fam in sorted(by_family):
        lines.append(f"### {fam}")
        lines.append("")
        for e in sorted(by_family[fam], key=lambda x: (x["source_type"], x["path"])):
            cats = ",".join(e["categories"])
            kws = ",".join(e.get("matched_keywords", [])[:8])
            lines.append(f"- `{e['source_type']}` `{e['path']}` — {e['title']} | cats={cats} | review={e['suggested_review_tier']} | kw={kws}")
        lines.append("")
    lines.append("## OMX/Codex history hit files")
    lines.append("")
    hc = Counter(h["path"] for h in hits)
    for path, count in sorted(hc.items()):
        lines.append(f"- `{path}`: {count} hits")
    lines.append("")
    lines.append("## Git commit alpha/feature trail")
    lines.append("")
    for c in commits:
        fams = ",".join(c["families"])
        lines.append(f"- `{c['hash']}` {c['date']} — {c['subject']} | families={fams}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=str(OUT_JSON))
    parser.add_argument("--markdown", default=str(OUT_MD))
    args = parser.parse_args()
    globals()["OUT_JSON"] = Path(args.json)
    globals()["OUT_MD"] = Path(args.markdown)
    artifacts = collect_artifacts()
    history_hits = collect_history_hits()
    commits = git_commits()
    payload = {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "protocol": {
            "purpose": "Exhaustive alpha/feature research provenance index across repo, OMX history, and git history.",
            "gamma_caution": "Do not demote to gamma from weak results alone; require strong noise-only/invalid/repeated-failure evidence.",
            "trading_target_asset": "BTCUSDT; external/cross-asset inputs are features only.",
        },
        "summary": summarize(artifacts, history_hits, commits),
        "artifacts": artifacts,
        "history_hits": history_hits,
        "git_commits": commits,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    write_markdown(payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
