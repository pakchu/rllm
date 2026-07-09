"""Build a source-by-source inventory of features used in alpha research scripts.

This is intentionally conservative and reproducible: it scans known alpha-search
and portfolio-search scripts for dataframe feature writes, gate tuples, feature
lists, and common feature-like string literals.  The output is a registry aid,
not executable strategy logic.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable

OUT = Path("research/pools/feature_inventory.json")

SOURCE_PATTERNS = [
    "training/search_*alpha*.py",
    "training/search_*portfolio*.py",
    "training/search_*combo*.py",
    "training/search_*pool*.py",
    "training/search_btc_only_quantile_standalone.py",
    "training/search_cross_asset_quantile_standalone.py",
    "training/search_dynamic_exit*.py",
    "training/evaluate_*alpha*.py",
    "training/evaluate_*portfolio*.py",
    "training/evaluate_*wave*.py",
    "training/evaluate_oi_llm_selector.py",
    "training/portfolio_with_dynamic_exit_sleeves.py",
    "training/scan_linear_combo_overlay.py",
    "training/alpha_linear_combo_scan.py",
    "training/long_regime_combo_scan.py",
    "training/price_action_combo_scan.py",
    "training/rolling_price_action_combo_scan.py",
    "training/rex_*validation.py",
    "training/rex_horizon_sweep.py",
    "training/wave_feature_ridge_policy.py",
    "training/*wave*threshold*.py",
    "training/*wave*validation*.py",
    "training/build_linear_alpha_meta_sft.py",
    "training/build_rex_candidate_ranker_records.py",
    "training/build_rex_listwise_choice_records.py",
    "training/event_action_policy_data.py",
    "training/vlm_trading_data.py",
]
CACHE_FILES = [
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz",
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
]

FEATURE_KEYWORDS = (
    "ret", "z", "zscore", "momentum", "kimchi", "dxy", "usdkrw", "btckrw", "premium", "funding",
    "oi", "taker", "volume", "qv", "vwap", "range", "rvol", "sma", "rsi", "bb", "trend", "htf",
    "wave", "rex", "pb30", "activity", "flow", "pos", "rank", "corr", "spread", "vpin", "clv",
    "intr", "gap", "session", "macro", "risk", "pressure", "upbit", "basis", "cvd", "alpha", "score",
)
EXCLUDE = {
    "date", "open", "high", "low", "close", "symbol", "side", "long", "short", "train", "test", "eval",
    "output", "doc", "name", "status", "family", "scope", "feature", "threshold", "terms", "stats", "top",
    "json", "csv", "gzip", "true", "false", "none", "all", "base", "config", "result", "results", "docs",
}


def looks_feature(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    if len(s) > 80:
        return False
    if s.startswith("_") or "." in s:
        return False
    if s.lower() in EXCLUDE:
        return False
    if "/" in s or " " in s or "{" in s or "}" in s:
        return False
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_*-]*$", s):
        return False
    sl = s.lower()
    return any(k in sl for k in FEATURE_KEYWORDS)


def is_semantic_token(s: str) -> bool:
    return bool(isinstance(s, str) and re.match(r"^[A-Z][A-Z0-9_]{2,}$", s) and any(k.upper() in s for k in FEATURE_KEYWORDS))


def string_literals(source: str) -> Iterable[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def regex_features(source: str) -> tuple[set[str], set[str]]:
    feats: set[str] = set()
    tokens: set[str] = set()
    patterns = [
        r"(?:f|feat|out|df|market|m)\[['\"]([^'\"]+)['\"]\]",
        r"\('([^']+)'\s*,\s*'[<>]=?'",
        r"\[\s*'([^']+)'\s*,\s*'[<>]=?'",
        r'"feature"\s*:\s*"([^"]+)"',
    ]
    for pat in patterns:
        for m in re.finditer(pat, source):
            val = m.group(1)
            if is_semantic_token(val):
                tokens.add(val)
            elif looks_feature(val):
                feats.add(val)
    for lit in string_literals(source):
        if is_semantic_token(lit):
            tokens.add(lit)
        elif looks_feature(lit):
            feats.add(lit)
        # Parse comma-separated feature lists embedded in args/default strings.
        if "," in lit and len(lit) < 500:
            for part in lit.split(","):
                part = part.strip().strip("'\"")
                if is_semantic_token(part):
                    tokens.add(part)
                elif looks_feature(part):
                    feats.add(part)
    return feats, tokens


def source_files() -> list[Path]:
    files: set[Path] = set()
    for pat in SOURCE_PATTERNS:
        files.update(Path().glob(pat))
    return sorted(p for p in files if p.exists() and p.is_file())


def cache_columns() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    try:
        import pandas as pd
    except Exception:
        return out
    for c in CACHE_FILES:
        p = Path(c)
        if not p.exists():
            continue
        try:
            cols = list(pd.read_csv(p, nrows=1).columns)
            out[c] = [x for x in cols if looks_feature(x) or x in {"open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"}]
        except Exception:
            pass
    return out


def main() -> None:
    by_source: dict[str, list[str]] = {}
    tokens_by_source: dict[str, list[str]] = {}
    union: set[str] = set()
    token_union: set[str] = set()
    for path in source_files():
        feats, toks = regex_features(path.read_text(errors="ignore"))
        if feats:
            by_source[str(path)] = sorted(feats)
            union.update(feats)
        if toks:
            tokens_by_source[str(path)] = sorted(toks)
            token_union.update(toks)
    caches = cache_columns()
    for src, cols in caches.items():
        by_source[src] = sorted(set(cols))
        union.update(cols)
    report = {
        "schema_version": 1,
        "updated_at": "2026-07-10",
        "description": "Feature inventory extracted from alpha/portfolio research scripts and market caches. Trading target remains BTC unless explicitly stated; external variables are feature inputs.",
        "source_count": len(by_source),
        "feature_count": len(union),
        "features": sorted(union),
        "features_by_source": by_source,
        "semantic_token_count": len(token_union),
        "semantic_tokens": sorted(token_union),
        "semantic_tokens_by_source": tokens_by_source,
        "notes": [
            "Generated by training/build_research_feature_inventory.py.",
            "Regex/AST extraction may include a few non-feature tokens and miss dynamically constructed names; source_by_source provenance is kept for review.",
            "Use research/pools/feature_pool.json for curated status/evidence and this file for exhaustive recall."
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "source_count": len(by_source), "feature_count": len(union), "semantic_token_count": len(token_union)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
