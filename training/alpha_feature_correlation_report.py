"""Build correlation statistics and heatmaps for alpha feature families.

The report focuses on feature_pool entries tagged alpha_feature plus explicit
alpha component masks used by current long/short candidates.  It records missing
pool features separately so absent sources (for example OI in a cache without OI)
do not silently disappear.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from preprocessing.market_features import build_market_feature_frame
from training.long_component_tp_union_scan import COMPONENTS as LONG_COMPONENTS, _component_mask
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features


@dataclass(frozen=True)
class AlphaFeatureCorrelationConfig(LongComboScanConfig):
    output_dir: str = "results/alpha_feature_correlation_2026-07-10"
    docs_output: str = "docs/alpha-feature-correlation-2026-07-10.md"
    feature_pool: str = "research/pools/feature_pool.json"
    alpha_pool: str = "research/pools/alpha_pool.json"
    start: str = "2024-01-01"
    end: str = "2026-06-02"
    exclude_from: str = "2026-06-02"
    max_heatmap_features: int = 80


# Explicit masks for alpha entries whose components are not defined in the long
# component scan.  These thresholds are copied from alpha_pool evidence entries.
EXTRA_COMPONENTS: dict[str, list[tuple[str, str, float]]] = {
    "short_fx_stress": [("htf_3d_return_1", "le", -0.0325294973), ("usdkrw_zscore", "ge", 1.3870063775)],
    "short_premium_panic": [("htf_3d_range_pos", "le", -0.5114186851), ("premium_index_zscore", "le", -1.47209312)],
    "short_kimchi_unwind": [("htf_3d_return_1", "le", -0.0303196833), ("kimchi_premium_change", "le", -0.0046123752)],
}

COMPONENT_GROUPS: dict[str, list[str]] = {
    "long_range_funding_premium": ["range_bb90", "funding10_trend70", "premium20_mom90"],
    "long_funding_compression_premium": ["funding10_trend70", "compress05_trend80", "premium20_mom90"],
    "long_range_funding_compression": ["range_bb90", "funding10_trend70", "compress05_trend80"],
    "long_minimal_funding_premium": ["funding10_trend70", "premium20_mom90"],
    "short_premium_kimchi_union": ["short_premium_panic", "short_kimchi_unwind"],
}


def _zscore(series: pd.Series, window: int) -> pd.Series:
    window = max(2, int(window))
    mean = series.rolling(window, min_periods=max(10, window // 5)).mean()
    std = series.rolling(window, min_periods=max(10, window // 5)).std(ddof=0)
    return ((series - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(-8.0, 8.0).fillna(0.0)


def _add_oi_derived_features(market: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Add OI aliases/divergence features used by older alpha pool entries."""
    out = features.copy()
    if "funding_zscore" in out.columns and "funding_z" not in out.columns:
        out["funding_z"] = out["funding_zscore"]
    if "premium_index_zscore" in out.columns and "premium_z" not in out.columns:
        out["premium_z"] = out["premium_index_zscore"]
    if "open_interest" not in market.columns:
        return out
    close = pd.to_numeric(market["close"], errors="coerce")
    oi = pd.to_numeric(market["open_interest"], errors="coerce").replace(0.0, np.nan).ffill()
    for bars, name in [(24, "2h"), (48, "4h"), (72, "6h"), (96, "8h"), (144, "12h"), (288, "1d")]:
        oi_ret = np.log(oi / oi.shift(bars).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        px_ret = np.log(close / close.shift(bars).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        div = oi_ret - px_ret
        out[f"oi_ret_z_{bars}"] = _zscore(oi_ret, max(288, bars * 4))
        out[f"px_ret_z_{bars}"] = _zscore(px_ret, max(288, bars * 4))
        out[f"oi_minus_px_z_{bars}"] = _zscore(div, max(288, bars * 4))
        out[f"px_minus_oi_z_{bars}"] = _zscore(-div, max(288, bars * 4))
        out[f"oi_ret_{name}_z"] = out[f"oi_ret_z_{bars}"]
        out[f"px_ret_{name}_z"] = out[f"px_ret_z_{bars}"]
        out[f"oi_minus_px_{name}_z"] = out[f"oi_minus_px_z_{bars}"]
        out[f"px_minus_oi_{name}_z"] = out[f"px_minus_oi_z_{bars}"]
    out["oi_available"] = pd.to_numeric(market.get("open_interest_available", pd.Series(1.0, index=market.index)), errors="coerce").fillna(0.0)
    out["btc_oi_unwind_long"] = (-out["oi_ret_z_72"]).clip(lower=0) + out.get("px_ret_z_72", 0).clip(lower=0)
    out["btc_oi_squeeze_short"] = out["oi_ret_z_72"].clip(lower=0) + (-out.get("px_ret_z_72", 0)).clip(lower=0)
    out["btc_liq_revert_long"] = (-out.get("px_ret_z_72", 0)).clip(lower=0) + (-out["oi_ret_z_72"]).clip(lower=0)
    taker = out.get("taker_imbalance", pd.Series(0.0, index=out.index))
    out["btc_cvd_absorb_long"] = (-out.get("px_ret_z_72", 0)).clip(lower=0) + taker.clip(lower=0)
    out["btc_overheat_short"] = out.get("px_ret_z_72", 0).clip(lower=0) + out["oi_ret_z_72"].clip(lower=0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _cluster_order(corr: pd.DataFrame) -> list[str]:
    if len(corr) <= 2:
        return list(corr.columns)
    mat = corr.fillna(0.0).to_numpy(float)
    dist = 1.0 - np.abs(mat)
    np.fill_diagonal(dist, 0.0)
    try:
        link = linkage(squareform(dist, checks=False), method="average")
        return list(corr.columns[leaves_list(link)])
    except Exception:
        return list(corr.columns)


def _save_heatmap(corr: pd.DataFrame, path: Path, *, title: str, max_features: int) -> list[str]:
    order = _cluster_order(corr)
    if len(order) > int(max_features):
        # Keep the features with the strongest average absolute relationship so
        # the heatmap remains readable while CSV keeps the full matrix.
        strength = corr.abs().replace(1.0, np.nan).mean(axis=1).sort_values(ascending=False)
        keep = set(strength.head(int(max_features)).index)
        order = [x for x in order if x in keep]
    view = corr.loc[order, order]
    fig_w = max(8, min(28, len(order) * 0.34))
    fig_h = max(7, min(26, len(order) * 0.32))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    im = ax.imshow(view.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticklabels(order, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return order


def _load_pool_features(path: str, available_columns: list[str]) -> tuple[list[str], dict[str, Any]]:
    pool = json.load(open(path))
    wanted: list[str] = []
    unavailable: dict[str, list[str]] = {}
    available_set = set(available_columns)
    for entry in pool.get("entries", []):
        if entry.get("feature_tier") != "alpha_feature":
            continue
        entry_missing: list[str] = []
        for raw in entry.get("feature_names", []):
            if not isinstance(raw, str):
                continue
            token = raw.strip()
            if not token or token.startswith("configs/"):
                continue
            if "*" in token:
                matches = sorted(fnmatch.filter(available_columns, token))
                if not matches and token.endswith("_*"):
                    matches = sorted([c for c in available_columns if c.startswith(token[:-1])])
                if matches:
                    wanted.extend(matches)
                else:
                    entry_missing.append(token)
            elif token in available_set:
                wanted.append(token)
            else:
                entry_missing.append(token)
        if entry_missing:
            unavailable[entry.get("id", "unknown")] = entry_missing
    return sorted(dict.fromkeys(wanted)), {"unavailable_by_entry": unavailable}


def _mask_from_conditions(features: pd.DataFrame, conditions: list[tuple[str, str, float]]) -> np.ndarray:
    mask = np.ones(len(features), dtype=bool)
    for feature, op, threshold in conditions:
        if feature not in features.columns:
            return np.zeros(len(features), dtype=bool)
        x = features[feature].to_numpy(float)
        mask &= np.isfinite(x) & ((x <= threshold) if op == "le" else (x >= threshold))
    return mask


def _build_component_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for name in sorted(LONG_COMPONENTS):
        cols[name] = _component_mask(features, name).astype(float)
        metadata[name] = {"kind": "long_component", "conditions": LONG_COMPONENTS[name]}
    for name, conditions in EXTRA_COMPONENTS.items():
        cols[name] = _mask_from_conditions(features, conditions).astype(float)
        metadata[name] = {"kind": "extra_component", "conditions": conditions}
    for group, members in COMPONENT_GROUPS.items():
        active = np.zeros(len(features), dtype=bool)
        for member in members:
            if member in LONG_COMPONENTS:
                active |= _component_mask(features, member)
            elif member in EXTRA_COMPONENTS:
                active |= _mask_from_conditions(features, EXTRA_COMPONENTS[member])
        cols[group] = active.astype(float)
        metadata[group] = {"kind": "candidate_union", "members": members}
    return pd.DataFrame(cols, index=features.index), metadata


def _top_pairs(corr: pd.DataFrame, n: int = 30, *, absolute: bool = True) -> list[dict[str, Any]]:
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = float(corr.loc[a, b])
            if np.isfinite(v):
                rows.append({"a": a, "b": b, "corr": v, "abs_corr": abs(v)})
    rows.sort(key=lambda x: x["abs_corr"] if absolute else x["corr"], reverse=True)
    return rows[:n]


def _jaccard_frame(binary: pd.DataFrame) -> pd.DataFrame:
    arr = binary.to_numpy(bool)
    names = list(binary.columns)
    out = np.zeros((len(names), len(names)), dtype=float)
    for i in range(len(names)):
        ai = arr[:, i]
        for j in range(len(names)):
            bj = arr[:, j]
            union = np.logical_or(ai, bj).sum()
            out[i, j] = np.logical_and(ai, bj).sum() / union if union else 0.0
    return pd.DataFrame(out, index=names, columns=names)


def run(cfg: AlphaFeatureCorrelationConfig) -> dict[str, Any]:
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    market = _load_market(cfg)
    base = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = _add_oi_derived_features(market, features)
    dates = pd.to_datetime(market["date"])
    mask = _split_mask(dates, cfg.start, cfg.end)

    feature_names, pool_meta = _load_pool_features(cfg.feature_pool, list(features.columns))
    # Always include source columns referenced by component masks, even if pool
    # metadata missed them.
    for conds in list(LONG_COMPONENTS.values()) + list(EXTRA_COMPONENTS.values()):
        for col, _, _ in conds:
            if col in features.columns:
                feature_names.append(col)
    feature_names = sorted(dict.fromkeys(feature_names))

    continuous = features.loc[mask, feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Drop constants after window filtering.
    continuous = continuous.loc[:, continuous.std(ddof=0) > 1e-12]
    pearson = continuous.corr(method="pearson")
    spearman = continuous.corr(method="spearman")
    pearson.to_csv(outdir / "continuous_pearson.csv")
    spearman.to_csv(outdir / "continuous_spearman.csv")

    component_frame, component_meta = _build_component_frame(features)
    binary = component_frame.loc[mask]
    binary = binary.loc[:, binary.sum(axis=0) > 0]
    phi = binary.corr(method="pearson").fillna(0.0)
    jaccard = _jaccard_frame(binary)
    phi.to_csv(outdir / "component_phi.csv")
    jaccard.to_csv(outdir / "component_jaccard.csv")

    continuous_order = _save_heatmap(spearman, outdir / "continuous_spearman_heatmap.png", title="Alpha continuous features Spearman correlation", max_features=int(cfg.max_heatmap_features))
    component_order = _save_heatmap(phi, outdir / "component_phi_heatmap.png", title="Alpha component phi correlation", max_features=200)

    component_activity = {
        col: {"active_rows": int(binary[col].sum()), "active_frac": float(binary[col].mean())}
        for col in binary.columns
    }
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "analysis_window": {"start": cfg.start, "end": cfg.end, "rows": int(mask.sum())},
        "continuous_features": list(continuous.columns),
        "continuous_feature_count": int(len(continuous.columns)),
        "component_count": int(len(binary.columns)),
        "pool_feature_availability": pool_meta,
        "component_metadata": component_meta,
        "component_activity": component_activity,
        "top_abs_spearman_pairs": _top_pairs(spearman, 40),
        "top_abs_phi_pairs": _top_pairs(phi, 40),
        "top_jaccard_pairs": _top_pairs(jaccard, 40, absolute=False),
        "cluster_order": {"continuous_spearman": continuous_order, "component_phi": component_order},
        "artifacts": {
            "continuous_pearson_csv": str(outdir / "continuous_pearson.csv"),
            "continuous_spearman_csv": str(outdir / "continuous_spearman.csv"),
            "component_phi_csv": str(outdir / "component_phi.csv"),
            "component_jaccard_csv": str(outdir / "component_jaccard.csv"),
            "continuous_spearman_heatmap_png": str(outdir / "continuous_spearman_heatmap.png"),
            "component_phi_heatmap_png": str(outdir / "component_phi_heatmap.png"),
        },
        "leakage_guard": {
            "correlation_uses_features_at_or_before_row": True,
            "no_forward_returns_used": True,
            "market_rows_after_exclude_from_removed_before_feature_build": True,
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _write_doc(cfg, summary)
    return summary


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def _write_doc(cfg: AlphaFeatureCorrelationConfig, summary: dict[str, Any]) -> None:
    lines = [
        "# Alpha feature correlation report (2026-07-10)",
        "",
        f"Window: `{cfg.start}` ~ `{cfg.end}`; rows={summary['analysis_window']['rows']:,}.",
        "",
        "## Artifacts",
        "",
    ]
    for k, v in summary["artifacts"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines += ["", "## Continuous feature correlation", "", "Top absolute Spearman pairs:", "", "| rank | feature A | feature B | rho |", "|---:|---|---|---:|"]
    for i, row in enumerate(summary["top_abs_spearman_pairs"][:20], 1):
        lines.append(f"| {i} | `{row['a']}` | `{row['b']}` | {row['corr']:.4f} |")
    lines += ["", "## Component / candidate correlation", "", "Top phi pairs:", "", "| rank | component A | component B | phi |", "|---:|---|---|---:|"]
    for i, row in enumerate(summary["top_abs_phi_pairs"][:20], 1):
        lines.append(f"| {i} | `{row['a']}` | `{row['b']}` | {row['corr']:.4f} |")
    lines += ["", "Top Jaccard overlaps:", "", "| rank | component A | component B | Jaccard |", "|---:|---|---|---:|"]
    for i, row in enumerate(summary["top_jaccard_pairs"][:20], 1):
        if row["a"] == row["b"]:
            continue
        lines.append(f"| {i} | `{row['a']}` | `{row['b']}` | {row['corr']:.4f} |")
    lines += ["", "## Component activity", "", "| component | active rows | active frac |", "|---|---:|---:|"]
    for name, st in sorted(summary["component_activity"].items(), key=lambda kv: kv[1]["active_frac"], reverse=True):
        lines.append(f"| `{name}` | {st['active_rows']:,} | {_fmt_pct(st['active_frac'])} |")
    missing = summary["pool_feature_availability"].get("unavailable_by_entry", {})
    lines += ["", "## Missing pool features", ""]
    if missing:
        for entry, cols in missing.items():
            lines.append(f"- `{entry}`: {', '.join(f'`{c}`' for c in cols[:30])}{' ...' if len(cols) > 30 else ''}")
    else:
        lines.append("None.")
    oi_present = any("oi_" in name or "open_interest" in name for name in summary["continuous_features"])
    lines += ["", "## Interpretation", "", "- Long squeeze candidates are intentionally related; high phi/Jaccard confirms they should be treated as one family until marginal contribution tests prove otherwise.", "- Continuous feature heatmap should be used to remove redundant raw numeric tokens before feeding an LLM/RLLM state card."]
    if oi_present:
        lines.append("- OI-derived features are present in this run; remaining OI-context missing items are higher-level aliases/composites not generated by this report script.")
    else:
        lines.append("- Missing OI-derived features indicate the current cache used for this report does not expose those columns; use a DB/OI-enriched cache for a full derivatives-positioning correlation report.")
    Path(cfg.docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.docs_output).write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", default="")
    p.add_argument("--output-dir", default=AlphaFeatureCorrelationConfig.output_dir)
    p.add_argument("--docs-output", default=AlphaFeatureCorrelationConfig.docs_output)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    p.add_argument("--feature-pool", default=AlphaFeatureCorrelationConfig.feature_pool)
    p.add_argument("--alpha-pool", default=AlphaFeatureCorrelationConfig.alpha_pool)
    p.add_argument("--start", default=AlphaFeatureCorrelationConfig.start)
    p.add_argument("--end", default=AlphaFeatureCorrelationConfig.end)
    p.add_argument("--exclude-from", default=AlphaFeatureCorrelationConfig.exclude_from)
    p.add_argument("--window-size", type=int, default=AlphaFeatureCorrelationConfig.window_size)
    p.add_argument("--max-heatmap-features", type=int, default=AlphaFeatureCorrelationConfig.max_heatmap_features)
    return p.parse_args()


def main() -> None:
    summary = run(AlphaFeatureCorrelationConfig(**vars(parse_args())))
    print(json.dumps({
        "output_dir": summary["config"]["output_dir"],
        "docs_output": summary["config"]["docs_output"],
        "analysis_window": summary["analysis_window"],
        "continuous_feature_count": summary["continuous_feature_count"],
        "component_count": summary["component_count"],
        "top_abs_spearman_pairs": summary["top_abs_spearman_pairs"][:10],
        "top_abs_phi_pairs": summary["top_abs_phi_pairs"][:10],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
