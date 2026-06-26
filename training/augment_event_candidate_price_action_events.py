"""Augment symbolic event-action rows with causal price-action event tokens.

The features come from shifted prior rolling ranges in
`price_action_event_scan.build_price_action_event_features`, so the current bar
is compared against prior range levels and no future bars are used.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.price_action_event_scan import build_price_action_event_features


@dataclass(frozen=True)
class AugmentPriceActionEventsCfg:
    input_jsonl: str
    market_csv: str
    output_jsonl: str
    summary_output: str = ""
    windows: tuple[int, ...] = (36, 72, 144, 288, 576, 2016, 4032, 8640)
    token_windows: tuple[int, ...] = (144, 288, 576, 2016, 4032, 8640)
    tolerance: str = "5min"


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def _load_feature_frame(path: str, windows: tuple[int, ...]) -> pd.DataFrame:
    market = pd.read_csv(path, parse_dates=["date"], compression="infer")
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    feats = build_price_action_event_features(market, list(windows))
    return pd.concat([market[["date"]], feats], axis=1).sort_values("date").reset_index(drop=True)


def _event_cols(frame: pd.DataFrame, token_windows: tuple[int, ...]) -> list[str]:
    prefixes = tuple(f"pae_w{int(w)}_" for w in token_windows)
    out = []
    for col in frame.columns:
        if col == "date" or not col.startswith(prefixes):
            continue
        vals = frame[col].dropna().unique()
        if len(vals) and set(float(v) for v in vals).issubset({0.0, 1.0}):
            out.append(col)
    return sorted(out)


def augment_rows(rows: list[dict[str, Any]], feature_frame: pd.DataFrame, cfg: AugmentPriceActionEventsCfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"rows": 0, "matched_rows": 0, "event_token_cols": 0}
    key = pd.DataFrame({"_row_id": np.arange(len(rows)), "date": pd.to_datetime([r.get("date") for r in rows], errors="raise")})
    merged = pd.merge_asof(
        key.sort_values("date"),
        feature_frame.sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(cfg.tolerance),
    ).sort_values("_row_id")
    cols = _event_cols(feature_frame, cfg.token_windows)
    out: list[dict[str, Any]] = []
    matched = 0
    active_counts: dict[str, int] = {c: 0 for c in cols}
    for row, (_, feat_row) in zip(rows, merged.iterrows()):
        nr = dict(row)
        toks = dict(nr.get("state_tokens", {}) if isinstance(nr.get("state_tokens"), dict) else {})
        active: list[str] = []
        any_match = False
        for col in cols:
            val = feat_row.get(col)
            if val is None or pd.isna(val):
                continue
            any_match = True
            if float(val) > 0.5:
                token_name = f"pae:{col}"
                toks[token_name] = "on"
                active.append(col)
                active_counts[col] += 1
        # Coarse count token helps the symbolic model learn crowded event regimes
        # without numeric raw values.
        n_active = len(active)
        if n_active == 0:
            toks["pae:active_count"] = "zero"
        elif n_active <= 2:
            toks["pae:active_count"] = "few"
        elif n_active <= 5:
            toks["pae:active_count"] = "some"
        else:
            toks["pae:active_count"] = "many"
        if any_match:
            matched += 1
        nr["state_tokens"] = dict(sorted(toks.items()))
        lg = dict(nr.get("leakage_guard", {}) if isinstance(nr.get("leakage_guard"), dict) else {})
        lg["price_action_event_tokens_backward_asof"] = True
        lg["price_action_event_tokens_use_shifted_prior_ranges"] = True
        nr["leakage_guard"] = lg
        out.append(nr)
    summary = {
        "rows": len(rows),
        "matched_rows": matched,
        "match_rate": matched / max(1, len(rows)),
        "event_token_cols": len(cols),
        "active_event_counts_top": sorted(active_counts.items(), key=lambda kv: kv[1], reverse=True)[:30],
        "first_date": rows[0].get("date"),
        "last_date": rows[-1].get("date"),
    }
    return out, summary


def run(cfg: AugmentPriceActionEventsCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    frame = _load_feature_frame(cfg.market_csv, cfg.windows)
    out, summary = augment_rows(rows, frame, cfg)
    _write_jsonl(cfg.output_jsonl, out)
    report = {
        "config": asdict(cfg) | {"windows": list(cfg.windows), "token_windows": list(cfg.token_windows)},
        "summary": summary,
        "leakage_guard": {
            "market_feature_join_direction": "backward_asof",
            "features_use_rows_at_or_before_t": True,
            "price_action_ranges_are_shifted_prior_levels": True,
            "reward_fields_unchanged": True,
        },
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment event-action rows with causal price-action event tokens")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--windows", default="36,72,144,288,576,2016,4032,8640")
    p.add_argument("--token-windows", default="144,288,576,2016,4032,8640")
    p.add_argument("--tolerance", default=AugmentPriceActionEventsCfg.tolerance)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = AugmentPriceActionEventsCfg(
        input_jsonl=a.input_jsonl,
        market_csv=a.market_csv,
        output_jsonl=a.output_jsonl,
        summary_output=a.summary_output,
        windows=_parse_ints(a.windows),
        token_windows=_parse_ints(a.token_windows),
        tolerance=a.tolerance,
    )
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
