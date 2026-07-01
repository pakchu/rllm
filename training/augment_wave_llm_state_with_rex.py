"""Add history-only rolling-extrema tokens to existing wave LLM state rows."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.build_wave_llm_state_dataset import _bucket_range_pos, _bucket_rex_gap, _bucket_rex_width, _safe_feature


@dataclass(frozen=True)
class AugmentWaveStateRexCfg:
    input_jsonl: str
    market_csv: str
    output_jsonl: str
    summary_output: str = ""
    window_size: int = 144


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _rex_tokens(features: pd.DataFrame, pos: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for rex_window in (36, 144, 576, 2016, 8640):
        prefix = f"rex_{rex_window}"
        out[f"{prefix}_loc"] = _bucket_range_pos(_safe_feature(features, pos, f"{prefix}_range_pos"))
        out[f"{prefix}_width"] = _bucket_rex_width(_safe_feature(features, pos, f"{prefix}_range_width_pct"))
        out[f"{prefix}_upper_gap"] = _bucket_rex_gap(_safe_feature(features, pos, f"{prefix}_max_to_cur_pct"))
        out[f"{prefix}_lower_gap"] = _bucket_rex_gap(_safe_feature(features, pos, f"{prefix}_cur_to_min_pct"))
    return out


def _rewrite_prompt(prompt: str, tokens: dict[str, str]) -> str:
    lines = str(prompt).splitlines()
    existing = {line.split(":", 1)[0].strip()[2:] for line in lines if line.startswith("- rex_") and ":" in line}
    additions = [f"- {k}: {tokens[k]}" for k in sorted(tokens) if k not in existing]
    if not additions:
        return prompt
    try:
        insert_at = lines.index("Return JSON with decision in {TAKE_FULL, TAKE_SMALL, ABSTAIN} and a short risk reason.")
    except ValueError:
        insert_at = len(lines)
    return "\n".join(lines[:insert_at] + additions + lines[insert_at:])


def run(cfg: AugmentWaveStateRexCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    market = _load_market(cfg.market_csv)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    out_rows = []
    token_counts: dict[str, int] = {}
    for row in rows:
        nr = dict(row)
        pos = int(nr.get("signal_pos", -1) or -1)
        rex = _rex_tokens(features, pos)
        state = dict(nr.get("state_tokens") or {})
        state.update(rex)
        nr["state_tokens"] = state
        nr["prompt"] = _rewrite_prompt(str(nr.get("prompt", "")), rex)
        leak = dict(nr.get("leakage_guard") or {})
        leak["rex_features_signal_time_or_prior"] = True
        nr["leakage_guard"] = leak
        for k, v in rex.items():
            token_counts[f"{k}={v}"] = token_counts.get(f"{k}={v}", 0) + 1
        out_rows.append(nr)
    Path(cfg.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out_rows) + ("\n" if out_rows else ""))
    summary = {
        "config": asdict(cfg),
        "rows": len(out_rows),
        "market_rows": len(market),
        "rex_token_keys_added": 20,
        "top_rex_tokens": sorted(({"token": k, "count": v} for k, v in token_counts.items()), key=lambda x: x["count"], reverse=True)[:40],
        "leakage_guard": {"features_use_rows_at_or_before_signal_pos": True, "future_rewards_unchanged_label_only": True},
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--window-size", type=int, default=AugmentWaveStateRexCfg.window_size)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(AugmentWaveStateRexCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
