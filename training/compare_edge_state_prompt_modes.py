"""Compare VLM/text prompt feature modes on identical samples."""
from __future__ import annotations

import argparse, json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from training.vlm_trading_data import build_vlm_training_samples


@dataclass(frozen=True)
class ComparePromptModesConfig:
    input_csv: str
    output: str
    samples_output: str = ""
    modes: str = "edge_state_v4,edge_state_v5"
    start: str = "2025-01-01"
    end: str = "2025-12-01 23:59:59"
    window_size: int = 144
    max_samples: int = 256
    sample_mode: str = "uniform"
    sample_seed: int = 42
    target_horizon: int = 288
    action_schema: str = "trade_side"
    label_mode: str = "path_outcome"
    prompt_style: str = "numeric"
    modality: str = "text_only"
    wave_trading_root: str = ""
    external_tolerance: str = "30min"


def _load_market(path: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    # Include warmup rows before start so rolling/window features are available.
    first = int(mask.to_numpy().argmax()) if mask.any() else 0
    warm = max(0, first - int(2 * 144))
    return df.iloc[warm:].reset_index(drop=True)


def _descriptor_fields(prompt: str) -> dict[str, str]:
    wanted = {"Kimchi Flow Regime", "Long Entry Context", "Short Entry Context", "Regime Failure Cue", "Trade Readiness", "Step Focus", "Regime Memory", "Regime Trap Risk"}
    out: dict[str, str] = {}
    for line in str(prompt).splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip(); val = val.strip()
        if key in wanted:
            out[key] = val
    return out


def _sample_rows(mode: str, samples) -> list[dict]:
    return [
        {
            "mode": mode,
            "date": s.date,
            "target_action": s.target_action,
            "next_return": s.next_return,
            "prompt": s.prompt,
            "descriptors": _descriptor_fields(s.prompt),
        }
        for s in samples
    ]


def _sample_summary(samples) -> dict:
    labels = Counter(str(s.target_action) for s in samples)
    prompt_lens = [len(str(s.prompt)) for s in samples]
    tag_counts = Counter()
    field_counts = Counter()
    for sample in samples:
        prompt = str(sample.prompt)
        for line in prompt.splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key in {"Kimchi Flow Regime", "Long Entry Context", "Short Entry Context", "Regime Failure Cue", "Trade Readiness", "Step Focus", "Regime Memory", "Regime Trap Risk"}:
                field_counts[f"{key}={val}"] += 1
                tag_counts[val] += 1
    return {
        "samples": len(samples),
        "labels": dict(sorted(labels.items())),
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens)/max(1,len(prompt_lens))},
        "top_tags": dict(tag_counts.most_common(20)),
        "top_symbolic_fields": dict(field_counts.most_common(30)),
        "example_prompt": samples[0].prompt if samples else "",
    }


def run(cfg: ComparePromptModesConfig) -> dict:
    market = _load_market(cfg.input_csv, cfg.start, cfg.end)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "modes": {}}
    sample_rows: list[dict] = []
    for mode in [x.strip() for x in cfg.modes.split(',') if x.strip()]:
        samples = build_vlm_training_samples(
            market,
            window_size=cfg.window_size,
            max_samples=cfg.max_samples,
            sample_mode=cfg.sample_mode,
            sample_seed=cfg.sample_seed,
            target_horizon=cfg.target_horizon,
            label_mode=cfg.label_mode,
            prompt_feature_mode=mode,
            action_schema=cfg.action_schema,
            prompt_style=cfg.prompt_style,
            modality=cfg.modality,
            path_entry_delay_bars=1,
            utility_fee_rate=0.0004,
            utility_slippage_rate=0.0001,
            utility_leverage=0.5,
            path_mae_penalty=1.0,
            path_min_net_return=-1.0,
            path_max_mae=1.0,
        )
        report["modes"][mode] = _sample_summary(samples)
        sample_rows.extend(_sample_rows(mode, samples))
    if cfg.samples_output:
        Path(cfg.samples_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.samples_output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in sample_rows) + "\n")
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input-csv', required=True); p.add_argument('--output', required=True); p.add_argument('--samples-output', default='')
    p.add_argument('--modes', default='edge_state_v4,edge_state_v5')
    p.add_argument('--start', default='2025-01-01'); p.add_argument('--end', default='2025-12-01 23:59:59')
    p.add_argument('--window-size', type=int, default=144); p.add_argument('--max-samples', type=int, default=256); p.add_argument('--sample-mode', default='uniform'); p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--target-horizon', type=int, default=288); p.add_argument('--action-schema', default='trade_side'); p.add_argument('--label-mode', default='path_outcome'); p.add_argument('--prompt-style', default='numeric'); p.add_argument('--modality', default='text_only')
    p.add_argument('--wave-trading-root', default=''); p.add_argument('--external-tolerance', default='30min')
    return p.parse_args()


def main():
    r=run(ComparePromptModesConfig(**vars(parse_args())))
    for mode, s in r['modes'].items():
        print(mode, 'samples', s['samples'], 'labels', s['labels'], 'prompt_chars', s['prompt_chars'])
        print('top fields', list(s['top_symbolic_fields'].items())[:8])

if __name__=='__main__': main()
