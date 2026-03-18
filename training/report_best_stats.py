"""Report detailed stats for the current best 5m scoreband configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.backtest import run_backtest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report best config stats (including trade counts).")
    p.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/ppo_option_a_real_w384_t786k_exp6_balanced.zip",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/best_config_stats_recent3m.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    common = dict(
        model_path=args.model_path,
        source="binance",
        symbol="BTCUSDT",
        timeframe="5m",
        market_type="futures",
        window_size=384,
        deterministic=True,
        decision_mode="score_band",
        debiased_action="mirror_scalar",
        flat_start_policy="as_is",
        score_centering="ema",
        score_center_alpha=0.02,
        score_entry_threshold=0.005,
        score_flip_threshold=0.02,
        score_neutral_band=0.001,
    )
    periods = {
        "2025-12": ("2025-12-01", "2025-12-31"),
        "2026-01": ("2026-01-01", "2026-01-31"),
        "2026-02": ("2026-02-01", "2026-02-28"),
        "recent_3m": ("2025-12-01", "2026-02-28"),
        "last_1y": ("2025-03-01", "2026-02-28"),
    }

    out = {}
    for name, (start, end) in periods.items():
        print(f"[run] {name}: {start}..{end}", flush=True)
        out[name] = run_backtest(start_date=start, end_date=end, **common)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"[saved] {output.resolve()}")
    for name, rep in out.items():
        tc = rep.get("trade_counts", {})
        print(
            f"{name}: ret={rep['cumulative_return_pct']:.3f}% "
            f"mdd={rep['max_drawdown_pct']:.3f}% "
            f"sharpe={rep['sharpe_ratio']:.3f} "
            f"rebalances={tc.get('rebalance_steps', 0)} "
            f"legs={tc.get('turnover_legs', 0)}"
        )


if __name__ == "__main__":
    main()

