"""Walk-forward evaluation utilities for Option A real-market backtests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.backtest import run_backtest


def parse_folds_spec(spec: str) -> list[dict]:
    """
    Parse fold spec string.

    Format:
      "val_start:val_end:test_start:test_end[, ...]"
    """
    out: list[dict] = []
    for i, tok in enumerate(str(spec).split(","), start=1):
        tok = tok.strip()
        if not tok:
            continue
        parts = [x.strip() for x in tok.split(":")]
        if len(parts) != 4:
            raise ValueError(
                "Each fold must have 4 date fields: "
                "val_start:val_end:test_start:test_end"
            )
        out.append(
            {
                "name": f"fold_{i}",
                "val_start": parts[0],
                "val_end": parts[1],
                "test_start": parts[2],
                "test_end": parts[3],
            }
        )
    if not out:
        raise ValueError("fold spec must not be empty")
    return out


def default_walkforward_folds() -> list[dict]:
    """Default real-market weekly walk-forward folds (March 2025)."""
    return parse_folds_spec(
        "2025-03-01:2025-03-07:2025-03-08:2025-03-14,"
        "2025-03-08:2025-03-14:2025-03-15:2025-03-21,"
        "2025-03-15:2025-03-21:2025-03-22:2025-03-28"
    )


def default_candidate_policies() -> list[dict]:
    """Candidate execution policies to select per fold."""
    return [
        {
            "name": "policy_det",
            "params": {
                "deterministic": True,
                "decision_mode": "policy",
                "debiased_action": "off",
                "flat_start_policy": "as_is",
                "score_centering": "off",
                "score_entry_threshold": 0.02,
                "score_flip_threshold": 0.05,
                "score_neutral_band": 0.005,
                "directional_tie_hold_eps": 0.0,
            },
        },
        {
            "name": "scoreband_active_A",
            "params": {
                "deterministic": True,
                "decision_mode": "score_band",
                "debiased_action": "mirror_scalar",
                "flat_start_policy": "prefer_entry",
                "score_centering": "ema",
                "score_center_alpha": 0.02,
                "score_entry_threshold": 0.0005,
                "score_flip_threshold": 0.002,
                "score_neutral_band": 0.0001,
                "directional_tie_hold_eps": 0.0,
            },
        },
        {
            "name": "scoreband_safe_D",
            "params": {
                "deterministic": True,
                "decision_mode": "score_band",
                "debiased_action": "mirror_scalar",
                "flat_start_policy": "as_is",
                "score_centering": "ema",
                "score_center_alpha": 0.02,
                "score_entry_threshold": 0.01,
                "score_flip_threshold": 0.03,
                "score_neutral_band": 0.002,
                "directional_tie_hold_eps": 0.005,
            },
        },
    ]


def compute_policy_score(report: dict) -> float:
    """
    Scalar objective for selecting policy on validation segment.

    Rewards return and min_sharpe; penalizes drawdown and degenerate hold-all.
    """
    ret = float(report.get("cumulative_return_pct", 0.0))
    min_sharpe = float(report.get("min_sharpe", report.get("sharpe_ratio", 0.0)))
    mdd = float(report.get("max_drawdown_pct", 0.0))
    hold_ratio = float((report.get("action_ratio") or {}).get("hold", 0.0))
    hold_penalty = max(0.0, hold_ratio - 0.98) * 2.0
    return float(ret + 0.25 * min_sharpe - 0.05 * mdd - hold_penalty)


def run_optiona_walkforward(
    *,
    model_path: str,
    source: str = "binance",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    market_type: str = "futures",
    window_size: int = 96,
    leverage: float = 1.0,
    initial_equity: float = 1000.0,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0001,
    flat_hold_penalty: float = 0.001,
    hold_action_mode: str = "auto",
    use_images: str = "auto",
    image_cache_dir: str = "data/image_cache_backtest",
    score_center_alpha: float = 0.02,
    trend_guard: str = "off",
    trend_threshold: float = 0.002,
    folds: list[dict] | None = None,
    candidates: list[dict] | None = None,
    output: str | None = None,
) -> dict:
    """Run walk-forward policy selection/evaluation on fixed checkpoint."""
    folds = folds or default_walkforward_folds()
    candidates = candidates or default_candidate_policies()

    common = dict(
        model_path=model_path,
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        market_type=market_type,
        window_size=window_size,
        leverage=leverage,
        initial_equity=initial_equity,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        flat_hold_penalty=flat_hold_penalty,
        hold_action_mode=(None if hold_action_mode == "auto" else hold_action_mode),
        use_images=(None if use_images == "auto" else use_images == "true"),
        image_cache_dir=image_cache_dir,
        score_center_alpha=score_center_alpha,
        trend_guard=trend_guard,
        trend_threshold=trend_threshold,
    )

    fold_reports: list[dict] = []
    test_returns: list[float] = []
    test_min_sharpes: list[float] = []

    for fold in folds:
        val_candidates = []
        for cand in candidates:
            params = dict(cand["params"])
            call_kwargs = dict(common)
            call_kwargs.update(params)
            val_rep = run_backtest(
                start_date=fold["val_start"],
                end_date=fold["val_end"],
                **call_kwargs,
            )
            val_score = compute_policy_score(val_rep)
            val_candidates.append(
                {
                    "name": str(cand["name"]),
                    "params": params,
                    "val_score": float(val_score),
                    "val_report": val_rep,
                }
            )

        best = max(
            val_candidates,
            key=lambda x: (float(x["val_score"]), float(x["val_report"]["cumulative_return_pct"])),
        )

        test_kwargs = dict(common)
        test_kwargs.update(best["params"])
        test_rep = run_backtest(
            start_date=fold["test_start"],
            end_date=fold["test_end"],
            **test_kwargs,
        )
        test_returns.append(float(test_rep["cumulative_return_pct"]))
        test_min_sharpes.append(float(test_rep["min_sharpe"]))

        fold_reports.append(
            {
                "name": fold["name"],
                "val_start": fold["val_start"],
                "val_end": fold["val_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "candidates": val_candidates,
                "selected_policy": {
                    "name": best["name"],
                    "params": best["params"],
                    "val_score": float(best["val_score"]),
                },
                "test_report": test_rep,
            }
        )

    mean_ret = float(sum(test_returns) / max(1, len(test_returns)))
    std_ret = float(
        (sum((x - mean_ret) ** 2 for x in test_returns) / max(1, len(test_returns))) ** 0.5
    )
    mean_min_sharpe = float(sum(test_min_sharpes) / max(1, len(test_min_sharpes)))

    report = {
        "model_path": str(Path(model_path).resolve()),
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
        "market_type": market_type,
        "num_folds": int(len(fold_reports)),
        "candidate_policy_names": [str(c["name"]) for c in candidates],
        "folds": fold_reports,
        "summary": {
            "mean_test_cumulative_return_pct": mean_ret,
            "std_test_cumulative_return_pct": std_ret,
            "mean_test_min_sharpe": mean_min_sharpe,
            "test_returns_pct": test_returns,
            "test_min_sharpes": test_min_sharpes,
        },
    }

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Option A walk-forward evaluation.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--source", type=str, default="binance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--market-type", type=str, default="futures")
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0001)
    parser.add_argument("--flat-hold-penalty", type=float, default=0.001)
    parser.add_argument(
        "--hold-action-mode",
        type=str,
        default="auto",
        choices=["auto", "flat", "maintain"],
    )
    parser.add_argument(
        "--use-images",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
    )
    parser.add_argument("--image-cache-dir", type=str, default="data/image_cache_backtest")
    parser.add_argument("--score-center-alpha", type=float, default=0.02)
    parser.add_argument("--trend-guard", type=str, default="off", choices=["off", "hard"])
    parser.add_argument("--trend-threshold", type=float, default=0.002)
    parser.add_argument(
        "--folds",
        type=str,
        default="",
        help="Optional folds spec: val_start:val_end:test_start:test_end[, ...]",
    )
    parser.add_argument("--output", type=str, default="results/optiona_walkforward.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = parse_folds_spec(args.folds) if args.folds.strip() else default_walkforward_folds()
    report = run_optiona_walkforward(
        model_path=args.model_path,
        source=args.source,
        symbol=args.symbol,
        timeframe=args.timeframe,
        market_type=args.market_type,
        window_size=args.window_size,
        leverage=args.leverage,
        initial_equity=args.initial_equity,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        flat_hold_penalty=args.flat_hold_penalty,
        hold_action_mode=args.hold_action_mode,
        use_images=args.use_images,
        image_cache_dir=args.image_cache_dir,
        score_center_alpha=args.score_center_alpha,
        trend_guard=args.trend_guard,
        trend_threshold=args.trend_threshold,
        folds=folds,
        output=args.output,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
