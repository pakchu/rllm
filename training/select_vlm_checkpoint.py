"""Train/evaluate multiple VLM GRPO checkpoints and select the best one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.eval_vlm_policy import evaluate_vlm_policy
from training.train_vlm_grpo import train_vlm_grpo_smoke


def parse_step_list(step_list: str) -> list[int]:
    """Parse comma-separated positive integer steps."""
    out = []
    for tok in str(step_list).split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = int(tok)
        if val <= 0:
            raise ValueError(f"steps must be > 0, got {val}")
        out.append(val)
    if not out:
        raise ValueError("step list must not be empty")
    return out


def compute_selection_score(metrics: dict) -> float:
    """
    Score checkpoint quality.

    Prioritize accuracy while encouraging balanced class recall and
    penalizing BUY/SELL directional imbalance.
    """
    accuracy = float(metrics["accuracy"])
    buy_recall = float(metrics["per_class"]["BUY"]["recall"])
    hold_recall = float(metrics["per_class"]["HOLD"]["recall"])
    sell_recall = float(metrics["per_class"]["SELL"]["recall"])
    balanced_recall = float(
        metrics.get("balanced_recall", (buy_recall + hold_recall + sell_recall) / 3.0)
    )
    directional_mean = float(metrics.get("directional_recall_mean", 0.5 * (buy_recall + sell_recall)))
    directional_gap = float(metrics.get("directional_recall_gap", abs(buy_recall - sell_recall)))
    return accuracy + 0.15 * balanced_recall + 0.05 * directional_mean - 0.10 * directional_gap


def select_best_vlm_checkpoint(
    *,
    candidate_steps: list[int],
    output_root: str = "results/vlm_checkpoint_selection",
    model_name: str = "auto",
    source: str = "synthetic",
    input_csv: str | None = None,
    timeframe: str = "1m",
    symbol: str = "BTCUSDT",
    start_date: str | None = None,
    end_date: str | None = None,
    market_type: str = "futures",
    num_rows: int = 8_000,
    synthetic_drift: float = 0.0,
    synthetic_regime_amplitude: float = 0.0004,
    synthetic_regime_period: int = 720,
    window_size: int = 96,
    resolution: int = 224,
    cache_dir: str | None = "data/image_cache_vlm",
    prompt_style: str = "numeric",
    prompt_feature_mode: str = "basic_v0",
    max_samples: int = 300,
    hold_band: float = 0.0005,
    target_horizon: int = 1,
    label_mode: str = "next_return",
    buy_reward_weight: float = 1.0,
    hold_reward_weight: float = 1.0,
    sell_reward_weight: float = 1.0,
    reward_mode: str = "classification",
    utility_hold_margin: float = 0.0,
    utility_fee_rate: float = 0.0005,
    utility_slippage_rate: float = 0.0001,
    utility_leverage: float = 1.0,
    utility_stop_loss: float | None = None,
    utility_take_profit: float | None = None,
    utility_use_log_return: bool = True,
    utility_base_risk_weight: float = 0.0,
    utility_regime_weight_volatility: float = 0.0,
    utility_regime_weight_downtrend: float = 0.0,
    utility_regime_weight_drawdown: float = 0.0,
    utility_min_risk_weight: float = 0.0,
    utility_max_risk_weight: float = 1.0,
    utility_hold_reward_bias: float = 0.0,
    path_entry_delay_bars: int = 1,
    path_mae_penalty: float = 1.0,
    path_mfe_bonus: float = 0.0,
    path_min_net_return: float = 0.0,
    path_max_mae: float = 1.0,
    multi_horizon_bars: str = "36,72,144",
    utility_reward_scale: float = 400.0,
    utility_gap_scale: float = 400.0,
    sample_mode: str = "balanced",
    sample_seed: int = 22,
    learning_rate: float = 1e-5,
    per_device_train_batch_size: int = 1,
    num_generations: int = 2,
    temperature: float = 2.0,
    top_p: float = 0.95,
    top_k: int = 50,
    scale_rewards: str = "batch",
    reward_variance_guard: str = "auto",
    max_completion_length: int = 8,
    min_new_tokens: int = 1,
    do_sample: bool = True,
    eval_max_samples: int = 90,
    eval_sample_mode: str = "balanced",
    eval_sample_seed: int = 33,
    eval_decision_mode: str = "generate",
    eval_action_bias_buy: float = 0.0,
    eval_action_bias_hold: float = 0.0,
    eval_action_bias_sell: float = 0.0,
    eval_store_action_scores: bool = False,
) -> dict:
    """Run candidate train/eval checkpoints and return best selection report."""
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = []
    for step in candidate_steps:
        candidate_dir = out_root / f"step_{int(step)}"
        eval_json = out_root / f"eval_step_{int(step)}.json"
        print(f"[select-vlm] training step={step}")
        train_vlm_grpo_smoke(
            model_name=model_name,
            output_dir=str(candidate_dir),
            source=source,
            input_csv=input_csv,
            timeframe=timeframe,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            market_type=market_type,
            num_rows=num_rows,
            synthetic_drift=synthetic_drift,
            synthetic_regime_amplitude=synthetic_regime_amplitude,
            synthetic_regime_period=synthetic_regime_period,
            window_size=window_size,
            resolution=resolution,
            cache_dir=cache_dir,
            prompt_style=prompt_style,
            prompt_feature_mode=prompt_feature_mode,
            max_samples=max_samples,
            hold_band=hold_band,
            target_horizon=target_horizon,
            label_mode=label_mode,
            buy_reward_weight=buy_reward_weight,
            hold_reward_weight=hold_reward_weight,
            sell_reward_weight=sell_reward_weight,
            reward_mode=reward_mode,
            utility_hold_margin=utility_hold_margin,
            utility_fee_rate=utility_fee_rate,
            utility_slippage_rate=utility_slippage_rate,
            utility_leverage=utility_leverage,
            utility_stop_loss=utility_stop_loss,
            utility_take_profit=utility_take_profit,
            utility_use_log_return=utility_use_log_return,
            utility_base_risk_weight=utility_base_risk_weight,
            utility_regime_weight_volatility=utility_regime_weight_volatility,
            utility_regime_weight_downtrend=utility_regime_weight_downtrend,
            utility_regime_weight_drawdown=utility_regime_weight_drawdown,
            utility_min_risk_weight=utility_min_risk_weight,
            utility_max_risk_weight=utility_max_risk_weight,
            utility_hold_reward_bias=utility_hold_reward_bias,
            path_entry_delay_bars=path_entry_delay_bars,
            path_mae_penalty=path_mae_penalty,
            path_mfe_bonus=path_mfe_bonus,
            path_min_net_return=path_min_net_return,
            path_max_mae=path_max_mae,
            multi_horizon_bars=multi_horizon_bars,
            utility_reward_scale=utility_reward_scale,
            utility_gap_scale=utility_gap_scale,
            sample_mode=sample_mode,
            sample_seed=sample_seed,
            max_steps=int(step),
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            num_generations=num_generations,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            scale_rewards=scale_rewards,
            reward_variance_guard=reward_variance_guard,
            max_completion_length=max_completion_length,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
        )

        print(f"[select-vlm] evaluating step={step}")
        eval_report = evaluate_vlm_policy(
            model_name=model_name,
            adapter_dir=str(candidate_dir),
            source=source,
            input_csv=input_csv,
            timeframe=timeframe,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            market_type=market_type,
            num_rows=num_rows,
            synthetic_drift=synthetic_drift,
            synthetic_regime_amplitude=synthetic_regime_amplitude,
            synthetic_regime_period=synthetic_regime_period,
            window_size=window_size,
            resolution=resolution,
            cache_dir=cache_dir,
            prompt_style=prompt_style,
            prompt_feature_mode=prompt_feature_mode,
            hold_band=hold_band,
            target_horizon=target_horizon,
            label_mode=label_mode,
            utility_hold_margin=utility_hold_margin,
            utility_fee_rate=utility_fee_rate,
            utility_slippage_rate=utility_slippage_rate,
            utility_leverage=utility_leverage,
            utility_stop_loss=utility_stop_loss,
            utility_take_profit=utility_take_profit,
            utility_use_log_return=utility_use_log_return,
            utility_base_risk_weight=utility_base_risk_weight,
            utility_regime_weight_volatility=utility_regime_weight_volatility,
            utility_regime_weight_downtrend=utility_regime_weight_downtrend,
            utility_regime_weight_drawdown=utility_regime_weight_drawdown,
            utility_min_risk_weight=utility_min_risk_weight,
            utility_max_risk_weight=utility_max_risk_weight,
            utility_hold_reward_bias=utility_hold_reward_bias,
            path_entry_delay_bars=path_entry_delay_bars,
            path_mae_penalty=path_mae_penalty,
            path_mfe_bonus=path_mfe_bonus,
            path_min_net_return=path_min_net_return,
            path_max_mae=path_max_mae,
            multi_horizon_bars=multi_horizon_bars,
            max_samples=eval_max_samples,
            sample_mode=eval_sample_mode,
            sample_seed=eval_sample_seed,
            max_completion_length=8,
            min_new_tokens=1,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            decision_mode=eval_decision_mode,
            action_bias_buy=eval_action_bias_buy,
            action_bias_hold=eval_action_bias_hold,
            action_bias_sell=eval_action_bias_sell,
            store_action_scores=eval_store_action_scores,
            output=str(eval_json),
        )
        score = compute_selection_score(eval_report["metrics"])
        candidates.append(
            {
                "step": int(step),
                "checkpoint_dir": str(candidate_dir.resolve()),
                "eval_report": str(eval_json.resolve()),
                "metrics": eval_report["metrics"],
                "score": float(score),
            }
        )

    best = max(candidates, key=lambda x: float(x["score"]))
    report = {
        "candidate_steps": [int(x) for x in candidate_steps],
        "selection_rule": (
            "accuracy + 0.15 * balanced_recall + 0.05 * mean(BUY_recall, SELL_recall)"
            " - 0.10 * abs(BUY_recall - SELL_recall)"
        ),
        "best_step": int(best["step"]),
        "best_checkpoint_dir": best["checkpoint_dir"],
        "best_eval_report": best["eval_report"],
        "best_score": float(best["score"]),
        "candidates": candidates,
    }
    report_path = out_root / "selection_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best VLM checkpoint by eval score.")
    parser.add_argument("--candidate-steps", type=str, default="20,30,40")
    parser.add_argument("--output-root", type=str, default="results/vlm_checkpoint_selection")
    parser.add_argument("--model-name", type=str, default="auto")
    parser.add_argument(
        "--source", type=str, default="synthetic", choices=["synthetic", "csv", "binance"]
    )
    parser.add_argument("--input-csv", type=str, default="")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--market-type", type=str, default="futures")
    parser.add_argument("--num-rows", type=int, default=8000)
    parser.add_argument("--synthetic-drift", type=float, default=0.0)
    parser.add_argument("--synthetic-regime-amplitude", type=float, default=0.0004)
    parser.add_argument("--synthetic-regime-period", type=int, default=720)
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--cache-dir", type=str, default="data/image_cache_vlm")
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="numeric",
        choices=["numeric", "symbolic", "hybrid"],
    )
    parser.add_argument(
        "--prompt-feature-mode",
        type=str,
        default="basic_v0",
        choices=["basic_v0", "engineered_v1", "edge_state_v2", "edge_state_v3", "edge_state_v4", "edge_state_v5", "edge_state_v6"],
    )
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--hold-band", type=float, default=0.0005)
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument(
        "--label-mode",
        type=str,
        default="next_return",
        choices=["next_return", "utility"],
    )
    parser.add_argument("--buy-reward-weight", type=float, default=1.0)
    parser.add_argument("--hold-reward-weight", type=float, default=1.0)
    parser.add_argument("--sell-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="classification",
        choices=["classification", "utility"],
    )
    parser.add_argument("--utility-hold-margin", type=float, default=0.0)
    parser.add_argument("--utility-fee-rate", type=float, default=0.0005)
    parser.add_argument("--utility-slippage-rate", type=float, default=0.0001)
    parser.add_argument("--utility-leverage", type=float, default=1.0)
    parser.add_argument("--utility-stop-loss", type=float, default=-1.0)
    parser.add_argument("--utility-take-profit", type=float, default=-1.0)
    parser.add_argument(
        "--utility-use-log-return",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument("--utility-base-risk-weight", type=float, default=0.0)
    parser.add_argument("--utility-regime-weight-volatility", type=float, default=0.0)
    parser.add_argument("--utility-regime-weight-downtrend", type=float, default=0.0)
    parser.add_argument("--utility-regime-weight-drawdown", type=float, default=0.0)
    parser.add_argument("--utility-min-risk-weight", type=float, default=0.0)
    parser.add_argument("--utility-max-risk-weight", type=float, default=1.0)
    parser.add_argument("--utility-hold-reward-bias", type=float, default=0.0)
    parser.add_argument("--utility-reward-scale", type=float, default=400.0)
    parser.add_argument("--utility-gap-scale", type=float, default=400.0)
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="balanced",
        choices=["sequential", "random", "balanced", "uniform"],
    )
    parser.add_argument("--sample-seed", type=int, default=22)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--scale-rewards", type=str, default="batch", choices=["group", "batch", "none"]
    )
    parser.add_argument(
        "--reward-variance-guard", type=str, default="auto", choices=["auto", "off"]
    )
    parser.add_argument("--max-completion-length", type=int, default=8)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--do-sample", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--eval-max-samples", type=int, default=90)
    parser.add_argument(
        "--eval-sample-mode",
        type=str,
        default="balanced",
        choices=["sequential", "random", "balanced", "uniform"],
    )
    parser.add_argument("--eval-sample-seed", type=int, default=33)
    parser.add_argument(
        "--eval-decision-mode",
        type=str,
        default="generate",
        choices=["generate", "likelihood"],
    )
    parser.add_argument("--eval-action-bias-buy", type=float, default=0.0)
    parser.add_argument("--eval-action-bias-hold", type=float, default=0.0)
    parser.add_argument("--eval-action-bias-sell", type=float, default=0.0)
    parser.add_argument(
        "--eval-store-action-scores",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utility_stop_loss = (
        None if float(args.utility_stop_loss) <= 0.0 else float(args.utility_stop_loss)
    )
    utility_take_profit = (
        None if float(args.utility_take_profit) <= 0.0 else float(args.utility_take_profit)
    )
    report = select_best_vlm_checkpoint(
        candidate_steps=parse_step_list(args.candidate_steps),
        output_root=args.output_root,
        model_name=args.model_name,
        source=args.source,
        input_csv=args.input_csv or None,
        timeframe=args.timeframe,
        symbol=args.symbol,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        market_type=args.market_type,
        num_rows=args.num_rows,
        synthetic_drift=args.synthetic_drift,
        synthetic_regime_amplitude=args.synthetic_regime_amplitude,
        synthetic_regime_period=args.synthetic_regime_period,
        window_size=args.window_size,
        resolution=args.resolution,
        cache_dir=args.cache_dir or None,
        prompt_style=args.prompt_style,
        prompt_feature_mode=args.prompt_feature_mode,
        max_samples=args.max_samples,
        hold_band=args.hold_band,
        target_horizon=args.target_horizon,
        label_mode=args.label_mode,
        buy_reward_weight=args.buy_reward_weight,
        hold_reward_weight=args.hold_reward_weight,
        sell_reward_weight=args.sell_reward_weight,
        reward_mode=args.reward_mode,
        utility_hold_margin=args.utility_hold_margin,
        utility_fee_rate=args.utility_fee_rate,
        utility_slippage_rate=args.utility_slippage_rate,
        utility_leverage=args.utility_leverage,
        utility_stop_loss=utility_stop_loss,
        utility_take_profit=utility_take_profit,
        utility_use_log_return=args.utility_use_log_return == "true",
        utility_base_risk_weight=args.utility_base_risk_weight,
        utility_regime_weight_volatility=args.utility_regime_weight_volatility,
        utility_regime_weight_downtrend=args.utility_regime_weight_downtrend,
        utility_regime_weight_drawdown=args.utility_regime_weight_drawdown,
        utility_min_risk_weight=args.utility_min_risk_weight,
        utility_max_risk_weight=args.utility_max_risk_weight,
        utility_hold_reward_bias=args.utility_hold_reward_bias,
        utility_reward_scale=args.utility_reward_scale,
        utility_gap_scale=args.utility_gap_scale,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        scale_rewards=args.scale_rewards,
        reward_variance_guard=args.reward_variance_guard,
        max_completion_length=args.max_completion_length,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample == "true",
        eval_max_samples=args.eval_max_samples,
        eval_sample_mode=args.eval_sample_mode,
        eval_sample_seed=args.eval_sample_seed,
        eval_decision_mode=args.eval_decision_mode,
        eval_action_bias_buy=args.eval_action_bias_buy,
        eval_action_bias_hold=args.eval_action_bias_hold,
        eval_action_bias_sell=args.eval_action_bias_sell,
        eval_store_action_scores=args.eval_store_action_scores == "true",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
