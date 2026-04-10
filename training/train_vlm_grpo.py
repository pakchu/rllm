"""Option B: VLM RL fine-tuning (GRPO) entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.option_b_vlm import (
    AUTO_MODEL_NAME,
    FALLBACK_VLM_MODEL,
    RECOMMENDED_VLM_MODEL,
    ACTION_SCHEMA_LABELS,
    auto_select_vlm_model,
    detect_gpu_vram_gb,
)
from training.data_sources import load_market_data
from training.vlm_trading_data import (
    build_vlm_training_samples,
    make_grpo_reward_func,
    samples_to_hf_records,
)
from utils import disable_transformers_allocator_warmup


def _resolve_model_name(model_name: str, allow_fallback: bool) -> str:
    key = model_name.strip()
    if key.lower() == AUTO_MODEL_NAME:
        # For GRPO action-token training, we prioritize exploration stability
        # over newest release preference.
        return auto_select_vlm_model(prefer_latest=False)
    if key:
        return model_name
    if allow_fallback:
        return FALLBACK_VLM_MODEL
    return RECOMMENDED_VLM_MODEL


def _apply_reward_variance_guard(
    *,
    per_device_train_batch_size: int,
    num_generations: int,
    scale_rewards: str,
    dataset_size: int,
    label_counts: dict[str, int],
    reward_variance_guard: str,
) -> tuple[int, int, list[str]]:
    """
    Auto-adjust smoke GRPO sampling so reward variance does not collapse.

    In tiny smoke settings (batch=1, num_generations=2), all completions can
    share identical rewards, producing zero advantages and zero gradients.
    """
    safe_batch = max(1, int(per_device_train_batch_size))
    safe_gens = max(1, int(num_generations))
    notes: list[str] = []

    guard_mode = str(reward_variance_guard).lower().strip()
    if guard_mode not in {"auto", "off"}:
        raise ValueError(
            "reward_variance_guard must be one of {'auto','off'}, "
            f"got {reward_variance_guard}"
        )
    if guard_mode == "off" or dataset_size <= 1:
        return safe_batch, safe_gens, notes

    if safe_batch > dataset_size:
        safe_batch = dataset_size
        notes.append(
            f"per_device_train_batch_size clipped to dataset size ({dataset_size})."
        )

    scale_key = str(scale_rewards).lower().strip()
    num_active_labels = sum(1 for v in label_counts.values() if int(v) > 0)
    # If labels are diverse, demand a slightly larger effective sample count.
    min_effective = 4 if num_active_labels >= 2 else 2

    if scale_key in {"batch", "none"}:
        # In GRPO, num_generations completions are grouped per prompt.
        # batch=2, gens=2 => only 1 prompt/group per optimizer step,
        # which often collapses reward_std to zero.
        prompts_per_step = max(1, safe_batch // safe_gens)
        min_prompts = 2 if num_active_labels >= 2 else 1
        if prompts_per_step < min_prompts:
            needed_batch = min_prompts * safe_gens
            adjusted_batch = min(max(safe_batch, needed_batch), dataset_size)
            if adjusted_batch > safe_batch:
                notes.append(
                    "Raised per_device_train_batch_size "
                    f"{safe_batch} -> {adjusted_batch} "
                    f"(prompt groups/step {prompts_per_step} -> {max(1, adjusted_batch // safe_gens)}) "
                    "to reduce zero-variance rewards."
                )
                safe_batch = adjusted_batch

        effective = safe_batch * safe_gens
        if effective < min_effective:
            # Extra safeguard for extremely tiny settings.
            needed_batch = (min_effective + safe_gens - 1) // safe_gens
            adjusted_batch = min(max(safe_batch, needed_batch), dataset_size)
            if adjusted_batch > safe_batch:
                notes.append(
                    "Raised per_device_train_batch_size "
                    f"{safe_batch} -> {adjusted_batch} "
                    f"(effective samples {effective} -> {adjusted_batch * safe_gens}) "
                    "to reduce zero-variance rewards."
                )
                safe_batch = adjusted_batch
    elif scale_key == "group":
        if safe_gens < 2:
            safe_gens = 2
            notes.append(
                "Raised num_generations 1 -> 2 for group reward scaling."
            )

    return safe_batch, safe_gens, notes


def train_vlm_grpo_smoke(
    model_name: str = RECOMMENDED_VLM_MODEL,
    output_dir: str = "checkpoints/vlm_grpo_smoke",
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
    max_samples: int = 256,
    action_schema: str = "buy_hold_sell",
    prompt_style: str = "numeric",
    prompt_feature_mode: str = "basic_v0",
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
    utility_reward_scale: float = 400.0,
    utility_gap_scale: float = 400.0,
    sample_mode: str = "balanced",
    sample_seed: int = 42,
    max_steps: int = 10,
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
    log_completions: bool = False,
    num_completions_to_print: int = 2,
    load_in_4bit: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    allow_fallback: bool = True,
    dry_run: bool = False,
) -> str:
    """
    Run GRPO training for VLM action-token policy.

    Notes:
      - Dataset is built from market windows (image + scalar prompt).
      - Reward is action correctness against next-open return label.
    """
    chosen_model = _resolve_model_name(model_name=model_name, allow_fallback=allow_fallback)
    vram_gb = detect_gpu_vram_gb()
    if vram_gb is None:
        print(f"[train-vlm-smoke] model={chosen_model} (VRAM: unknown)")
    else:
        print(f"[train-vlm-smoke] model={chosen_model} (VRAM: {vram_gb:.2f} GB)")
    market_df = load_market_data(
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
    )
    samples = build_vlm_training_samples(
        market_df=market_df,
        timeframe=timeframe,
        window_size=window_size,
        resolution=resolution,
        cache_dir=cache_dir,
        action_schema=action_schema,
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
        max_samples=max_samples,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
    )
    if not samples:
        raise ValueError("No VLM training samples were generated.")

    label_counts: dict[str, int] = {}
    for s in samples:
        label_counts[s.target_action] = int(label_counts.get(s.target_action, 0) + 1)

    safe_per_device_batch, safe_num_generations, variance_guard_notes = (
        _apply_reward_variance_guard(
            per_device_train_batch_size=per_device_train_batch_size,
            num_generations=num_generations,
            scale_rewards=scale_rewards,
            dataset_size=len(samples),
            label_counts=label_counts,
            reward_variance_guard=reward_variance_guard,
        )
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if dry_run:
        preview = {
            "model": chosen_model,
            "num_samples": len(samples),
            "source": source,
            "timeframe": timeframe,
            "window_size": window_size,
            "sample_mode": sample_mode,
            "sample_seed": sample_seed,
            "action_schema": str(action_schema),
            "prompt_style": str(prompt_style),
            "prompt_feature_mode": str(prompt_feature_mode),
            "target_horizon": int(target_horizon),
            "label_mode": str(label_mode),
            "label_counts": label_counts,
            "buy_reward_weight": float(buy_reward_weight),
            "hold_reward_weight": float(hold_reward_weight),
            "sell_reward_weight": float(sell_reward_weight),
            "reward_mode": str(reward_mode),
            "utility_hold_margin": float(utility_hold_margin),
            "utility_fee_rate": float(utility_fee_rate),
            "utility_slippage_rate": float(utility_slippage_rate),
            "utility_leverage": float(utility_leverage),
            "utility_stop_loss": None if utility_stop_loss is None else float(utility_stop_loss),
            "utility_take_profit": None
            if utility_take_profit is None
            else float(utility_take_profit),
            "utility_use_log_return": bool(utility_use_log_return),
            "utility_base_risk_weight": float(utility_base_risk_weight),
            "utility_regime_weight_volatility": float(utility_regime_weight_volatility),
            "utility_regime_weight_downtrend": float(utility_regime_weight_downtrend),
            "utility_regime_weight_drawdown": float(utility_regime_weight_drawdown),
            "utility_min_risk_weight": float(utility_min_risk_weight),
            "utility_max_risk_weight": float(utility_max_risk_weight),
            "utility_hold_reward_bias": float(utility_hold_reward_bias),
            "utility_reward_scale": float(utility_reward_scale),
            "utility_gap_scale": float(utility_gap_scale),
            "reward_variance_guard": str(reward_variance_guard),
            "effective_per_device_train_batch_size": int(safe_per_device_batch),
            "effective_num_generations": int(safe_num_generations),
            "variance_guard_notes": variance_guard_notes,
            "example_prompt": samples[0].prompt,
            "example_target_action": samples[0].target_action,
            "example_next_return": samples[0].next_return,
        }
        preview_file = out / "dry_run_preview.json"
        preview_file.write_text(json.dumps(preview, indent=2))
        return str(preview_file.resolve())

    # Lazy imports to avoid requiring heavy deps unless actually used.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from trl import GRPOConfig, GRPOTrainer

    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    processor = AutoProcessor.from_pretrained(chosen_model, trust_remote_code=True)
    with disable_transformers_allocator_warmup():
        model = AutoModelForImageTextToText.from_pretrained(
            chosen_model,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    dataset = Dataset.from_list(samples_to_hf_records(samples, action_schema=action_schema))
    dataset = dataset.shuffle(seed=int(sample_seed))

    for note in variance_guard_notes:
        print(f"[train-vlm-smoke][variance-guard] {note}")
    # TRL requires generation_batch_size % num_generations == 0.
    # For smoke defaults (batch=1, num_generations=2), this becomes 2.
    generation_batch_size = (
        (safe_per_device_batch + safe_num_generations - 1)
        // safe_num_generations
    ) * safe_num_generations

    train_cfg = GRPOConfig(
        output_dir=str(out),
        learning_rate=learning_rate,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=safe_per_device_batch,
        gradient_accumulation_steps=1,
        num_generations=safe_num_generations,
        generation_batch_size=generation_batch_size,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        max_completion_length=max(1, int(max_completion_length)),
        generation_kwargs={
            "min_new_tokens": max(1, int(min_new_tokens)),
            "do_sample": bool(do_sample),
            "renormalize_logits": True,
        },
        log_completions=bool(log_completions),
        num_completions_to_print=max(1, int(num_completions_to_print)),
        scale_rewards=scale_rewards,
        logging_steps=1,
        save_steps=max(1, max_steps),
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=make_grpo_reward_func(
            hold_band=hold_band,
            buy_reward_weight=buy_reward_weight,
            hold_reward_weight=hold_reward_weight,
            sell_reward_weight=sell_reward_weight,
            reward_mode=reward_mode,
            utility_reward_scale=utility_reward_scale,
            utility_gap_scale=utility_gap_scale,
            action_schema=action_schema,
        ),
        train_dataset=dataset,
        args=train_cfg,
        peft_config=peft_cfg,
    )
    trainer.train()

    trainer.save_model(str(out))
    run_meta = {
        "model": chosen_model,
        "num_samples": int(len(samples)),
        "label_counts": label_counts,
        "sample_mode": str(sample_mode),
        "sample_seed": int(sample_seed),
        "action_schema": str(action_schema),
        "prompt_style": str(prompt_style),
        "prompt_feature_mode": str(prompt_feature_mode),
        "target_horizon": int(target_horizon),
        "label_mode": str(label_mode),
        "buy_reward_weight": float(buy_reward_weight),
        "hold_reward_weight": float(hold_reward_weight),
        "sell_reward_weight": float(sell_reward_weight),
        "reward_mode": str(reward_mode),
        "utility_hold_margin": float(utility_hold_margin),
        "utility_fee_rate": float(utility_fee_rate),
        "utility_slippage_rate": float(utility_slippage_rate),
        "utility_leverage": float(utility_leverage),
        "utility_stop_loss": None if utility_stop_loss is None else float(utility_stop_loss),
        "utility_take_profit": None if utility_take_profit is None else float(utility_take_profit),
        "utility_use_log_return": bool(utility_use_log_return),
        "utility_base_risk_weight": float(utility_base_risk_weight),
        "utility_regime_weight_volatility": float(utility_regime_weight_volatility),
        "utility_regime_weight_downtrend": float(utility_regime_weight_downtrend),
        "utility_regime_weight_drawdown": float(utility_regime_weight_drawdown),
        "utility_min_risk_weight": float(utility_min_risk_weight),
        "utility_max_risk_weight": float(utility_max_risk_weight),
        "utility_hold_reward_bias": float(utility_hold_reward_bias),
        "utility_reward_scale": float(utility_reward_scale),
        "utility_gap_scale": float(utility_gap_scale),
        "reward_variance_guard": str(reward_variance_guard),
        "effective_per_device_train_batch_size": int(safe_per_device_batch),
        "effective_num_generations": int(safe_num_generations),
        "scale_rewards": str(scale_rewards),
        "max_completion_length": int(max_completion_length),
        "min_new_tokens": int(min_new_tokens),
        "do_sample": bool(do_sample),
        "log_completions": bool(log_completions),
        "num_completions_to_print": int(num_completions_to_print),
        "variance_guard_notes": variance_guard_notes,
    }
    (out / "train_config.json").write_text(json.dumps(run_meta, indent=2))
    return str(out.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Option B VLM with TRL GRPO.")
    parser.add_argument("--model-name", type=str, default=AUTO_MODEL_NAME)
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
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument(
        "--action-schema",
        type=str,
        default="buy_hold_sell",
        choices=sorted(ACTION_SCHEMA_LABELS),
    )
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
        choices=["basic_v0", "engineered_v1"],
        help="Prompt feature set: legacy prompt fields only or engineered LLM-oriented features.",
    )
    parser.add_argument("--hold-band", type=float, default=0.0005)
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument(
        "--label-mode",
        type=str,
        default="next_return",
        choices=["next_return", "utility"],
        help="Target label source: raw horizon sign or cost/risk-aware utility argmax.",
    )
    parser.add_argument("--buy-reward-weight", type=float, default=1.0)
    parser.add_argument("--hold-reward-weight", type=float, default=1.0)
    parser.add_argument("--sell-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="classification",
        choices=["classification", "utility"],
        help="GRPO reward style: action classification or utility/regret shaping.",
    )
    parser.add_argument("--utility-hold-margin", type=float, default=0.0)
    parser.add_argument("--utility-fee-rate", type=float, default=0.0005)
    parser.add_argument("--utility-slippage-rate", type=float, default=0.0001)
    parser.add_argument("--utility-leverage", type=float, default=1.0)
    parser.add_argument(
        "--utility-stop-loss",
        type=float,
        default=-1.0,
        help=">0 enables utility clipping at -stop_loss, <=0 disables.",
    )
    parser.add_argument(
        "--utility-take-profit",
        type=float,
        default=-1.0,
        help=">0 enables utility clipping at +take_profit, <=0 disables.",
    )
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
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints/vlm_grpo_smoke")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-completion-length", type=int, default=8)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--do-sample", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--log-completions", action="store_true", default=False)
    parser.add_argument("--num-completions-to-print", type=int, default=2)
    parser.add_argument(
        "--scale-rewards",
        type=str,
        default="batch",
        choices=["group", "batch", "none"],
        help="GRPO reward scaling mode. batch/none helps when group reward_std collapses to zero.",
    )
    parser.add_argument(
        "--reward-variance-guard",
        type=str,
        default="auto",
        choices=["auto", "off"],
        help="Auto-adjust tiny smoke batches to avoid zero-variance reward updates.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--allow-fallback", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utility_stop_loss = (
        None if float(args.utility_stop_loss) <= 0.0 else float(args.utility_stop_loss)
    )
    utility_take_profit = (
        None if float(args.utility_take_profit) <= 0.0 else float(args.utility_take_profit)
    )
    out = train_vlm_grpo_smoke(
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
        max_samples=args.max_samples,
        action_schema=args.action_schema,
        prompt_style=args.prompt_style,
        prompt_feature_mode=args.prompt_feature_mode,
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
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_completion_length=args.max_completion_length,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample == "true",
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        scale_rewards=args.scale_rewards,
        reward_variance_guard=args.reward_variance_guard,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        allow_fallback=args.allow_fallback,
        dry_run=args.dry_run,
    )
    print(f"Saved VLM GRPO checkpoint to: {out}")


if __name__ == "__main__":
    main()
