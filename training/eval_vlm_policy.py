"""Evaluate Option B VLM policy checkpoints on trading-derived labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from models.option_b_vlm import (
    AUTO_MODEL_NAME,
    ACTION_SCHEMA_LABELS,
    get_action_labels,
    get_default_action_label,
    make_action_system_prompt,
    parse_action_label,
    resolve_vlm_model_alias,
)
from preprocessing.external_features import attach_wave_trading_external_features
from training.data_sources import load_market_data
from training.vlm_trading_data import build_vlm_training_samples
from utils import disable_transformers_allocator_warmup


ACTION_LABELS = ("BUY", "HOLD", "SELL")


def select_action_from_scores(
    scores: dict[str, float],
    action_biases: dict[str, float] | None = None,
    labels: tuple[str, ...] = ACTION_LABELS,
) -> tuple[str, dict[str, float]]:
    """
    Pick action from score table with optional additive class biases.

    Tie-break uses ACTION_LABELS order for deterministic behavior.
    """
    biases = action_biases or {}
    adjusted = {
        label: float(scores.get(label, float("-inf")) + float(biases.get(label, 0.0)))
        for label in labels
    }
    best = max(labels, key=lambda k: (adjusted[k], -labels.index(k)))
    return best, adjusted


def summarize_action_metrics(
    targets: Iterable[str],
    predictions: Iterable[str],
    labels: tuple[str, ...] = ACTION_LABELS,
) -> dict:
    """Compute accuracy + confusion statistics for action labels."""
    t = [str(x).upper() for x in targets]
    p = [str(x).upper() for x in predictions]
    if len(t) != len(p):
        raise ValueError("targets and predictions must have equal length")
    if not t:
        raise ValueError("targets/predictions must not be empty")

    label_to_idx = {k: i for i, k in enumerate(labels)}
    conf = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for yt, yp in zip(t, p):
        if yt not in label_to_idx:
            continue
        if yp not in label_to_idx:
            continue
        conf[label_to_idx[yt], label_to_idx[yp]] += 1

    correct = int(np.trace(conf))
    total = int(np.sum(conf))
    acc = float(correct / max(1, total))

    per_class = {}
    for i, label in enumerate(labels):
        tp = int(conf[i, i])
        fn = int(np.sum(conf[i, :]) - tp)
        fp = int(np.sum(conf[:, i]) - tp)
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        denom = precision + recall
        f1 = float((2 * precision * recall / denom) if denom > 0 else 0.0)
        support = int(np.sum(conf[i, :]))
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    recalls = [float(per_class[label]["recall"]) for label in labels]
    balanced_recall = float(np.mean(recalls)) if recalls else 0.0
    directional_pair = None
    if "BUY" in per_class and "SELL" in per_class:
        directional_pair = ("BUY", "SELL")
    elif "LONG" in per_class and "SHORT" in per_class:
        directional_pair = ("LONG", "SHORT")
    if directional_pair is not None:
        lhs_recall = float(per_class.get(directional_pair[0], {}).get("recall", 0.0))
        rhs_recall = float(per_class.get(directional_pair[1], {}).get("recall", 0.0))
        directional_recall_mean = 0.5 * (lhs_recall + rhs_recall)
        directional_recall_gap = abs(lhs_recall - rhs_recall)
    else:
        directional_recall_mean = 0.0
        directional_recall_gap = 0.0

    target_counts = {k: int(np.sum(conf[label_to_idx[k], :])) for k in labels}
    pred_counts = {k: int(np.sum(conf[:, label_to_idx[k]])) for k in labels}
    confusion = {
        yt: {yp: int(conf[label_to_idx[yt], label_to_idx[yp]]) for yp in labels}
        for yt in labels
    }
    return {
        "accuracy": acc,
        "balanced_recall": balanced_recall,
        "directional_recall_mean": directional_recall_mean,
        "directional_recall_gap": directional_recall_gap,
        "num_samples": total,
        "target_counts": target_counts,
        "pred_counts": pred_counts,
        "confusion": confusion,
        "per_class": per_class,
    }


def _resolve_eval_model_name(model_name: str) -> str:
    return resolve_vlm_model_alias(model_name, prefer_latest=True)


def load_sample_dates(path: str | None) -> list[str] | None:
    """Load evaluation sample dates from JSON report/list or newline text file."""
    if not path:
        return None
    src = Path(path)
    raw = src.read_text().strip()
    if not raw:
        raise ValueError(f"sample date file is empty: {path}")
    dates: list[str] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        rows = data.get("action_scores") or data.get("samples") or data.get("dates")
        if rows is None:
            raise ValueError(
                "sample date JSON object must contain action_scores, samples, or dates"
            )
        data = rows
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                value = item.get("date")
            else:
                value = item
            if value is None:
                continue
            dates.append(str(np.datetime64(str(value).replace(" ", "T"))).replace("T", " "))
    else:
        dates = [line.strip() for line in raw.splitlines() if line.strip()]
    normalized = [str(np.datetime64(str(x).replace(" ", "T"))).replace("T", " ") for x in dates]
    if not normalized:
        raise ValueError(f"No sample dates loaded from: {path}")
    return normalized


def evaluate_vlm_policy(
    model_name: str = AUTO_MODEL_NAME,
    adapter_dir: str | None = None,
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
    wave_trading_root: str = "",
    external_tolerance: str = "",
    window_size: int = 96,
    resolution: int = 224,
    cache_dir: str | None = "data/image_cache_vlm",
    modality: str = "multimodal",
    action_schema: str = "buy_hold_sell",
    trade_side_sample_policy: str = "trade_only",
    prompt_style: str = "numeric",
    prompt_feature_mode: str = "basic_v0",
    hold_band: float = 0.0005,
    target_horizon: int = 1,
    label_mode: str = "next_return",
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
    max_samples: int = 128,
    sample_mode: str = "balanced",
    sample_seed: int = 42,
    sample_date_file: str | None = None,
    max_completion_length: int = 8,
    min_new_tokens: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    decision_mode: str = "generate",
    action_bias_buy: float = 0.0,
    action_bias_hold: float = 0.0,
    action_bias_sell: float = 0.0,
    eval_batch_size: int = 1,
    store_action_scores: bool = False,
    load_in_4bit: bool = False,
    output: str | None = None,
) -> dict:
    """Evaluate VLM (base or LoRA adapter) against next-return-derived labels."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

    chosen_model = _resolve_eval_model_name(model_name)
    labels = get_action_labels(action_schema)
    default_label = get_default_action_label(action_schema)
    system_prompt = make_action_system_prompt(action_schema)
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
    external_columns: list[str] = []
    if wave_trading_root:
        market_df = attach_wave_trading_external_features(
            market_df,
            wave_trading_root=wave_trading_root,
            tolerance=external_tolerance or None,
        )
        external_columns = [
            c
            for c in (
                "dxy",
                "dxy_zscore",
                "dxy_momentum",
                "kimchi_premium",
                "kimchi_premium_zscore",
                "kimchi_premium_change",
                "usdkrw",
                "usdkrw_zscore",
                "usdkrw_momentum",
            )
            if c in market_df.columns
        ]
    requested_sample_dates = load_sample_dates(sample_date_file)
    samples = build_vlm_training_samples(
        market_df=market_df,
        timeframe=timeframe,
        window_size=window_size,
        resolution=resolution,
        cache_dir=cache_dir,
        modality=modality,
        action_schema=action_schema,
        trade_side_sample_policy=trade_side_sample_policy,
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
        max_samples=max_samples,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
        sample_dates=requested_sample_dates,
    )
    if not samples:
        raise ValueError("No evaluation samples generated.")

    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    modality_key = str(modality).lower().strip()
    if modality_key not in {"multimodal", "text_only"}:
        raise ValueError(
            "modality must be one of {'multimodal','text_only'}, "
            f"got {modality}"
        )
    if modality_key == "text_only":
        processor = AutoTokenizer.from_pretrained(chosen_model, trust_remote_code=True, use_fast=False)
        if getattr(processor, 'pad_token_id', None) is None:
            processor.pad_token = processor.eos_token
        with disable_transformers_allocator_warmup():
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    chosen_model,
                    device_map="auto",
                    dtype=torch.bfloat16,
                    quantization_config=quant_cfg,
                    trust_remote_code=True,
                )
            except ValueError as exc:
                if "Unrecognized configuration class" not in str(exc):
                    raise
                processor = AutoProcessor.from_pretrained(chosen_model, trust_remote_code=True)
                model = AutoModelForImageTextToText.from_pretrained(
                    chosen_model,
                    device_map="auto",
                    dtype=torch.bfloat16,
                    quantization_config=quant_cfg,
                    trust_remote_code=True,
                )
        text_processor = getattr(processor, "tokenizer", processor)
    else:
        processor = AutoProcessor.from_pretrained(chosen_model, trust_remote_code=True)
        text_processor = getattr(processor, "tokenizer", processor)
        with disable_transformers_allocator_warmup():
            model = AutoModelForImageTextToText.from_pretrained(
                chosen_model,
                device_map="auto",
                dtype=torch.bfloat16,
                quantization_config=quant_cfg,
                trust_remote_code=True,
            )
    used_adapter = None
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        used_adapter = str(Path(adapter_dir).resolve())

    model.eval()
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    mode = str(decision_mode).strip().lower()
    if mode not in {"generate", "likelihood"}:
        raise ValueError(f"decision_mode must be one of {{'generate','likelihood'}}, got {decision_mode}")

    action_biases = {label: 0.0 for label in labels}
    legacy_biases = {
        "BUY": float(action_bias_buy),
        "HOLD": float(action_bias_hold),
        "SELL": float(action_bias_sell),
    }
    for label, value in legacy_biases.items():
        if label in action_biases:
            action_biases[label] = float(value)

    action_token_ids: dict[str, "torch.Tensor"] = {}
    if mode == "likelihood":
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None and hasattr(processor, "encode"):
            tokenizer = processor
        if tokenizer is None:
            raise ValueError("Processor/tokenizer must expose encode() for likelihood decision_mode.")
        for label in labels:
            ids = tokenizer.encode(label, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(f" {label}", add_special_tokens=False)
            if not ids:
                raise ValueError(f"Could not tokenize action label: {label}")
            action_token_ids[label] = torch.tensor(ids, dtype=torch.long, device=device)

    def _score_actions_likelihood(inputs: dict) -> list[dict[str, float]]:
        import torch.nn.functional as F

        prompt_ids = inputs["input_ids"]
        batch_n = int(prompt_ids.shape[0])
        has_attention = "attention_mask" in inputs
        if has_attention:
            prompt_lens = [int(x) for x in inputs["attention_mask"].sum(dim=1).tolist()]
            attn_dtype = inputs["attention_mask"].dtype
        else:
            prompt_lens = [int(prompt_ids.shape[1])] * batch_n
            attn_dtype = torch.long
        extra_inputs = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}

        inner_tokenizer = getattr(processor, "tokenizer", None)
        if inner_tokenizer is None and hasattr(processor, "pad_token_id"):
            inner_tokenizer = processor
        pad_token_id = getattr(inner_tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        seqs = []
        atts = []
        meta: list[tuple[int, str, int]] = []
        for i in range(batch_n):
            base_ids = prompt_ids[i, : prompt_lens[i]]
            for label in labels:
                a_ids = action_token_ids[label]
                seq = torch.cat([base_ids, a_ids], dim=0)
                seqs.append(seq)
                atts.append(torch.ones(int(seq.shape[0]), dtype=attn_dtype, device=prompt_ids.device))
                meta.append((i, label, int(a_ids.shape[0])))

        max_len = max(int(x.shape[0]) for x in seqs)
        seq_batch = torch.full(
            (len(seqs), max_len),
            fill_value=int(pad_token_id),
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        att_batch = torch.zeros(
            (len(atts), max_len),
            dtype=attn_dtype,
            device=prompt_ids.device,
        )
        for i, (seq, att) in enumerate(zip(seqs, atts)):
            seq_batch[i, : int(seq.shape[0])] = seq
            att_batch[i, : int(att.shape[0])] = att

        packed_inputs = {"input_ids": seq_batch, "attention_mask": att_batch}
        image_grid = extra_inputs.get("image_grid_thw")
        image_patch_counts: list[int] | None = None
        if image_grid is not None:
            image_patch_counts = [int(t * h * w) for t, h, w in image_grid.tolist()]
        for k, v in extra_inputs.items():
            if k == "pixel_values" and image_patch_counts is not None:
                blocks = []
                start = 0
                for count in image_patch_counts:
                    block = v[start : start + count]
                    for _ in labels:
                        blocks.append(block)
                    start += count
                packed_inputs[k] = torch.cat(blocks, dim=0)
            else:
                packed_inputs[k] = v.repeat_interleave(len(labels), dim=0)

        outputs = model(**packed_inputs)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

        scores_per_sample: list[dict[str, float]] = [
            {label: float("-inf") for label in labels} for _ in range(batch_n)
        ]
        for row_idx, (sample_idx, label, act_len) in enumerate(meta):
            a_ids = action_token_ids[label]
            s = 0.0
            prompt_len = prompt_lens[sample_idx]
            for j in range(int(act_len)):
                pos = prompt_len + j
                logit_pos = pos - 1
                tok = int(a_ids[j].item())
                s += float(log_probs[row_idx, logit_pos, tok].item())
            scores_per_sample[sample_idx][label] = float(s / max(1, int(act_len)))
        return scores_per_sample

    preds: list[str] = []
    raw_texts: list[str] = []
    action_scores: list[dict] = []
    targets = [s.target_action for s in samples]

    with torch.inference_mode():
        batch_size = max(1, int(eval_batch_size))
        for batch_start in range(0, len(samples), batch_size):
            batch_samples = samples[batch_start : batch_start + batch_size]
            batch_texts = []
            batch_images = []
            for s in batch_samples:
                if modality_key == "text_only":
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": s.prompt},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": s.prompt}],
                        },
                    ]
                batch_texts.append(
                    text_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
                batch_images.append(s.image)

            if modality_key == "text_only":
                inputs = text_processor(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                ).to(device)
            else:
                inputs = processor(
                    text=batch_texts,
                    images=batch_images,
                    return_tensors="pt",
                    padding=True,
                ).to(device)
            if mode == "generate":
                out = model.generate(
                    **inputs,
                    max_new_tokens=max(1, int(max_completion_length)),
                    min_new_tokens=max(1, int(min_new_tokens)),
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                )
                prompt_lens = (
                    [int(x) for x in inputs["attention_mask"].sum(dim=1).tolist()]
                    if "attention_mask" in inputs
                    else [int(inputs["input_ids"].shape[1])] * len(batch_samples)
                )
                for s, one_out, prompt_len in zip(batch_samples, out, prompt_lens):
                    gen = one_out[int(prompt_len) :]
                    txt = text_processor.decode(gen, skip_special_tokens=True).strip()
                    pred = parse_action_label(txt, default=default_label, labels=labels)
                    preds.append(pred)
                    raw_texts.append(txt)
            else:
                batch_scores = _score_actions_likelihood(inputs)
                for s, scores in zip(batch_samples, batch_scores):
                    pred, adjusted = select_action_from_scores(
                        scores,
                        action_biases=action_biases,
                        labels=labels,
                    )
                    txt = json.dumps({"scores": scores, "adjusted": adjusted}, ensure_ascii=False)
                    preds.append(pred)
                    raw_texts.append(txt)
                    if bool(store_action_scores):
                        action_scores.append(
                            {
                                "date": s.date,
                                "target": s.target_action,
                                "next_return": float(s.next_return),
                                "pred": pred,
                                "scores": {k: float(v) for k, v in scores.items()},
                                "adjusted_scores": {k: float(v) for k, v in adjusted.items()},
                            }
                        )

            done = batch_start + len(batch_samples)
            if done % 20 == 0 or done == len(samples):
                print(f"[eval-vlm] {done}/{len(samples)}")

    metrics = summarize_action_metrics(targets=targets, predictions=preds, labels=labels)
    report = {
        "model": chosen_model,
        "adapter_dir": used_adapter,
        "num_samples": int(len(samples)),
        "source": source,
        "input_csv": str(Path(input_csv).resolve()) if input_csv else None,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "market_type": market_type,
        "source_num_rows": int(len(market_df)),
        "external_features": {
            "wave_trading_root": str(wave_trading_root),
            "external_tolerance": str(external_tolerance),
            "columns": external_columns,
            "join": "backward_asof_no_future" if wave_trading_root else "disabled",
        },
        "timeframe": timeframe,
        "window_size": int(window_size),
        "sample_mode": sample_mode,
        "sample_seed": int(sample_seed),
        "sample_date_file": str(Path(sample_date_file).resolve()) if sample_date_file else None,
        "requested_sample_dates": 0 if requested_sample_dates is None else int(len(requested_sample_dates)),
        "modality": str(modality),
        "action_schema": str(action_schema),
        "prompt_style": str(prompt_style),
        "prompt_feature_mode": str(prompt_feature_mode),
        "hold_band": float(hold_band),
        "target_horizon": int(target_horizon),
        "label_mode": str(label_mode),
        "utility": {
            "hold_margin": float(utility_hold_margin),
            "fee_rate": float(utility_fee_rate),
            "slippage_rate": float(utility_slippage_rate),
            "leverage": float(utility_leverage),
            "stop_loss": None if utility_stop_loss is None else float(utility_stop_loss),
            "take_profit": None
            if utility_take_profit is None
            else float(utility_take_profit),
            "use_log_return": bool(utility_use_log_return),
            "base_risk_weight": float(utility_base_risk_weight),
            "regime_weight_volatility": float(utility_regime_weight_volatility),
            "regime_weight_downtrend": float(utility_regime_weight_downtrend),
            "regime_weight_drawdown": float(utility_regime_weight_drawdown),
            "min_risk_weight": float(utility_min_risk_weight),
            "max_risk_weight": float(utility_max_risk_weight),
            "hold_reward_bias": float(utility_hold_reward_bias),
        },
        "path_outcome": {
            "entry_delay_bars": int(path_entry_delay_bars),
            "mae_penalty": float(path_mae_penalty),
            "mfe_bonus": float(path_mfe_bonus),
            "min_net_return": float(path_min_net_return),
            "max_mae": float(path_max_mae),
            "multi_horizon_bars": str(multi_horizon_bars),
        },
        "decision": {
            "mode": mode,
            "action_biases": action_biases,
            "store_action_scores": bool(store_action_scores),
        },
        "generation": {
            "load_in_4bit": bool(load_in_4bit),
            "max_completion_length": int(max_completion_length),
            "min_new_tokens": int(min_new_tokens),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "eval_batch_size": int(max(1, int(eval_batch_size))),
        },
        "metrics": metrics,
        "examples": [
            {
                "date": samples[k].date,
                "target": targets[k],
                "pred": preds[k],
                "raw": raw_texts[k],
            }
            for k in range(min(8, len(samples)))
        ],
    }
    if mode == "likelihood" and bool(store_action_scores):
        report["action_scores"] = action_scores
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Option B VLM policy checkpoints.")
    parser.add_argument("--model-name", type=str, default=AUTO_MODEL_NAME)
    parser.add_argument("--adapter-dir", type=str, default="")
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
    parser.add_argument(
        "--wave-trading-root",
        type=str,
        default="",
        help="Optional wave_trading root for backward-asof DXY/Kimchi external features.",
    )
    parser.add_argument(
        "--external-tolerance",
        type=str,
        default="",
        help="Optional pandas Timedelta tolerance for external feature joins.",
    )
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--cache-dir", type=str, default="data/image_cache_vlm")
    parser.add_argument(
        "--modality",
        type=str,
        default="multimodal",
        choices=["multimodal", "text_only"],
    )
    parser.add_argument(
        "--action-schema",
        type=str,
        default="buy_hold_sell",
        choices=sorted(ACTION_SCHEMA_LABELS),
    )
    parser.add_argument(
        "--trade-side-sample-policy",
        type=str,
        default="trade_only",
        choices=["trade_only", "directional_all"],
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
        choices=["basic_v0", "engineered_v1", "edge_state_v2"],
    )
    parser.add_argument("--hold-band", type=float, default=0.0005)
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument(
        "--label-mode",
        type=str,
        default="next_return",
        choices=["next_return", "utility", "path_outcome"],
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
    parser.add_argument("--path-entry-delay-bars", type=int, default=1)
    parser.add_argument("--path-mae-penalty", type=float, default=1.0)
    parser.add_argument("--path-mfe-bonus", type=float, default=0.0)
    parser.add_argument("--path-min-net-return", type=float, default=0.0)
    parser.add_argument("--path-max-mae", type=float, default=1.0)
    parser.add_argument(
        "--multi-horizon-bars",
        type=str,
        default="36,72,144",
        help="Comma-separated hold bars for action_schema=multi_horizon_side.",
    )
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="balanced",
        choices=["sequential", "random", "balanced", "uniform"],
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sample-date-file",
        type=str,
        default="",
        help="Optional JSON report/list or newline file of exact sample dates to evaluate.",
    )
    parser.add_argument("--max-completion-length", type=int, default=8)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--do-sample", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--decision-mode",
        type=str,
        default="generate",
        choices=["generate", "likelihood"],
    )
    parser.add_argument("--action-bias-buy", type=float, default=0.0)
    parser.add_argument("--action-bias-hold", type=float, default=0.0)
    parser.add_argument("--action-bias-sell", type=float, default=0.0)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--store-action-scores",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="results/eval_vlm.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utility_stop_loss = (
        None if float(args.utility_stop_loss) <= 0.0 else float(args.utility_stop_loss)
    )
    utility_take_profit = (
        None if float(args.utility_take_profit) <= 0.0 else float(args.utility_take_profit)
    )
    report = evaluate_vlm_policy(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir or None,
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
        wave_trading_root=args.wave_trading_root,
        external_tolerance=args.external_tolerance,
        window_size=args.window_size,
        resolution=args.resolution,
        cache_dir=args.cache_dir or None,
        modality=args.modality,
        action_schema=args.action_schema,
        trade_side_sample_policy=args.trade_side_sample_policy,
        prompt_style=args.prompt_style,
        prompt_feature_mode=args.prompt_feature_mode,
        hold_band=args.hold_band,
        target_horizon=args.target_horizon,
        label_mode=args.label_mode,
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
        path_entry_delay_bars=args.path_entry_delay_bars,
        path_mae_penalty=args.path_mae_penalty,
        path_mfe_bonus=args.path_mfe_bonus,
        path_min_net_return=args.path_min_net_return,
        path_max_mae=args.path_max_mae,
        multi_horizon_bars=args.multi_horizon_bars,
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        sample_date_file=args.sample_date_file or None,
        max_completion_length=args.max_completion_length,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample == "true",
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        decision_mode=args.decision_mode,
        action_bias_buy=args.action_bias_buy,
        action_bias_hold=args.action_bias_hold,
        action_bias_sell=args.action_bias_sell,
        eval_batch_size=args.eval_batch_size,
        store_action_scores=args.store_action_scores == "true",
        load_in_4bit=args.load_in_4bit,
        output=args.output or None,
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
