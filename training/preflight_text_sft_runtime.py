"""Preflight check for local text SFT runtime dependencies."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.train_text_sft import TextSFTConfig, train_text_sft


@dataclass(frozen=True)
class PreflightCfg:
    train_jsonl: str
    output: str
    output_dir: str = "checkpoints/text_sft_preflight_dryrun"
    model_name: str = "gemma4-e4b"
    max_samples: int = 192
    sample_mode: str = "balanced"
    max_seq_length: int = 1536


def _module_status() -> dict[str, bool]:
    return {m: importlib.util.find_spec(m) is not None for m in ("numpy", "torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes")}


def _nvidia_smi() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
        return {"available": True, "gpus": [line.strip() for line in out.splitlines() if line.strip()]}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def run(cfg: PreflightCfg) -> dict[str, Any]:
    dry = train_text_sft(
        TextSFTConfig(
            model_name=cfg.model_name,
            train_jsonl=cfg.train_jsonl,
            output_dir=cfg.output_dir,
            max_samples=cfg.max_samples,
            sample_mode=cfg.sample_mode,
            max_seq_length=cfg.max_seq_length,
            max_steps=1,
        ),
        dry_run=True,
    )
    mods = _module_status()
    required = ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "python": {"executable": sys.executable, "version": sys.version, "platform": platform.platform()},
        "modules": mods,
        "gpu": _nvidia_smi(),
        "dry_run": dry,
        "can_train_now": all(mods.get(m, False) for m in required),
        "missing_for_training": [m for m in required if not mods.get(m, False)],
        "note": "dry_run validates data/model alias only; can_train_now requires local ML dependencies in this Python environment",
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight text SFT runtime")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--output-dir", default=PreflightCfg.output_dir)
    p.add_argument("--model-name", default="gemma4-e4b")
    p.add_argument("--max-samples", type=int, default=192)
    p.add_argument("--sample-mode", choices=["sequential", "random", "balanced", "gate_balanced"], default="balanced")
    p.add_argument("--max-seq-length", type=int, default=1536)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(PreflightCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
