"""Dataset preparation: chart images + scalar features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from preprocessing.chart_generator import ChartGenerator, ChartGeneratorConfig
from preprocessing.scalars import build_scalar_frame
from training.data_sources import load_market_data


def prepare_observation_dataset(
    market_df: pd.DataFrame,
    window_size: int = 96,
    resolution: int = 320,
    cache_dir: Optional[str] = None,
    image_render_backend: str = "matplotlib",
    position_size_pct: float = 0.0,
    last_entry_price: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Build aligned multimodal dataset.

    Returns:
        Dict with:
          - images: (N, 3, H, W), float32 [0,1]
          - scalars: (N, 3), float32
          - dates: (N,), datetime64[ns]
    """
    renderer = ChartGenerator(
        ChartGeneratorConfig(
            resolution=resolution,
            cache_dir=cache_dir,
            show_indicators=True,
            show_oscillators=True,
            backend=image_render_backend,
        )
    )
    images = renderer.build_image_dataset(market_df, window_size=window_size)

    scalar_df = build_scalar_frame(
        market_df,
        window_size=window_size,
        position_size_pct=position_size_pct,
        last_entry_price=last_entry_price,
    )
    scalars = scalar_df[
        ["position_size_pct", "last_entry_price", "range_volatility_pct"]
    ].to_numpy(dtype=np.float32)
    dates = scalar_df["date"].to_numpy()

    if len(images) != len(scalars):
        raise ValueError(
            f"image/scalar length mismatch: {len(images)} vs {len(scalars)}"
        )
    return {"images": images, "scalars": scalars, "dates": dates}

def save_dataset_npz(dataset: Dict[str, np.ndarray], output_path: str) -> str:
    """Save dataset as compressed NPZ."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=dataset["images"],
        scalars=dataset["scalars"],
        dates=dataset["dates"].astype("datetime64[ns]").astype(str),
    )
    return str(out.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chart+scalar observation data.")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--cache-dir", type=str, default="data/image_cache")
    parser.add_argument(
        "--image-render-backend",
        type=str,
        default="matplotlib",
        choices=["matplotlib", "fast"],
    )
    parser.add_argument("--output", type=str, default="data/observations.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    market_df = load_market_data(
        source=args.source,
        input_csv=args.input_csv or None,
        timeframe=args.timeframe,
        symbol=args.symbol,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        market_type=args.market_type,
        num_rows=args.num_rows,
        seed=args.seed,
    )
    dataset = prepare_observation_dataset(
        market_df=market_df,
        window_size=args.window_size,
        resolution=args.resolution,
        cache_dir=args.cache_dir,
        image_render_backend=args.image_render_backend,
    )
    out = save_dataset_npz(dataset, output_path=args.output)
    print(
        f"Saved dataset to {out} | images={dataset['images'].shape} "
        f"scalars={dataset['scalars'].shape}"
    )


if __name__ == "__main__":
    main()
