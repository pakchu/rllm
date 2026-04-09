"""Chart image generation utilities (black theme, leak-safe windowing)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from preprocessing.indicators import bollinger_bands, envelopes, mfi, rsi, sma
from preprocessing.timeframe import make_window


_COLOR_BULL = np.asarray([38, 166, 154], dtype=np.float32) / 255.0
_COLOR_BEAR = np.asarray([239, 83, 80], dtype=np.float32) / 255.0
_COLOR_LEVEL = np.asarray([68, 68, 68], dtype=np.float32) / 255.0
_COLOR_RSI = np.asarray([187, 134, 252], dtype=np.float32) / 255.0
_COLOR_MFI = np.asarray([3, 218, 198], dtype=np.float32) / 255.0
_OVERLAY_COLORS = (
    np.asarray([121, 134, 203], dtype=np.float32) / 255.0,
    np.asarray([255, 213, 79], dtype=np.float32) / 255.0,
    np.asarray([255, 138, 101], dtype=np.float32) / 255.0,
    np.asarray([129, 199, 132], dtype=np.float32) / 255.0,
    np.asarray([79, 195, 247], dtype=np.float32) / 255.0,
    np.asarray([244, 143, 177], dtype=np.float32) / 255.0,
)


@dataclass
class ChartGeneratorConfig:
    """Renderer configuration."""

    resolution: int = 320
    show_indicators: bool = True
    show_oscillators: bool = True
    sma_lengths: tuple[int, ...] = (5, 20, 60)
    bb_length: int = 20
    bb_sigma: float = 2.0
    env_length: int = 96
    env_pct: tuple[float, ...] = (1.0, 3.0)
    cache_dir: Optional[str] = None
    backend: str = "matplotlib"


class ChartGenerator:
    """
    Render candlestick chart images from OHLCV windows.

    Output:
      - `np.ndarray` of shape `(3, resolution, resolution)`, float32, [0, 1]
    """

    def __init__(self, config: Optional[ChartGeneratorConfig] = None) -> None:
        self.config = config or ChartGeneratorConfig()
        self.backend = str(self.config.backend).lower().strip()
        if self.backend not in {"matplotlib", "fast"}:
            raise ValueError(
                f"Unsupported chart backend '{self.config.backend}'. "
                "Expected one of {'matplotlib', 'fast'}."
            )
        self._cache_path: Optional[Path] = (
            Path(self.config.cache_dir) if self.config.cache_dir else None
        )
        if self._cache_path:
            self._cache_path.mkdir(parents=True, exist_ok=True)

    def _semantic_config(self) -> dict:
        config_dict = asdict(self.config)
        config_dict.pop("cache_dir", None)
        return config_dict

    def _cache_key(self, window_df: pd.DataFrame) -> str:
        values = window_df[["open", "high", "low", "close", "volume"]].to_numpy(
            dtype=np.float64
        )
        payload = values.tobytes() + json.dumps(
            self._semantic_config(), sort_keys=True
        ).encode("utf-8")
        return hashlib.md5(payload).hexdigest()

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        if not self._cache_path:
            return None
        cache_file = self._cache_path / f"{key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                # Treat corrupted / truncated cache artifacts as cache misses.
                # This keeps long-running eval jobs alive by regenerating bad entries.
                try:
                    cache_file.unlink()
                except FileNotFoundError:
                    pass
                return None
        return None

    def _save_cache(self, key: str, image: np.ndarray) -> None:
        if not self._cache_path:
            return
        cache_file = self._cache_path / f"{key}.npy"
        np.save(cache_file, image)

    @staticmethod
    def _draw_candles(ax: plt.Axes, window_df: pd.DataFrame) -> None:
        x = np.arange(len(window_df))
        opens = window_df["open"].to_numpy(dtype=np.float64)
        highs = window_df["high"].to_numpy(dtype=np.float64)
        lows = window_df["low"].to_numpy(dtype=np.float64)
        closes = window_df["close"].to_numpy(dtype=np.float64)

        for i in range(len(window_df)):
            color = "#26A69A" if closes[i] >= opens[i] else "#EF5350"
            ax.plot([x[i], x[i]], [lows[i], highs[i]], color=color, linewidth=1.0)
            body_low = min(opens[i], closes[i])
            body_high = max(opens[i], closes[i])
            height = max(body_high - body_low, 1e-9)
            rect = Rectangle(
                (x[i] - 0.35, body_low),
                0.7,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.0,
            )
            ax.add_patch(rect)

    @staticmethod
    def _value_to_row(
        values: np.ndarray,
        y_min: float,
        y_max: float,
        row_top: int,
        row_bottom: int,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if row_bottom <= row_top:
            return np.full(values.shape, row_top, dtype=np.int32)
        span = max(float(y_max) - float(y_min), 1e-12)
        norm = (values - float(y_min)) / span
        rows = row_bottom - norm * (row_bottom - row_top)
        rows = np.rint(rows).astype(np.int32)
        return np.clip(rows, row_top, row_bottom)

    @staticmethod
    def _x_centers(size: int, count: int) -> np.ndarray:
        if count <= 1:
            return np.array([size // 2], dtype=np.int32)
        return np.rint(np.linspace(0, size - 1, count)).astype(np.int32)

    @staticmethod
    def _paint_points(
        canvas: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        color: np.ndarray,
        thickness: int = 1,
    ) -> None:
        radius = max(0, int(thickness) // 2)
        xs = np.asarray(xs, dtype=np.int32)
        ys = np.asarray(ys, dtype=np.int32)
        h, w, _ = canvas.shape
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                xx = xs + dx
                yy = ys + dy
                mask = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                if np.any(mask):
                    canvas[yy[mask], xx[mask]] = color

    @classmethod
    def _draw_line_fast(
        cls,
        canvas: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: np.ndarray,
        thickness: int = 1,
    ) -> None:
        steps = int(max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0)))) + 1
        xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int32)
        ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int32)
        cls._paint_points(canvas, xs, ys, color=color, thickness=thickness)

    @classmethod
    def _draw_polyline_fast(
        cls,
        canvas: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        color: np.ndarray,
        thickness: int = 1,
    ) -> None:
        if len(xs) < 2:
            return
        for idx in range(1, len(xs)):
            x0 = xs[idx - 1]
            x1 = xs[idx]
            y0 = ys[idx - 1]
            y1 = ys[idx]
            if (
                not np.isfinite(x0)
                or not np.isfinite(x1)
                or not np.isfinite(y0)
                or not np.isfinite(y1)
            ):
                continue
            cls._draw_line_fast(
                canvas,
                x0=int(x0),
                y0=int(y0),
                x1=int(x1),
                y1=int(y1),
                color=color,
                thickness=thickness,
            )

    @staticmethod
    def _draw_rect_fast(
        canvas: np.ndarray,
        x_center: int,
        width: int,
        y0: int,
        y1: int,
        color: np.ndarray,
    ) -> None:
        h, w, _ = canvas.shape
        half = max(0, int(width) // 2)
        left = max(0, int(x_center) - half)
        right = min(w - 1, int(x_center) + half)
        top = max(0, min(int(y0), int(y1)))
        bottom = min(h - 1, max(int(y0), int(y1)))
        if left <= right and top <= bottom:
            canvas[top : bottom + 1, left : right + 1] = color

    def _indicator_payload(self, window_df: pd.DataFrame) -> dict:
        close = window_df["close"].astype(float)
        high = window_df["high"].astype(float)
        low = window_df["low"].astype(float)
        volume = window_df["volume"].astype(float)

        price_lines: list[tuple[np.ndarray, np.ndarray, float]] = []
        overlay_values: list[np.ndarray] = []
        color_idx = 0

        for length in self.config.sma_lengths:
            if length <= len(close):
                series = sma(close, length=length).to_numpy(dtype=np.float64)
                price_lines.append((series, _OVERLAY_COLORS[color_idx % len(_OVERLAY_COLORS)], 1.0))
                overlay_values.append(series)
                color_idx += 1

        if self.config.bb_length <= len(close):
            bb = bollinger_bands(
                close, length=self.config.bb_length, sigma=self.config.bb_sigma
            )
            for key in ("bb_upper", "bb_lower"):
                series = bb[key].to_numpy(dtype=np.float64)
                price_lines.append(
                    (series, _OVERLAY_COLORS[color_idx % len(_OVERLAY_COLORS)], 1.0)
                )
                overlay_values.append(series)
                color_idx += 1

        for pct in self.config.env_pct:
            if self.config.env_length <= len(close):
                env = envelopes(close, length=self.config.env_length, pct=pct)
                for key in ("env_upper", "env_lower"):
                    series = env[key].to_numpy(dtype=np.float64)
                    price_lines.append(
                        (series, _OVERLAY_COLORS[color_idx % len(_OVERLAY_COLORS)], 1.0)
                    )
                    overlay_values.append(series)
                    color_idx += 1

        lows_arr = low.to_numpy(dtype=np.float64)
        highs_arr = high.to_numpy(dtype=np.float64)
        y_min = float(np.nanmin(lows_arr))
        y_max = float(np.nanmax(highs_arr))
        valid_overlay = [arr[np.isfinite(arr)] for arr in overlay_values if np.isfinite(arr).any()]
        if valid_overlay:
            merged = np.concatenate(valid_overlay)
            if merged.size > 0:
                y_min = min(y_min, float(np.nanmin(merged)))
                y_max = max(y_max, float(np.nanmax(merged)))

        oscillator_lines: list[tuple[np.ndarray, np.ndarray, float]] = []
        if self.config.show_oscillators:
            oscillator_lines = [
                (rsi(close, length=14).to_numpy(dtype=np.float64), _COLOR_RSI, 1.0),
                (mfi(high, low, close, volume, length=14).to_numpy(dtype=np.float64), _COLOR_MFI, 1.0),
            ]

        return {
            "price_lines": price_lines,
            "oscillator_lines": oscillator_lines,
            "y_min": y_min,
            "y_max": y_max,
        }

    def _plot_overlays(
        self, ax_price: plt.Axes, ax_osc: Optional[plt.Axes], window_df: pd.DataFrame
    ) -> None:
        payload = self._indicator_payload(window_df)
        x = np.arange(len(window_df))
        for series, color, width in payload["price_lines"]:
            ax_price.plot(x, series, linewidth=width, alpha=0.8, color=tuple(color))

        y_min = float(payload["y_min"])
        y_max = float(payload["y_max"])
        pad = (y_max - y_min) * 0.03 if y_max > y_min else max(y_max * 0.01, 1.0)
        ax_price.set_ylim(y_min - pad, y_max + pad)

        if ax_osc is not None:
            for series, color, width in payload["oscillator_lines"]:
                ax_osc.plot(x, series, linewidth=width, color=tuple(color), alpha=0.9)
            for level in (5, 10, 20, 50, 80, 90, 95):
                ax_osc.axhline(level, color=tuple(_COLOR_LEVEL), linewidth=0.7, alpha=0.5)
            ax_osc.set_ylim(0, 100)

    @staticmethod
    def _style_axis(ax: plt.Axes) -> None:
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def render_window(self, window_df: pd.DataFrame) -> np.ndarray:
        """Render one window dataframe into chart image tensor."""
        key = self._cache_key(window_df)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        req_cols = {"open", "high", "low", "close", "volume"}
        missing = req_cols.difference(window_df.columns)
        if missing:
            raise ValueError(f"window_df missing columns: {sorted(missing)}")

        if self.backend == "fast":
            chw = self._render_window_fast(window_df)
            self._save_cache(key, chw)
            return chw

        size = int(self.config.resolution)
        dpi = 100
        fig = plt.figure(
            figsize=(size / dpi, size / dpi),
            dpi=dpi,
            facecolor="black",
            constrained_layout=False,
        )
        gs = fig.add_gridspec(10, 1, hspace=0.0)
        ax_price = fig.add_subplot(gs[:7, 0], facecolor="black")
        ax_osc = (
            fig.add_subplot(gs[7:, 0], sharex=ax_price, facecolor="black")
            if self.config.show_oscillators
            else None
        )

        self._draw_candles(ax_price, window_df)
        if self.config.show_indicators:
            self._plot_overlays(ax_price, ax_osc, window_df)
        ax_price.set_xlim(-1, len(window_df))
        self._style_axis(ax_price)
        if ax_osc is not None:
            self._style_axis(ax_osc)

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.0)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = rgba[..., :3].copy()
        plt.close(fig)

        chw = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        self._save_cache(key, chw)
        return chw

    def _render_window_fast(self, window_df: pd.DataFrame) -> np.ndarray:
        size = int(self.config.resolution)
        canvas = np.zeros((size, size, 3), dtype=np.float32)
        count = len(window_df)
        xs = self._x_centers(size=size, count=count)

        open_arr = window_df["open"].to_numpy(dtype=np.float64)
        high_arr = window_df["high"].to_numpy(dtype=np.float64)
        low_arr = window_df["low"].to_numpy(dtype=np.float64)
        close_arr = window_df["close"].to_numpy(dtype=np.float64)

        osc_enabled = bool(self.config.show_indicators and self.config.show_oscillators)
        price_bottom = max(0, int(round(size * 0.7)) - 1) if osc_enabled else size - 1
        osc_top = min(size - 1, price_bottom + 1) if osc_enabled else None
        osc_bottom = size - 1

        if self.config.show_indicators:
            payload = self._indicator_payload(window_df)
            y_min = float(payload["y_min"])
            y_max = float(payload["y_max"])
        else:
            y_min = float(np.nanmin(low_arr))
            y_max = float(np.nanmax(high_arr))
            payload = {"price_lines": [], "oscillator_lines": []}

        pad = (y_max - y_min) * 0.03 if y_max > y_min else max(abs(y_max) * 0.01, 1.0)
        y_min -= pad
        y_max += pad

        row_open = self._value_to_row(open_arr, y_min, y_max, 0, price_bottom)
        row_high = self._value_to_row(high_arr, y_min, y_max, 0, price_bottom)
        row_low = self._value_to_row(low_arr, y_min, y_max, 0, price_bottom)
        row_close = self._value_to_row(close_arr, y_min, y_max, 0, price_bottom)

        body_width = max(1, int(round(size / max(count, 1) * 0.8)))
        for idx, x_center in enumerate(xs):
            color = _COLOR_BULL if close_arr[idx] >= open_arr[idx] else _COLOR_BEAR
            self._draw_line_fast(
                canvas,
                x0=int(x_center),
                y0=int(row_high[idx]),
                x1=int(x_center),
                y1=int(row_low[idx]),
                color=color,
                thickness=1,
            )
            self._draw_rect_fast(
                canvas,
                x_center=int(x_center),
                width=body_width,
                y0=int(row_open[idx]),
                y1=int(row_close[idx]),
                color=color,
            )

        for series, color, width in payload["price_lines"]:
            valid = np.isfinite(series)
            if not np.any(valid):
                continue
            ys = self._value_to_row(series[valid], y_min, y_max, 0, price_bottom)
            self._draw_polyline_fast(
                canvas,
                xs=xs[valid],
                ys=ys,
                color=color,
                thickness=max(1, int(round(width))),
            )

        if osc_enabled and osc_top is not None and osc_top <= osc_bottom:
            level_rows = self._value_to_row(
                np.asarray([5, 10, 20, 50, 80, 90, 95], dtype=np.float64),
                0.0,
                100.0,
                osc_top,
                osc_bottom,
            )
            for row in level_rows:
                self._draw_line_fast(
                    canvas,
                    x0=0,
                    y0=int(row),
                    x1=size - 1,
                    y1=int(row),
                    color=_COLOR_LEVEL,
                    thickness=1,
                )
            for series, color, width in payload["oscillator_lines"]:
                valid = np.isfinite(series)
                if not np.any(valid):
                    continue
                ys = self._value_to_row(series[valid], 0.0, 100.0, osc_top, osc_bottom)
                self._draw_polyline_fast(
                    canvas,
                    xs=xs[valid],
                    ys=ys,
                    color=color,
                    thickness=max(1, int(round(width))),
                )

        return canvas.transpose(2, 0, 1).astype(np.float32)

    def render_at(self, market_df: pd.DataFrame, t: int, w: int = 96) -> np.ndarray:
        """Render leak-safe image using window `[t-w+1, t]`."""
        window_df = make_window(market_df, t=t, w=w)
        return self.render_window(window_df)

    def build_image_dataset(
        self, market_df: pd.DataFrame, window_size: int = 96
    ) -> np.ndarray:
        """Render image dataset for all valid timesteps."""
        images = []
        for t in range(window_size - 1, len(market_df)):
            images.append(self.render_at(market_df, t=t, w=window_size))
        return np.stack(images, axis=0)


def build_images(
    market_df: pd.DataFrame,
    window_size: int = 96,
    resolution: int = 320,
    cache_dir: Optional[str] = None,
    backend: str = "matplotlib",
) -> np.ndarray:
    """Convenience wrapper to render full dataset images."""
    generator = ChartGenerator(
        ChartGeneratorConfig(
            resolution=resolution,
            cache_dir=cache_dir,
            show_indicators=True,
            backend=backend,
        )
    )
    return generator.build_image_dataset(market_df, window_size=window_size)


def iter_image_batches(
    market_df: pd.DataFrame,
    window_size: int = 96,
    batch_size: int = 32,
    generator: Optional[ChartGenerator] = None,
    backend: str = "matplotlib",
) -> Iterable[np.ndarray]:
    """Yield CHW image batches without loading all samples in memory."""
    gen = generator or ChartGenerator(ChartGeneratorConfig(backend=backend))
    batch = []
    for t in range(window_size - 1, len(market_df)):
        batch.append(gen.render_at(market_df, t=t, w=window_size))
        if len(batch) == batch_size:
            yield np.stack(batch, axis=0)
            batch = []
    if batch:
        yield np.stack(batch, axis=0)
