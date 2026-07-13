#!/usr/bin/env python3
"""Strictly causal pre-2024 prototype: leveraged-inventory conservation residual alpha.

The experiment physically truncates every source
before 2024, fits residual parameters/thresholds on data through 2022, selects
only on 2023 and both 2023 halves, and uses next-bar-open execution with the
canonical strict OHLC intratrade MDD simulator.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import (
    attach_binance_um_aux_frames,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)
from training.search_positioning_disagreement_alpha import (
    _attach_delayed_metrics,
    _future_extreme,
    _simulate_no_stop,
)
from training.search_positioning_hgb_path_alpha import _read_before

SELECTION_CUTOFF = '2024-01-01'
WINDOWS = {
    'fit_through_2022': ('2020-10-15', '2023-01-01'),
    'select_2023': ('2023-01-01', '2024-01-01'),
    'select_2023_h1': ('2023-01-01', '2023-07-01'),
    'select_2023_h2': ('2023-07-01', '2024-01-01'),
}
SEGMENT_WINDOWS = {
    '2023Q1': ('2023-01-01', '2023-04-01'),
    '2023Q2': ('2023-04-01', '2023-07-01'),
    '2023Q3': ('2023-07-01', '2023-10-01'),
    '2023Q4': ('2023-10-01', '2024-01-01'),
}
OOS_WINDOWS = {
    'test_2024': ('2024-01-01', '2025-01-01'),
    'test_2024_h1': ('2024-01-01', '2024-07-01'),
    'test_2024_h2': ('2024-07-01', '2025-01-01'),
    'eval_2025': ('2025-01-01', '2026-01-01'),
    'eval_2025_h1': ('2025-01-01', '2025-07-01'),
    'eval_2025_h2': ('2025-07-01', '2026-01-01'),
    'holdout_2026': ('2026-01-01', '2026-06-02'),
    'holdout_2026_q1': ('2026-01-01', '2026-04-01'),
    'holdout_2026_q2': ('2026-04-01', '2026-06-02'),
    'combined_oos': ('2024-01-01', '2026-06-02'),
}


@dataclass(frozen=True)
class Config:
    market_csv: str = str(REPO / 'data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz')
    metrics_csv: str = str(REPO / 'data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz')
    funding_csv: str = str(REPO / 'data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz')
    premium_csv: str = str(REPO / 'data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz')
    output_json: str = str(REPO / 'results/inventory_conservation_residual_alpha_scan_2026-07-13.json')
    manifest_output: str = str(REPO / 'results/inventory_conservation_residual_frozen_manifest_2026-07-13.json')
    verification_output: str = str(REPO / 'results/inventory_conservation_residual_replay_verification_2026-07-13.json')
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stride_bars: int = 12
    metrics_tolerance: str = '5min'
    metrics_delay_bars: int = 1
    funding_tolerance: str = '12h'
    premium_tolerance: str = '65min'


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update('\n'.join(map(str, frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype='<u8').tobytes())
    return h.hexdigest()


def _signal_hash(long_active: np.ndarray, short_active: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(long_active, np.uint8).tobytes())
    h.update(np.asarray(short_active, np.uint8).tobytes())
    return h.hexdigest()


def _read_premium_before(path: str, cutoff: str) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    boundary = pd.Timestamp(cutoff)
    for chunk in pd.read_csv(path, compression='infer', chunksize=100_000):
        if 'close_time' in chunk.columns:
            close_time = pd.to_datetime(pd.to_numeric(chunk['close_time'], errors='raise'), unit='ms', utc=True).dt.tz_convert(None)
            keep = close_time < boundary
        else:
            dates = pd.to_datetime(chunk['date'], utc=True, errors='raise', format='mixed').dt.tz_convert(None)
            keep = dates < boundary
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
        if (~keep).any():
            break
    if not chunks:
        raise ValueError(f'no premium rows before {cutoff}')
    return pd.concat(chunks, ignore_index=True)


def _rolling_z(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window, min_periods=max(24, window // 2)).mean()
    s = x.rolling(window, min_periods=max(24, window // 2)).std(ddof=0).replace(0.0, np.nan)
    return ((x - m) / s).replace([np.inf, -np.inf], np.nan)


def _fit_linear_residual(y: np.ndarray, x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = mask & np.isfinite(y) & np.isfinite(x).all(axis=1)
    if int(finite.sum()) < 20_000:
        raise RuntimeError(f'insufficient fit rows for residual: {int(finite.sum())}')
    design = np.column_stack([np.ones(int(finite.sum())), x[finite]])
    beta = np.linalg.lstsq(design, y[finite], rcond=None)[0]
    full_design = np.column_stack([np.ones(len(y)), x])
    pred = full_design @ beta
    resid = y - pred
    resid[~np.isfinite(pred)] = np.nan
    return beta.astype(float), resid.astype(float)


def _load_pre2024(cfg: Config) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    market_raw = _read_before(cfg.market_csv, 'date', SELECTION_CUTOFF)
    metrics_raw = _read_before(cfg.metrics_csv, 'create_time', SELECTION_CUTOFF)
    funding_raw = _read_before(cfg.funding_csv, 'date', SELECTION_CUTOFF)
    premium_raw = _read_premium_before(cfg.premium_csv, SELECTION_CUTOFF)
    boundary = pd.Timestamp(SELECTION_CUTOFF)
    source_max = {
        'market': str(pd.to_datetime(market_raw['date'], utc=True).dt.tz_convert(None).max()),
        'metrics_create_time': str(pd.to_datetime(metrics_raw['create_time'], utc=True).dt.tz_convert(None).max()),
        'funding': str(pd.to_datetime(funding_raw['date'], utc=True).dt.tz_convert(None).max()),
        'premium_close_time': str(pd.to_datetime(pd.to_numeric(premium_raw['close_time']), unit='ms', utc=True).dt.tz_convert(None).max()) if 'close_time' in premium_raw else str(pd.to_datetime(premium_raw['date'], utc=True).dt.tz_convert(None).max()),
    }
    market = market_raw.copy()
    market['date'] = pd.to_datetime(market['date'], utc=True, errors='raise').dt.tz_convert(None)
    market = market.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    if market['date'].max() >= boundary:
        raise RuntimeError('market was not physically truncated before 2024')
    market = attach_binance_um_aux_frames(
        market,
        funding_frame=normalise_funding_history_frame(funding_raw),
        premium_frame=normalise_premium_index_frame(premium_raw),
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=cfg.premium_tolerance,
    )
    market = _attach_delayed_metrics(
        market,
        metrics_raw,
        tolerance=cfg.metrics_tolerance,
        delay_bars=cfg.metrics_delay_bars,
    )
    if pd.to_datetime(market.loc[market['positioning_source_time'].notna(), 'positioning_source_time']).max() >= boundary:
        raise RuntimeError('delayed metrics source breached pre-2024 cutoff')
    dates = pd.to_datetime(market['date'])
    audit = {
        'physical_cutoff': SELECTION_CUTOFF,
        'source_max_timestamp_before_cutoff': source_max,
        'raw_prefix_hashes': {
            'market': _frame_hash(market_raw),
            'metrics': _frame_hash(metrics_raw),
            'funding': _frame_hash(funding_raw),
            'premium': _frame_hash(premium_raw),
        },
        'file_hashes_full_files_recorded_not_used_for_fit': {p: _sha256(p) for p in [cfg.market_csv, cfg.metrics_csv, cfg.funding_csv, cfg.premium_csv]},
        'causal_joins': {
            'metrics': f'backward asof, then shifted by {cfg.metrics_delay_bars} complete 5m source bar(s)',
            'funding': f'backward asof tolerance {cfg.funding_tolerance}',
            'premium': f'premium kline close_time availability, backward asof tolerance {cfg.premium_tolerance}',
            'execution': 'signal at completed bar t enters at next bar open t+1',
        },
    }
    return market, dates, audit


def _load_full(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Open full sources only after a frozen manifest has been written."""
    market = pd.read_csv(cfg.market_csv, compression='infer')
    metrics = pd.read_csv(cfg.metrics_csv, compression='infer')
    funding = pd.read_csv(cfg.funding_csv, compression='infer')
    premium = pd.read_csv(cfg.premium_csv, compression='infer')
    market['date'] = pd.to_datetime(market['date'], utc=True, errors='raise').dt.tz_convert(None)
    market = market.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    market = attach_binance_um_aux_frames(
        market,
        funding_frame=normalise_funding_history_frame(funding),
        premium_frame=normalise_premium_index_frame(premium),
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=cfg.premium_tolerance,
    )
    market = _attach_delayed_metrics(
        market,
        metrics,
        tolerance=cfg.metrics_tolerance,
        delay_bars=cfg.metrics_delay_bars,
    )
    return market, pd.to_datetime(market['date'])


def _build_features(market: pd.DataFrame, dates: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    close = pd.to_numeric(market['close'], errors='coerce')
    log_close = np.log(close.where(close > 0.0))
    oi = np.log(pd.to_numeric(market['sum_open_interest'], errors='coerce').where(lambda v: v > 0.0))
    funding = pd.to_numeric(market.get('funding_rate'), errors='coerce')
    premium = pd.to_numeric(market.get('premium_index'), errors='coerce')
    premium_change = pd.to_numeric(market.get('premium_index_change'), errors='coerce')
    avail = (
        (pd.to_numeric(market.get('positioning_available', 0.0), errors='coerce').fillna(0.0) > 0.5)
        & (pd.to_numeric(market.get('funding_available', 0.0), errors='coerce').fillna(0.0) > 0.5)
        & (pd.to_numeric(market.get('premium_available', 0.0), errors='coerce').fillna(0.0) > 0.5)
    )
    fit_mask = ((dates >= pd.Timestamp(WINDOWS['fit_through_2022'][0])) & (dates < pd.Timestamp(WINDOWS['fit_through_2022'][1]))).to_numpy(bool)
    cols: dict[str, pd.Series] = {}
    meta: dict[str, Any] = {}
    for w in (48, 144, 288):
        oi_chg = oi - oi.shift(w)
        price_ret = log_close - log_close.shift(w)
        # Carry pressure is the point-in-time proxy for which side owns the new inventory.
        carry = _rolling_z(funding, 2016).fillna(0.0) + _rolling_z(premium, 2016).fillna(0.0) + 0.5 * _rolling_z(premium_change, 2016).fillna(0.0)
        x_df = pd.DataFrame({
            'abs_price_ret': price_ret.abs(),
            'price_ret': price_ret,
            'funding': funding,
            'premium': premium,
            'premium_change': premium_change,
            'abs_carry': carry.abs(),
        })
        beta, resid = _fit_linear_residual(oi_chg.to_numpy(float), x_df.to_numpy(float), fit_mask & avail.to_numpy(bool))
        resid_s = pd.Series(resid, index=market.index)
        cols[f'resid_z{w}'] = _rolling_z(resid_s, 2016).where(avail)
        cols[f'oi_chg_z{w}'] = _rolling_z(oi_chg, 2016).where(avail)
        cols[f'price_ret_z{w}'] = _rolling_z(price_ret, 2016).where(avail)
        meta[f'w{w}'] = {'beta_columns': ['intercept', *x_df.columns.tolist()], 'beta': beta.tolist()}
    cols['carry_z'] = (_rolling_z(funding, 2016).fillna(0.0) + _rolling_z(premium, 2016).fillna(0.0) + 0.5 * _rolling_z(premium_change, 2016).fillna(0.0)).where(avail)
    frame = pd.DataFrame(cols, index=market.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return frame, meta


def _mask_window(dates: pd.Series, name: str) -> np.ndarray:
    a, b = WINDOWS[name]
    return ((dates >= pd.Timestamp(a)) & (dates < pd.Timestamp(b))).to_numpy(bool)


def _stats_for(market: pd.DataFrame, dates: pd.Series, long_active: np.ndarray, short_active: np.ndarray, hold: int, cfg: Config, windows: dict[str, tuple[str, str]] = WINDOWS) -> dict[str, dict[str, Any]]:
    extremes = (_future_extreme(market['low'].to_numpy(float), hold, 'min'), _future_extreme(market['high'].to_numpy(float), hold, 'max'))
    return {
        name: _simulate_no_stop(
            market,
            dates,
            long_active,
            short_active,
            window=name,
            hold_bars=hold,
            stride_bars=cfg.stride_bars,
            leverage=cfg.leverage,
            fee_rate=cfg.fee_rate,
            slippage_rate=cfg.slippage_rate,
            extremes=extremes,
            windows=windows,
        )
        for name in windows
    }


def _score(stats: dict[str, dict[str, Any]]) -> float:
    fit, sel, h1, h2 = stats['fit_through_2022'], stats['select_2023'], stats['select_2023_h1'], stats['select_2023_h2']
    if fit['trades'] < 60 or sel['trades'] < 20 or h1['trades'] < 6 or h2['trades'] < 6:
        return -1e12
    if min(fit['cagr_pct'], sel['cagr_pct'], h1['cagr_pct'], h2['cagr_pct']) <= 0.0:
        return -1e12
    if fit['strict_mdd_pct'] > 35.0 or sel['strict_mdd_pct'] > 25.0:
        return -1e12
    ratios = np.asarray([fit['ratio'], sel['ratio'], h1['ratio'], h2['ratio']], dtype=float)
    return float(np.min(ratios) + 0.25 * np.median(ratios) + min(sel['trades'] / 200.0, 0.25))


def _spec_masks(features: pd.DataFrame, spec: dict[str, Any], *, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    resid = features[spec['feature']].to_numpy(float)
    carry = features['carry_z'].to_numpy(float)
    finite = np.isfinite(resid) & np.isfinite(carry)
    if spec['control'] == 'carry_only':
        active = finite & (np.abs(carry) >= spec['carry_abs'])
    else:
        active = finite & (resid >= spec['resid_threshold']) & (np.abs(carry) >= spec['carry_abs'])
    long_active = active & (carry <= -spec['carry_abs'])
    short_active = active & (carry >= spec['carry_abs'])
    if spec['control'] == 'price_direction':
        price_z = features[spec['price_feature']].to_numpy(float)
        long_active = active & (price_z <= -0.25)
        short_active = active & (price_z >= 0.25)
    if flip:
        return short_active.copy(), long_active.copy()
    return long_active, short_active


def _policy_feature_columns(policy: dict[str, Any]) -> list[str]:
    window = int(policy['window_bars'])
    return [policy['feature'], 'carry_z', f'oi_chg_z{window}', f'price_ret_z{window}']


def _fit_feature_threshold(
    features: pd.DataFrame,
    dates: pd.Series,
    feature: str,
    quantile: float,
) -> float:
    fit_mask = (
        (dates >= pd.Timestamp(WINDOWS['fit_through_2022'][0]))
        & (dates < pd.Timestamp(WINDOWS['fit_through_2022'][1]))
    ).to_numpy(bool)
    values = features[feature].to_numpy(float)
    return float(np.quantile(values[fit_mask & np.isfinite(values)], quantile))


def _freeze_manifest(
    cfg: Config,
    market: pd.DataFrame,
    dates: pd.Series,
    features: pd.DataFrame,
    residual_meta: dict[str, Any],
    selected: dict[str, Any],
    load_audit: dict[str, Any],
) -> dict[str, Any]:
    policy = selected['spec']
    long_active, short_active = _spec_masks(features, policy)
    feature_columns = _policy_feature_columns(policy)
    raw_oi_feature = f"oi_chg_z{policy['window_bars']}"
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'phase': 'frozen_before_future_open',
        'selection_cutoff': SELECTION_CUTOFF,
        'policy': policy,
        'raw_oi_control_threshold': _fit_feature_threshold(
            features,
            dates,
            raw_oi_feature,
            float(policy['fit_quantile']),
        ),
        'residual_fit': residual_meta,
        'pre2024_admission': selected['stats'],
        'pre2024_quarters': selected['segment_stats_2023_quarters'],
        'market_rows': len(market),
        'market_prefix_hash': _frame_hash(market[['date', 'open', 'high', 'low', 'close']]),
        'feature_prefix_hash': _frame_hash(
            pd.concat([dates.rename('date'), features[feature_columns]], axis=1)
        ),
        'signal_prefix_hash': _signal_hash(long_active, short_active),
        'source_audit': load_audit,
        'future_windows_unopened': list(OOS_WINDOWS),
    }
    output = Path(cfg.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, allow_nan=True) + '\n')
    return manifest


def _replay_frozen(cfg: Config, manifest: dict[str, Any]) -> dict[str, Any]:
    """Replay one immutable policy after verifying all pre-2024 prefixes."""
    frozen = json.loads(Path(cfg.manifest_output).read_text())
    if frozen != manifest:
        raise RuntimeError('in-memory and on-disk frozen manifests differ')
    market, dates = _load_full(cfg)
    features, residual_meta = _build_features(market, dates)
    if residual_meta != frozen['residual_fit']:
        raise RuntimeError('fit coefficients changed after OOS rows opened')
    n = int(frozen['market_rows'])
    if _frame_hash(market.iloc[:n][['date', 'open', 'high', 'low', 'close']]) != frozen['market_prefix_hash']:
        raise RuntimeError('market prefix changed after OOS rows opened')
    feature_columns = _policy_feature_columns(frozen['policy'])
    prefix_features = pd.concat(
        [
            dates.iloc[:n].rename('date').reset_index(drop=True),
            features.iloc[:n][feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    if _frame_hash(prefix_features) != frozen['feature_prefix_hash']:
        raise RuntimeError('feature prefix changed after OOS rows opened')
    long_active, short_active = _spec_masks(features, frozen['policy'])
    if _signal_hash(long_active[:n], short_active[:n]) != frozen['signal_prefix_hash']:
        raise RuntimeError('signal prefix changed after OOS rows opened')

    hold = int(frozen['policy']['hold_bars'])
    primary = _stats_for(market, dates, long_active, short_active, hold, cfg, OOS_WINDOWS)
    stress_cfg = Config(**(asdict(cfg) | {'fee_rate': 0.0008, 'slippage_rate': 0.0002}))
    stress = _stats_for(market, dates, long_active, short_active, hold, stress_cfg, OOS_WINDOWS)
    direction_flip = _stats_for(market, dates, short_active, long_active, hold, cfg, OOS_WINDOWS)

    carry_policy = dict(frozen['policy'], control='carry_only')
    carry_long, carry_short = _spec_masks(features, carry_policy)
    carry_only = _stats_for(market, dates, carry_long, carry_short, hold, cfg, OOS_WINDOWS)

    raw_oi_feature = f"oi_chg_z{frozen['policy']['window_bars']}"
    raw_oi_policy = dict(
        frozen['policy'],
        control='oi_chg_simplified',
        feature=raw_oi_feature,
        resid_threshold=float(frozen['raw_oi_control_threshold']),
    )
    raw_oi_long, raw_oi_short = _spec_masks(features, raw_oi_policy)
    raw_oi = _stats_for(market, dates, raw_oi_long, raw_oi_short, hold, cfg, OOS_WINDOWS)

    passes = bool(
        primary['test_2024']['ratio'] >= 3.0
        and primary['eval_2025']['ratio'] >= 3.0
        and primary['holdout_2026']['return_pct'] > 0.0
        and primary['combined_oos']['ratio'] >= 3.0
    )
    return {
        'prefix_verified': True,
        'decision': 'candidate_oos' if passes else 'reject_oos',
        'primary_6bp': primary,
        'stress_10bp': stress,
        'direction_flip_6bp': direction_flip,
        'carry_only_6bp': carry_only,
        'raw_oi_change_6bp': raw_oi,
    }


def run() -> dict[str, Any]:
    cfg = Config()
    market, dates, load_audit = _load_pre2024(cfg)
    features, residual_meta = _build_features(market, dates)
    fit_mask = _mask_window(dates, 'fit_through_2022')

    candidates: list[dict[str, Any]] = []
    for w in (48, 144, 288):
        for feature in (f'resid_z{w}',):
            fit_values = features.loc[fit_mask, feature].to_numpy(float)
            fit_values = fit_values[np.isfinite(fit_values)]
            for q in (0.80, 0.90, 0.95):
                thr = float(np.quantile(fit_values, q))
                for carry_abs in (0.5, 1.0, 1.5):
                    for hold in (144, 288, 576):
                        for control in ('residual_carry_fade', 'flip_direction', 'oi_chg_simplified', 'carry_only', 'price_direction'):
                            spec_feature = feature if control not in {'oi_chg_simplified'} else f'oi_chg_z{w}'
                            spec = {
                                'control': control,
                                'window_bars': w,
                                'feature': spec_feature,
                                'price_feature': f'price_ret_z{w}',
                                'fit_quantile': q,
                                'resid_threshold': thr if control != 'oi_chg_simplified' else float(np.quantile(features.loc[fit_mask, spec_feature].dropna().to_numpy(float), q)),
                                'carry_abs': carry_abs,
                                'hold_bars': hold,
                                'stride_bars': cfg.stride_bars,
                            }
                            long_active, short_active = _spec_masks(features, spec, flip=(control == 'flip_direction'))
                            stats = _stats_for(market, dates, long_active, short_active, hold, cfg)
                            score = _score(stats)
                            candidates.append({'spec': spec, 'selection_score': score, 'stats': stats})

    ranked = sorted(candidates, key=lambda r: (r['selection_score'], r['stats']['select_2023']['ratio'], r['stats']['select_2023']['return_pct']), reverse=True)
    top = ranked[:10]
    for row in top:
        spec = row['spec']
        la, sa = _spec_masks(features, spec, flip=(spec['control'] == 'flip_direction'))
        row['segment_stats_2023_quarters'] = _stats_for(market, dates, la, sa, spec['hold_bars'], cfg, SEGMENT_WINDOWS)
        qs = row['segment_stats_2023_quarters']
        row['segment_stability'] = {
            'positive_quarters': int(sum(v['return_pct'] > 0 for v in qs.values())),
            'negative_quarters': int(sum(v['return_pct'] < 0 for v in qs.values())),
            'active_quarters': int(sum(v['trades'] > 0 for v in qs.values())),
            'min_quarter_ratio': float(min((v['ratio'] for v in qs.values() if v['trades'] > 0), default=0.0)),
        }

    def ensure_segments(row: dict[str, Any]) -> None:
        if 'segment_stability' in row:
            return
        spec = row['spec']
        la, sa = _spec_masks(features, spec, flip=(spec['control'] == 'flip_direction'))
        row['segment_stats_2023_quarters'] = _stats_for(market, dates, la, sa, spec['hold_bars'], cfg, SEGMENT_WINDOWS)
        qs = row['segment_stats_2023_quarters']
        row['segment_stability'] = {
            'positive_quarters': int(sum(v['return_pct'] > 0 for v in qs.values())),
            'negative_quarters': int(sum(v['return_pct'] < 0 for v in qs.values())),
            'active_quarters': int(sum(v['trades'] > 0 for v in qs.values())),
            'min_quarter_ratio': float(min((v['ratio'] for v in qs.values() if v['trades'] > 0), default=0.0)),
        }

    best_overall = top[0] if top else None
    primary_rows = [r for r in ranked if r['spec']['control'] == 'residual_carry_fade']
    primary_best = primary_rows[0] if primary_rows else None
    if primary_best:
        ensure_segments(primary_best)
    formalize = False
    rationale = 'No eligible positive/stable residual-carry candidate.'
    if primary_best:
        s = primary_best['stats']
        st = primary_best['segment_stability']
        formalize = (
            primary_best['selection_score'] > -1e11
            and s['select_2023']['ratio'] >= 2.0
            and s['select_2023_h1']['ratio'] > 0.75
            and s['select_2023_h2']['ratio'] > 0.75
            and st['positive_quarters'] >= 3
            and st['negative_quarters'] <= 1
        )
        rationale = 'Passes provisional formalization gates.' if formalize else 'Residual prototype is promising directionally but does not clear all stability/formalization gates.'

    report = {
        'as_of': datetime.now(timezone.utc).isoformat(),
        'hypothesis': {
            'name': 'leveraged_inventory_conservation_residual',
            'logic': 'Fit a pre-2023 conservation model for delayed open-interest change from contemporaneously completed price movement, funding, and premium. A large positive residual means leveraged inventory was created beyond what price/carry explain. Funding/premium sign proxies which side likely owns that excess inventory; fade positive-carry excess inventory as long-crowding unwind risk and fade negative-carry excess inventory as short-crowding squeeze risk.',
            'point_in_time_inputs': ['5m OHLCV close completed before signal bar', 'Binance UM open interest delayed by one complete 5m metrics bar', 'funding history by backward asof', 'premium index kline close by close_time backward asof'],
        },
        'config': asdict(cfg),
        'windows': WINDOWS,
        'leakage_audit': load_audit | {
            'fit_rule': 'all coefficients and quantile thresholds fit only on 2020-10-15 <= t < 2023-01-01',
            'selection_rule': 'selection/reporting only on 2023, 2023H1, 2023H2 after physical pre-2024 cutoff',
            'post_2024_rows_loaded_during_selection': False,
            'next_bar_open_execution': True,
            'costs': '0.5x leverage, 5bp fee + 1bp slippage per side = 6bp/side',
            'strict_intratrade_mdd': 'canonical _simulate_no_stop favorable-high-water then adverse OHLC extreme',
        },
        'residual_fit': residual_meta,
        'search_space': {
            'rows': len(candidates),
            'windows': [48, 144, 288],
            'residual_quantiles': [0.80, 0.90, 0.95],
            'carry_abs_thresholds': [0.5, 1.0, 1.5],
            'hold_bars': [144, 288, 576],
            'controls': ['residual_carry_fade', 'flip_direction', 'oi_chg_simplified', 'carry_only', 'price_direction'],
        },
        'top': top,
        'best_overall': best_overall,
        'primary_residual_carry_fade': primary_best,
        'control_summary': {},
        'merits_formalization': bool(formalize),
        'formalization_rationale': rationale,
    }
    for control in ['residual_carry_fade', 'flip_direction', 'oi_chg_simplified', 'carry_only', 'price_direction']:
        rows = [r for r in ranked if r['spec']['control'] == control]
        if rows:
            report['control_summary'][control] = {
                'best_score': rows[0]['selection_score'],
                'best_spec': rows[0]['spec'],
                'best_stats': rows[0]['stats'],
            }
    if formalize and primary_best is not None:
        manifest = _freeze_manifest(
            cfg,
            market,
            dates,
            features,
            residual_meta,
            primary_best,
            load_audit,
        )
        oos = _replay_frozen(cfg, manifest)
        report['manifest'] = manifest
        report['oos'] = oos
        report['decision'] = oos['decision']
    else:
        report['manifest'] = None
        report['oos'] = None
        report['decision'] = 'fail_pre2024_admission'
    output_path = Path(cfg.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=True) + '\n')
    verification = {
        'verified_at': datetime.now(timezone.utc).isoformat(),
        'selection_cutoff': SELECTION_CUTOFF,
        'manifest_sha256': _sha256(cfg.manifest_output) if report['manifest'] else None,
        'report_sha256': _sha256(cfg.output_json),
        'prefix_verified': bool(report['oos'] and report['oos']['prefix_verified']),
        'decision': report['decision'],
        'replay_command': 'PYTHONPATH=. .venv/bin/python -m training.search_inventory_conservation_residual_alpha',
    }
    verification_path = Path(cfg.verification_output)
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_path.write_text(json.dumps(verification, indent=2) + '\n')
    lines = []
    lines.append('Leveraged-inventory conservation residual alpha prototype (strict pre-2024)')
    lines.append(f"Primary residual merits formalization: {formalize} — {rationale}")
    if primary_best:
        lines.append(f"Primary residual spec: {json.dumps(primary_best['spec'], sort_keys=True)}")
        for name in ['fit_through_2022','select_2023','select_2023_h1','select_2023_h2']:
            v = primary_best['stats'][name]
            lines.append(f"primary {name}: abs_ret={v['return_pct']:.2f}% cagr={v['cagr_pct']:.2f}% strict_mdd={v['strict_mdd_pct']:.2f}% ratio={v['ratio']:.2f} trades={v['trades']} longs={v['longs']} shorts={v['shorts']}")
        lines.append(f"Primary segment stability: {primary_best['segment_stability']}")
    if best_overall and (not primary_best or best_overall is not primary_best):
        s2023 = best_overall['stats']['select_2023']
        lines.append(f"Best overall was a control ({best_overall['spec']['control']}): 2023 ratio={s2023['ratio']:.2f} ret={s2023['return_pct']:.2f}% trades={s2023['trades']}")
    lines.append('Controls:')
    for control, row in report['control_summary'].items():
        s = row['best_stats']['select_2023']
        lines.append(f"  {control}: score={row['best_score']:.3f} 2023 ratio={s['ratio']:.2f} ret={s['return_pct']:.2f}% mdd={s['strict_mdd_pct']:.2f}% trades={s['trades']} spec={row['best_spec']}")
    lines.append('Leakage audit: physical cutoff before 2024; delayed OI; close_time premium; next-bar open; frozen prefix verified before OOS scoring.')
    if report['oos']:
        for name in ['test_2024', 'eval_2025', 'holdout_2026', 'combined_oos']:
            value = report['oos']['primary_6bp'][name]
            lines.append(
                f"oos {name}: abs_ret={value['return_pct']:.2f}% cagr={value['cagr_pct']:.2f}% "
                f"strict_mdd={value['strict_mdd_pct']:.2f}% ratio={value['ratio']:.2f} trades={value['trades']}"
            )
    lines.append(f"Decision: {report['decision']}")
    print('\n'.join(lines))
    return report


if __name__ == '__main__':
    run()
