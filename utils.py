import hashlib, json 
import numpy as np
import sys
import os

dtype = np.float128 if sys.platform == 'linux' else np.float64

def generate_cache_key(ticker_list, time_interval, **kwargs):
    """Generate a unique cache key based on the download parameters."""
    # Sort ticker_list to ensure consistent cache keys
    sorted_tickers = sorted(ticker_list)
    key_dict = {
        "tickers": sorted_tickers,
        "time_interval": time_interval,
        **kwargs  # Include any additional parameters
    }
    # Convert to a deterministic string representation
    key_str = json.dumps(key_dict, sort_keys=True)
    # Create a hash of the string
    return hashlib.md5(key_str.encode()).hexdigest()

def get_feature_set_hash(feature_columns: list) -> str:
    """Generates a unique hash for a sorted list of feature columns."""
    # Sort the list to ensure consistency
    sorted_features = sorted(feature_columns)
    # Serialize the list to a string
    feature_str = ",".join(sorted_features)
    # Create a hash and return the first 8 characters
    return hashlib.md5(feature_str.encode()).hexdigest()[:8]

def update_feature_log(feature_hash: str, feature_columns: list, log_file: str = 'feature_sets.json'):
    """Logs the feature set against its hash in a JSON file."""
    log_data = {}
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                pass  # Handle empty or corrupted file
    
    if feature_hash not in log_data:
        log_data[feature_hash] = sorted(feature_columns)
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=4)

candle_in_a_day = {
    '1h': 24,
    '1d': 1,
    '1m': 60 * 24,
    '3m': 60 * 24 // 3,
    '5m': 60 * 24 // 5,
    '15m': 60 * 24 // 15,
}

def sharpe_ratio(history, risk_free_rate=0.00, trading_days=365):
    # 포트폴리오 가치에서 일별 수익률 계산
    portfolio_values = np.array(history["portfolio_valuation"], dtype=dtype)
    if np.any(portfolio_values[:-1] == 0):
        return '0.0000'  
    returns = np.diff(portfolio_values) / portfolio_values[:-1] 
    
    # 초과 수익률(무위험이자율 차감)
    excess_returns = returns - risk_free_rate / trading_days
    num_candles_in_a_day = candle_in_a_day[os.environ['CANDLE']]
    # Sharpe 지수 연환산
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    sharpe = (mean_excess_return / std_excess_return) * np.sqrt(trading_days * num_candles_in_a_day) if std_excess_return != 0 else 0
    return f'{sharpe:.4f}'

def max_drawdown(history):
    portfolio_values = np.array(history["portfolio_valuation"], dtype=dtype)
    # 누적 최대값 계산
    cumulative_max = np.maximum.accumulate(portfolio_values)
    # 각 시점의 드로우다운 비율 계산
    drawdowns = (portfolio_values - cumulative_max) / cumulative_max
    # MDD는 드로우다운의 최소값(음수)이므로 절댓값으로 반환
    mdd = np.min(drawdowns)
    return f'{-mdd * 100:.4f}'


def log_returns(series):
    """Compute log returns from a positive-valued series."""
    values = np.asarray(series, dtype=dtype)
    if len(values) < 2:
        return np.asarray([], dtype=dtype)
    if np.any(values <= 0):
        raise ValueError("log_returns requires strictly positive values")
    return np.log(values[1:] / values[:-1])


def range_volatility_pct(highs, lows):
    """
    Compute `(max_high - min_low) / midpoint` where midpoint is
    `(max_high + min_low)/2`.
    """
    highs_arr = np.asarray(highs, dtype=dtype)
    lows_arr = np.asarray(lows, dtype=dtype)
    if len(highs_arr) == 0 or len(lows_arr) == 0:
        raise ValueError("highs/lows must be non-empty")
    max_high = np.max(highs_arr)
    min_low = np.min(lows_arr)
    midpoint = (max_high + min_low) / dtype(2.0)
    if midpoint == 0:
        return 0.0
    return float((max_high - min_low) / midpoint)


def sharpe_ratio_log(log_rets, periods_per_year, risk_free_rate=0.0):
    """Annualized Sharpe ratio for log-return series."""
    arr = np.asarray(log_rets, dtype=dtype)
    if len(arr) == 0:
        return 0.0
    rf_per_period = dtype(risk_free_rate) / dtype(periods_per_year)
    excess = arr - rf_per_period
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float((np.mean(excess) / std) * np.sqrt(dtype(periods_per_year)))


def min_sharpe(equity, underlying, rf=0.0, periods_per_year=365):
    """
    Compute `min(Sharpe_cash, Sharpe_underlying)`.

    - Sharpe_cash: excess log-return Sharpe against risk-free rate.
    - Sharpe_underlying: relative Sharpe against underlying benchmark.
    """
    strategy_log = log_returns(equity)
    benchmark_log = log_returns(underlying)

    n = min(len(strategy_log), len(benchmark_log))
    if n == 0:
        return 0.0
    strategy_log = strategy_log[-n:]
    benchmark_log = benchmark_log[-n:]

    sharpe_cash = sharpe_ratio_log(
        strategy_log, periods_per_year=periods_per_year, risk_free_rate=rf
    )

    std_strategy = np.std(strategy_log)
    if std_strategy == 0:
        sharpe_underlying = 0.0
    else:
        sharpe_underlying = float(
            ((np.mean(strategy_log) - np.mean(benchmark_log)) / std_strategy)
            * np.sqrt(dtype(periods_per_year))
        )

    return float(min(sharpe_cash, sharpe_underlying))
