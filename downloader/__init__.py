from downloader.binancedownloader import BinanceDownloader
from utils import generate_cache_key
from config import DATA_SAVE_DIR
import os

def download(start_date, end_date, ticker_list, time_interval='1d', **kwargs):
    dp = BinanceDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=ticker_list,
        market_type=kwargs.get("market_type", 'futures')
    )

    cache_key = generate_cache_key(
        ticker_list, 
        time_interval, 
        start_date=start_date, 
        end_date=end_date, 
        market_type=kwargs.get("market_type", 'futures')
    )
    import pandas as pd

    cached_file_dir = os.path.join(DATA_SAVE_DIR, f"{pd.to_datetime(start_date).strftime('%Y-%m-%d')}_{pd.to_datetime(end_date).strftime('%Y-%m-%d')}_{cache_key}.csv.gz")
    if os.path.exists(cached_file_dir):
        print(f"Loading FULLY processed cached data for {cache_key} from {DATA_SAVE_DIR}")
        import pandas as pd
        data = pd.read_csv(
            cached_file_dir,
            parse_dates=["date"], 
            compression='gzip',
            index_col=0
        )
        
    else:
        data = dp.fetch_data(time_interval)
        data.to_csv(os.path.join(DATA_SAVE_DIR, f"{pd.to_datetime(start_date).strftime('%Y-%m-%d')}_{pd.to_datetime(end_date).strftime('%Y-%m-%d')}_{cache_key}.csv.gz"), compression='gzip')
        
    return data