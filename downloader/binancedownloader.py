"""Contains methods and classes to collect data from
Binance API
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime
import time
from binance.client import Client
from typing import List, Optional


class BinanceDownloader:
    """Provides methods for retrieving daily stock data from
    Binance API

    Attributes
    ----------
        start_date : str
            start date of the data (format: 'YYYY-MM-DD')
        end_date : str
            end date of the data (format: 'YYYY-MM-DD')
        ticker_list : list
            a list of crypto tickers (e.g., ["BTCUSDT", "ETHUSDT"])
        market_type : str
            type of market to fetch data from ('spot' or 'futures')

    Methods
    -------
    fetch_data()
        Fetches data from Binance API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list, market_type='spot'):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = [tic.upper() for tic in ticker_list]

        self.market_type = market_type
        try:
            self.client = Client()  # Initialize without API keys for public endpoints only
        except:
            pass

    def fetch_data(self, interval="1d", proxy=None, auto_adjust=False) -> pd.DataFrame:
        """Fetches data from Binance API
        Parameters
        ----------
        interval : str, optional
            Frequency of data (default: '1d' for daily)
        proxy : str, optional
            URL for proxy server
        auto_adjust : bool, optional
            Adjust prices (not applicable for Binance, kept for compatibility)

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified crypto tickers
        """
        # Convert dates to millisecond timestamps for Binance API
        start_ts = int(datetime.strptime(self.start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(self.end_date, '%Y-%m-%d').timestamp() * 1000)

        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            klines = []
            # try:
            # Get historical klines from Binance based on market type
            current_ts = start_ts
            while pd.to_datetime(current_ts, unit='ms').ceil(freq='1s') < pd.to_datetime(end_ts, unit='ms'):
                if self.market_type == 'futures':
                    _klines = self.client.futures_klines(
                        symbol=tic,
                        interval=interval,
                        startTime=current_ts,
                        endTime=end_ts
                    )
                else:  # default to 'spot'
                    _klines = self.client.get_historical_klines(
                        symbol=tic,
                        interval=interval,
                        start_str=current_ts,
                        end_str=end_ts
                    )
                time.sleep(0.1)
                _klines = pd.DataFrame(_klines)
                _klines = _klines.convert_dtypes(['int', 'float', 'float', 'float', 'float', 
                                        'float', 'int', 'float', 
                                        'int', 'float', 'float', 'int'])

                current_ts = int(_klines.iloc[:, 6].max())
                klines.append(_klines)
                
            klines = pd.concat(klines)
            if len(klines) > 0:
                # Create DataFrame with standard column names
                temp_df = pd.DataFrame(klines.values, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 
                    'volume', 'close_time', 'quote_asset_volume', 
                    'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'
                ])
                
                # Convert timestamp to date and select required columns
                temp_df['date'] = pd.to_datetime(temp_df['timestamp'], unit='ms')
                temp_df = temp_df.drop(columns=['ignored', 'close_time', 'timestamp'])
                
                # Convert string values to float
                for col in temp_df.columns.drop(['date']):
                    temp_df[col] = temp_df[col].astype(float)
                
                # Add ticker column
                temp_df['tic'] = tic
                
                # Append to main dataframe
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
                
                
            # except Exception as e:
            #     print(f"Error downloading {tic}: {str(e)}")
            #     num_failures += 1
                
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
            
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index(drop=True)
        
        # create day of the week column (monday = 0)
        data_df['day'] = data_df['date'].dt.dayofweek
        
        # # convert date to standard string format, easy to filter
        # data_df['date'] = data_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        
        print("Shape of DataFrame: ", data_df.shape)
        
        # Sort by date and ticker
        data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        data_df = data_df[['date'] + data_df.columns.drop('date').tolist()]  # 
        return data_df

    def _adjust_prices(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder method to maintain compatibility with YahooDownloader interface.
        No adjustment needed for Binance data as it doesn't have adjusted close prices.
        """
        return data_df

    def select_equal_rows_stock(self, df):
        """Select stocks that have certain number of rows.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe with crypto data
            
        Returns
        -------
        pd.DataFrame
            The dataframe with selected cryptos that have enough data
        """
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
    
    
    
if __name__ == "__main__":
    # Example usage
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    ticker_list = ['BTCUSDT', 'ETHUSDT']
    
    downloader = BinanceDownloader(start_date, end_date, ticker_list)
    data_df = downloader.fetch_data(interval='1d')
    
    print(data_df.head())
    print(data_df.info())