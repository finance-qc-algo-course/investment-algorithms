import numpy as np
import pandas as pd
import datetime as dttm
from typing import Callable, List

import io
from urllib.request import urlopen


class BaseDataProvider:
    def __init__(self):
        self.tickers = []
        self.start_date = dttm.date(1970, 1, 1)
        self.end_date = dttm.date(2037, 12, 31)

        self.history = None
        self.returns = None

    def GetHistory(self, tickers: List[str],
                   start_date: dttm.date, end_date: dttm.date) -> pd.DataFrame:
        assert set(tickers).issubset(self.tickers)
        assert start_date < end_date
        assert start_date >= self.start_date and end_date <= self.end_date

        tickers_idx = self.history.index.get_level_values(0)
        date_idx = self.history.index.get_level_values(1)
        return self.history[(tickers_idx.isin(tickers)) &
                            (date_idx >= str(start_date)) & (date_idx < str(end_date))]

    def GetReturns(self, tickers: List[str],
                   start_date: dttm.date, end_date: dttm.date) -> pd.DataFrame:
        assert set(tickers).issubset(self.tickers)
        assert start_date < end_date
        assert start_date >= self.start_date and end_date <= self.end_date

        idx = self.returns.index
        return self.returns[(idx >= str(start_date)) & (idx < str(end_date))][tickers]


class YahooDataProvider(BaseDataProvider):
    RAW_HISTORY_URL_PATTERN = \
        "https://query1.finance.yahoo.com/v7/finance/download/" + \
        "{}?period1={}&period2={}&interval=1d&events=history"

    def DownloadHistory(tickers: List[str],
                        start_date: dttm.date, end_date: dttm.date) -> List[pd.DataFrame]:
        start_period = int(
            (start_date - dttm.date(1970, 1, 1)).total_seconds())
        end_period = int((end_date - dttm.date(1970, 1, 1)).total_seconds())
        raw_history = []
        for ticker in tickers:
            url = YahooDataProvider.RAW_HISTORY_URL_PATTERN \
                .format(ticker.upper(), start_period, end_period)
            # XXX:
            print("Loading URL: {}".format(url))
            output = urlopen(url).read()
            str_io = io.StringIO(output.decode('utf-8'))
            raw_history.append(pd.read_csv(str_io, parse_dates=['Date'])
                               .fillna(method='ffill'))

        return raw_history

    def HistoryYahoo2QC(tickers: List[str],
                        raw_history: List[pd.DataFrame]) -> pd.DataFrame:
        rename_dict = {
            'Date': 'time', 'Open': 'open', 'Close': 'close',
            'High': 'high', 'Low': 'low', 'Volume': 'volume'
        }
        raw_history = raw_history.copy()
        for df, ticker in zip(raw_history, tickers):
            df['symbol'] = ticker
            df.drop(columns=['Adj Close'], inplace=True)
            df.rename(columns=rename_dict, inplace=True)
            df.set_index(['symbol', 'time'], inplace=True)

        history = pd.concat(raw_history)
        return history

    def CalculateReturns(tickers: List[str],
                         history: pd.DataFrame) -> pd.DataFrame:
        upsampled_history = [
            history.loc[t].resample(rule='d').ffill() for t in tickers
        ]
        raw_returns = [
            (((df['close'] - df.shift(1)['close']) / df['close'])
             .fillna(0.0)).rename(t)
            for t, df in zip(tickers, upsampled_history)
        ]

        returns = pd.concat(raw_returns, axis=1)
        return returns[returns.index.dayofweek < 5]

    def __init__(self, tickers: List[str],
                 start_date: dttm.date, end_date: dttm.date):
        super().__init__()
        assert len(tickers) > 0
        assert start_date < end_date

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        raw_history = YahooDataProvider.DownloadHistory(
            self.tickers, self.start_date, self.end_date)
        self.history = YahooDataProvider.HistoryYahoo2QC(
            self.tickers, raw_history)
        self.returns = YahooDataProvider.CalculateReturns(
            self.tickers, self.history)
