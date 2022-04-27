import numpy as np
import pandas as pd
import scipy.stats as sps
import datetime as dttm
from typing import Callable, List

from .LocalDataProvider import BaseDataProvider

class MockDataProvider(BaseDataProvider):

    def GenerateHistory(tickers: List[str], returns: np.ndarray, \
            start_date: dttm.date, end_date: dttm.date) -> pd.DataFrame:
        raw_history = []
        dates = pd.bdate_range(start_date, end_date)
        volumes = np.full(len(dates), 100000)
        for r, ticker in zip(returns, tickers):
            ratios = np.exp(sps.norm.rvs( \
                    loc=np.log(1 + r), scale=np.log(1 + 10 * r), size=len(dates) + 1))
            values = np.cumprod(ratios)
            low = values[:-1]
            high = values[1:]
            data = {
                'symbol': ticker, 'time': dates,
                'open': low, 'close': high,
                'high': high, 'low': low,
                'volume': volumes,
            }
            df = pd.DataFrame(data=data)
            df.set_index(['symbol', 'time'], inplace=True)
            raw_history.append(df)

        history = pd.concat(raw_history)
        return history

    def CalculateReturns(tickers: List[str], \
            history: pd.DataFrame) -> pd.DataFrame:
        upsampled_history = [ \
            history.loc[t].resample(rule='d').ffill() for t in tickers \
        ]
        raw_returns = [ \
            (((df['close'] - df.shift(1)['close']) / df['close']) \
            .fillna(0.0)).rename(t) \
            for t, df in zip(tickers, upsampled_history) \
        ]

        returns = pd.concat(raw_returns, axis=1)
        return returns[returns.index.dayofweek < 5]
    
    def __init__(self, tickers: List[str], returns: np.ndarray, \
            start_date: dttm.date, end_date: dttm.date):
        super().__init__()
        assert len(tickers) > 0
        assert start_date < end_date

        self.tickers = tickers
        self.returns = returns
        self.start_date = start_date
        self.end_date = end_date

        self.history = MockDataProvider.GenerateHistory( \
                self.tickers, self.returns, self.start_date, self.end_date)
        self.returns = MockDataProvider.CalculateReturns( \
                self.tickers, self.history)

