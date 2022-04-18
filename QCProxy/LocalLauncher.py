import abc
import numpy as np
import pandas as pd
import datetime as dttm
from typing import Callable, List

import io
from urllib.request import urlopen

class BaseScore:
    TRADE_DAYS = 252
    def __init__(self):
        pass

    @abc.abstractmethod
    def Zero(self):
        pass
    
    @abc.abstractmethod
    def Update(self, delta_returns: np.ndarray):
        pass

    @abc.abstractmethod
    def Eval(self) -> np.float64:
        pass

# See
# https://github.com/QuantConnect/Lean/blob/master/Common/Statistics/Statistics.cs#L462-L477
# https://github.com/QuantConnect/Lean/blob/master/Common/Statistics/Statistics.cs#L564-L567
class SharpeRatioScore(BaseScore):
    def __init__(self, risk_free: float = 0.0):
        super().__init__()
        self.risk_free = risk_free
        self.returns = np.array([], dtype=np.float64)

    def Zero(self):
        self.returns = np.array([], dtype=np.float64)
    
    def Update(self, delta_returns: np.ndarray):
        self.returns = np.append(self.returns, delta_returns)

    def Eval(self) -> np.float64:
        mean = np.power(np.mean(self.returns) + 1.0, BaseScore.TRADE_DAYS) - 1.0
        var = np.var(self.returns) * BaseScore.TRADE_DAYS
        return np.divide(mean - self.risk_free, var)

# TODO:
"""
class SortinoRatioScore(BaseScore):
    def __init__(self, risk_free: float):
        super().__init__()
        self.risk_free = risk_free
        self.returns = np.array([], dtype=np.float64)

    def Zero(self):
        self.returns = np.array([], dtype=np.float64)
    
    def Update(self, delta_returns: np.ndarray):
        self.returns = np.append(self.returns, delta_returns)

    def Eval(self) -> np.float64:
        mean = np.power(np.mean(self.returns) + 1.0, BaseScore.TRADE_DAYS) - 1.0
        std = np.std(self.returns) * BaseScore.TRADE_DAYS
        return np.divide(mean - self.risk_free, std)
"""

class BaseLauncher:
    RAW_HISTORY_URL_PATTERN = \
            "https://query1.finance.yahoo.com/v7/finance/download/" + \
            "{}?period1={}&period2={}&interval=1d&events=history"
    def DownloadHistory(tickers: List[str], \
            start_date: dttm.date, end_date: dttm.date) -> List[pd.DataFrame]:
        start_period = int((start_date - dttm.date(1970, 1, 1)).total_seconds())
        end_period = int((end_date - dttm.date(1970, 1, 1)).total_seconds())
        raw_history = []
        for ticker in tickers:
            url = BaseLauncher.RAW_HISTORY_URL_PATTERN.format(ticker.upper(), start_period, end_period)
            # XXX:
            print("Loading URL: {}".format(url))
            output = urlopen(url).read()
            str_io = io.StringIO(output.decode('utf-8'))
            raw_history.append(pd.read_csv(str_io, parse_dates=['Date']).fillna(method='ffill'))
        
        return raw_history
    
    def HistoryYahoo2QC(tickers, dfs):
        rename_dict = {\
            'Date': 'time', 'Open': 'open', 'Close': 'close', \
            'High': 'high', 'Low': 'low', 'Volume': 'volume'\
        }
        for df, ticker in zip(dfs, tickers):
            df['symbol'] = ticker
            df.drop(columns=['Adj Close'], inplace=True)
            df.rename(columns=rename_dict, inplace=True)
            df.set_index(['symbol', 'time'], inplace=True)

        return pd.concat(dfs)
    
    def __init__(self, score):
        self.start_date = None
        self.end_date = None
        self.cur_date = None
        self.callbacks = []

        self.tickers = []
        self.cash = 0
        self.weights = np.array([])

        self.hist_start_date = None
        self.hist_end_date = None
        self.hist_tickers = []
        self.history = None
        self.returns = None

        self.score = score

    def InitializeStartEnd(self, start_date: dttm.date, end_date: dttm.date):
        assert self.start_date is None and self.end_date is None
        assert start_date < end_date
        self.start_date = start_date
        self.end_date = end_date
        self.cur_date = self.start_date
    
    def InitializeTickers(self, tickers: List[str]):
        assert len(self.tickers) == 0
        assert len(tickers) > 0
        self.tickers = tickers
        self.weights = np.zeros(len(self.tickers))

    def InitializeCash(self, cash):
        assert cash > 0
        self.cash = cash

    def InitializeHistoryStartEnd(self, start_date: dttm.date, end_date: dttm.date):
        assert self.hist_start_date is None and self.hist_end_date is None
        assert start_date < end_date
        self.hist_start_date = start_date
        self.hist_end_date = end_date

    def InitializeHistoryTickers(self, tickers: List[str]):
        assert len(self.hist_tickers) == 0
        assert len(tickers) > 0
        self.hist_tickers = tickers
    
    def InitializeHistory(self):
        assert len(self.hist_tickers) > 0
        assert self.hist_start_date is not None and self.hist_end_date is not None
        raw_history = BaseLauncher.DownloadHistory(self.hist_tickers, self.hist_start_date, self.hist_end_date)
        self.history = BaseLauncher.HistoryYahoo2QC(self.hist_tickers, raw_history)

    def InitializeReturns(self):
        assert self.returns is None
        assert len(self.tickers) > 0
        assert self.start_date is not None and self.end_date is not None
        raw_history = BaseLauncher.DownloadHistory(self.tickers, self.start_date, self.end_date)
        history = BaseLauncher.HistoryYahoo2QC(self.tickers, raw_history)
        upsampled = [ \
            history.loc[t].resample(rule='D').ffill() \
            for t in self.tickers \
        ]
        returns = [ \
            (((df['close'] - df.shift(1)['close']) / df['close']).fillna(0.0)).rename(t) \
            for t, df in zip(self.tickers, upsampled)
        ]
        self.returns = pd.concat(returns, axis=1)
    
    def GetCurrentDate(self) -> dttm.date:
        return self.cur_date
    
    def SetCallback(self, callback: Callable[[], None], period_days: int):
        assert period_days > 0
        self.callbacks.append((period_days, callback))
    
    def GetHistory(self, tickers: List[str], \
            start_date: dttm.date, end_date: dttm.date) -> pd.DataFrame:
        assert set(tickers).issubset(self.hist_tickers)
        assert start_date >= self.hist_start_date
        assert end_date <= self.hist_end_date
        assert start_date < end_date

        if self.history is None:
            self.InitializeHistory()
        tickers_idx = self.history.index.get_level_values(0)
        date_idx = self.history.index.get_level_values(1)
        return self.history[(tickers_idx.isin(tickers)) & \
                (date_idx >= str(start_date)) & (date_idx < str(end_date))]

    def GetTickers(self) -> List[str]:
        return self.tickers
    
    def SetWeights(self, weights: np.ndarray):
        # FIXME: needs more precise checks
        assert np.all(weights >= -1e-3)
        assert np.sum(weights) <= 1.0 + 1e-3
        # TODO: to calculate the fee
        self.weights = weights
    
    def Run(self) -> np.float64:
        return self.RunUntil(self.end_date)

    def RunUntil(self, date: dttm.date, zero_score=True) -> np.float64:
        assert date <= self.end_date
        if zero_score:
            self.score.Zero()
        # TODO: to jump over the "empty" days
        while self.cur_date < date:
            elapsed = (self.cur_date - self.start_date).days + 1
            for period, callback in self.callbacks:
                if elapsed % period == 0:
                    callback()
            self.AdvanceDays(1)
        
        return self.score.Eval()

    def AdvanceDays(self, count: int):
        assert count > 0
        assert count <= (self.end_date - self.cur_date).days
        delta_returns = self.CalculateNextReturns(count)
        self.score.Update(delta_returns)
        self.cur_date += dttm.timedelta(days=count)

    def CalculateNextReturns(self, count: int) -> np.ndarray:
        assert count > 0
        assert count <= (self.end_date - self.cur_date).days
        start_date = self.cur_date
        end_date = start_date + dttm.timedelta(days=count)

        if self.returns is None:
            self.InitializeReturns()
        idx = self.returns.index
        returns = self.returns[(idx >= str(start_date)) & (idx < str(end_date))]
        # TODO: share splits (?)
        return (returns * self.weights).sum(axis=1).to_numpy()
