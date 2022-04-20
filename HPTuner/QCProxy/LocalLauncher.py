import abc
import numpy as np
import pandas as pd
import datetime as dttm
from typing import Callable, List

from .LocalDataProvider import BaseDataProvider, YahooDataProvider

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

# TODO check if thiss is the right way according to the Lean engine:
class SortinoRatioScore(BaseScore):
    def __init__(self, risk_free: float = 0.0):
        super().__init__()
        self.risk_free = risk_free
        self.returns = np.array([], dtype=np.float64)

    def Zero(self):
        self.returns = np.array([], dtype=np.float64)
    
    def Update(self, delta_returns: np.ndarray):
        self.returns = np.append(self.returns, delta_returns)

    def Eval(self) -> np.float64:
        mean = np.power(np.mean(np.maximum(self.returns, 0.0)) + 1.0, \
                BaseScore.TRADE_DAYS) - 1.0
        var = np.var(self.returns) * BaseScore.TRADE_DAYS
        return np.divide(mean - self.risk_free, var)

class BaseLauncher:
    def __init__(self, score: BaseScore, data_provider: BaseDataProvider = None):
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
        self.data_provider = data_provider
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

    def InitializeHistoryStartEnd(self, start_date: dttm.date, \
            end_date: dttm.date):
        assert self.hist_start_date is None and self.hist_end_date is None
        assert start_date < end_date
        self.hist_start_date = start_date
        self.hist_end_date = end_date

    def InitializeHistoryTickers(self, tickers: List[str]):
        assert len(self.hist_tickers) == 0
        assert len(tickers) > 0
        self.hist_tickers = tickers
    
    def InitializeDataProvider(self):
        assert self.data_provider is None
        assert len(self.hist_tickers) > 0
        assert self.hist_start_date is not None and \
                self.hist_end_date is not None
        self.data_provider = YahooDataProvider(self.hist_tickers, \
                self.hist_start_date, self.hist_end_date)
    
    def GetCurrentDate(self) -> dttm.date:
        return self.cur_date
    
    def SetCallback(self, callback: Callable[[], None], period_days: int):
        assert period_days > 0
        self.callbacks.append((period_days, callback))
    
    def GetHistory(self, tickers: List[str], \
            start_date: dttm.date, end_date: dttm.date) -> pd.DataFrame:
        if self.data_provider is None:
            self.InitializeDataProvider()
        return self.data_provider.GetHistory(tickers, start_date, end_date)

    def GetReturns(self, tickers: List[str], \
            start_date: dttm.date, end_date: dttm.date) -> pd.DataFrame:
        if self.data_provider is None:
            self.InitializeDataProvider()
        return self.data_provider.GetReturns(tickers, start_date, end_date)

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

        # Collecting events & sorting them by the date
        events = []
        for period, callback in self.callbacks:
            dr = pd.date_range(start=self.cur_date, end=date) \
                    [: : period].to_pydatetime()
            events.append(np.stack([dr, np.full(len(dr), callback)], axis=1))
        events = np.concatenate(events)
        events = events[np.argsort(events[:,0], kind='stable')]

        # "Jumping" over the empty days
        for dt, callback in events:
            self.AdvanceDays((dt.date() - self.cur_date).days)
            callback()
        self.AdvanceDays((date - self.cur_date).days)
        
        return self.score.Eval()

    def AdvanceDays(self, count: int):
        assert count >= 0
        assert count <= (self.end_date - self.cur_date).days
        if count == 0:
            return
        delta_returns = self.CalculateNextReturns(count)
        self.score.Update(delta_returns)
        self.cur_date += dttm.timedelta(days=count)

    def CalculateNextReturns(self, count: int) -> np.ndarray:
        assert count > 0
        assert count <= (self.end_date - self.cur_date).days
        start_date = self.cur_date
        end_date = start_date + dttm.timedelta(days=count)
        returns = self.GetReturns(self.tickers, start_date, end_date)
        # TODO: share splits (?)
        return (returns * self.weights).sum(axis=1).to_numpy()

