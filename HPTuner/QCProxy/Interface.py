import numpy as np
import pandas as pd
import datetime as dttm
from typing import Callable, List, Type

class Interface:
    def __init__(self, cash: int, tickers: List[str], \
                 start_date: dttm.date, end_date: dttm.date, \
                 portfolio_manager, history_manager, event_manager):
        assert len(tickers) > 0
        assert start_date < end_date

        self.portfolio_manager_ = portfolio_manager
        self.portfolio_manager_.SetCash(cash)
        self.portfolio_manager_.SetTickers(tickers)
        assert self.portfolio_manager_.Ready()

        self.history_manager_ = history_manager
        self.history_manager_.SetTickers(tickers)
        self.history_manager_.SetStartEnd(start_date, end_date)
        assert self.history_manager_.Ready()

        self.event_manager_ = event_manager
        self.event_manager_.SetStartEnd(start_date, end_date)
        assert self.event_manager_.Ready()

    def Ready(self) -> bool:
        return \
            self.portfolio_manager_.Ready() and \
            self.history_manager_.Ready() and \
            self.event_manager_.Ready()

    def GetTickers(self) -> List[str]:
        return self.portfolio_manager_.GetTickers()

    def SetTickers(self, tickers: List[str]):
        self.portfolio_manager_.SetTickers(tickers)

    def SetWeights(self, weights: np.ndarray):
        assert len(weights) == len(self.portfolio_manager_.GetTickers())
        self.portfolio_manager_.SetWeights(weights)
    
    def GetHistory(self, tickers: List[str], window_days: int) -> pd.DataFrame:
        assert window_days > 0
        curr_date = self.event_manager_.GetCurrentDate()
        prev_date = curr_date - dttm.timedelta(days=window_days)
        return self.history_manager_.GetHistory(tickers, \
                start_date=prev_date, end_date=curr_date)

    def SetCallback(self, callback: Callable[[], None], \
                    period_days: int):
        assert period_days > 0
        self.event_manager_.SetCallback(callback, period_days)

    # TODO:
    """
    def SetUniverse(self, tickers: List[str], predicate: Callable[[T], float], prefix: int, period_days: int)
        ...
        T == "AAPL"
        ("R", period_days)
        if day % period_days == 0:
            GetHistory([t for t in tickers if QC.available(t)], period) -> calculate Return
            self.tickers = sort(tickers, predicate)[prefix:]
    """

