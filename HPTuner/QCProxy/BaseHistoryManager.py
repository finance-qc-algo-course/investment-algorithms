import abc
import datetime as dttm
import pandas as pd
from typing import List


class HistoryManager(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def SetTickers(self, tickers: List[str]):
        pass

    @abc.abstractmethod
    def SetStartEnd(self, start_date: dttm.date, end_date: dttm.date):
        pass

    @abc.abstractmethod
    def GetHistory(self, tickers: List[str],
                   start_date: dttm.date, end_date: dttm.date) -> pd.DataFrame:
        pass

    def Ready(self) -> bool:
        return False
