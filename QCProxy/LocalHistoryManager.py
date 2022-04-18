from BaseHistoryManager import HistoryManager
from LocalLauncher import BaseLauncher

import datetime as dttm
import pandas as pd
from typing import List

class LocalHistoryManager(HistoryManager):
    def __init__(self, world: BaseLauncher):
        super().__init__()
        self.world = world
    
    def SetTickers(self, tickers: List[str]):
        self.world.InitializeHistoryTickers(tickers)

    def SetStartEnd(self, start_date: dttm.date, end_date: dttm.date):
        self.world.InitializeHistoryStartEnd(start_date, end_date)

    def GetHistory(self, tickers: List[str], \
            start_date: dttm.date, end_date: dttm.date) -> pd.DataFrame:
        return self.world.GetHistory(tickers, start_date, end_date)

    def Ready(self) -> bool:
        return self.world is not None
