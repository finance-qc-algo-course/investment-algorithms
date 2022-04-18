from ../../BaseHistoryManager import HistoryManager

import datetime as dttm
import pandas as pd
from typing import List

class QuantConnectHistoryManager(HistoryManager):
    def __init__(self, world: QCAlgorithm):
        super().__init__()
        self.world = world
    
    def SetTickers(self, tickers: List[str]):
        # No need to put anything here because QC already has data for all tickers
        pass

    def SetStartEnd(self, start_date: dttm.date, end_date: dttm.date):
        # No need to put anything here because QC already has data for all time
        pass

    def GetHistory(self, tickers: List[str], \
            start_date: dttm.date, end_date: dttm.date) -> pd.DataFrame:
        return self.world.History(tickers, start_date, end_date, Resolution.Daily)

    def Ready(self) -> bool:
        return self.world is not None
