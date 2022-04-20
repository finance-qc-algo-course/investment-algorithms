from BaseEventManager import EventManager

import datetime as dttm
from typing import Callable, Type

class QuantConnectEventManager(EventManager):
    def __init__(self, world: QCAlgorithm):
        super().__init__()
        self.world = world
        self.callbacks = []
        self.start_date = None

    def SetStartEnd(self, start_date: dttm.date, end_date: dttm.date):
        assert not self.Ready()
        self.start_date = start_date
        self.world.SetStartDate(start_date.year, start_date.month, start_date.day)
        self.world.SetEndDate(end_date.year, end_date.month, end_date.day)
        self.world.Schedule.On(self.world.DateRules.EveryDay(), \
                               self.world.TimeRules.At(0, 0), \
                               self.OnEveryDay)
    
    def SetCallback(self, callback: Callable[[], None], \
                    period_days: int):
        self.callbacks.append((period_days, callback))

    def GetCurrentDate(self) -> dttm.date:
        return dttm.date(self.world.Time.year, self.world.Time.month, self.world.Time.day)

    def Ready(self) -> bool:
        return self.world is not None and \
                self.start_date is not None

    def OnEveryDay(self):
        elapsed = (self.GetCurrentDate() - self.start_date).days
        for period, cb in self.callbacks:
            if elapsed % period == 0:
                cb()

