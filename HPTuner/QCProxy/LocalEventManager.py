from .BaseEventManager import EventManager
from .LocalLauncher import BaseLauncher

import datetime as dttm
from typing import Callable, Type

class LocalEventManager(EventManager):
    def __init__(self, world: BaseLauncher):
        super().__init__()
        self.world = world

    def SetStartEnd(self, start_date: dttm.date, end_date: dttm.date):
        self.world.InitializeStartEnd(start_date, end_date)
    
    def SetCallback(self, callback: Callable[[], None], \
                    period_days: int):
        self.world.SetCallback(callback, period_days)

    def GetCurrentDate(self) -> dttm.date:
        return self.world.GetCurrentDate()

    def Ready(self) -> bool:
        return self.world is not None

