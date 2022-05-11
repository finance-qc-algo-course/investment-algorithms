import abc
import datetime as dttm
from typing import Callable, Type


class EventManager(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def SetStartEnd(self, start_date: dttm.date, end_date: dttm.date):
        pass

    @abc.abstractmethod
    def SetCallback(self, callback: Callable[[], None],
                    period_days: int):
        pass

    @abc.abstractmethod
    def GetCurrentDate(self) -> dttm.date:
        pass

    def Ready(self) -> bool:
        return False
