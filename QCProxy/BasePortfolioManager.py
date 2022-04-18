import abc
import numpy as np
from typing import List

class PortfolioManager(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def SetCash(self, cash: int):
        pass

    @abc.abstractmethod
    def GetTickers(self) -> List[str]:
        pass

    @abc.abstractmethod
    def SetTickers(self, tickers: List[str]):
        pass

    @abc.abstractmethod
    def SetWeights(self, weights: np.ndarray):
        pass

    def Ready(self) -> bool:
        return False
