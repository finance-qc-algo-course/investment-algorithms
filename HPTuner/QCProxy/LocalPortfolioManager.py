from .BasePortfolioManager import PortfolioManager
from .LocalLauncher import BaseLauncher

import numpy as np
from typing import List

class LocalPortfolioManager(PortfolioManager):
    def __init__(self, world: BaseLauncher):
        super().__init__()
        self.world = world
    
    def SetCash(self, cash: int):
        self.world.InitializeCash(cash)

    def GetTickers(self) -> List[str]:
        return self.world.GetTickers()

    def SetTickers(self, tickers: List[str]):
        self.world.InitializeTickers(tickers)

    def SetWeights(self, weights: np.ndarray):
        self.world.SetWeights(weights)

    def Ready(self) -> bool:
        return self.world is not None

