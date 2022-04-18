from ../../BasePortfolioManager import PortfolioManager

import numpy as np
from typing import List

class QuantConnectPortfolioManager(PortfolioManager):
    def __init__(self, world: QCAlgorithm):
        super().__init__()
        self.world = world
        self.cash = None
        self.tickers = None
        self.symbols = None
    
    def SetCash(self, cash: int):
        assert not self.Ready()
        self.cash = cash
        self.world.SetCash(cash)

    def GetTickers(self) -> List[str]:
        return self.tickers

    def SetTickers(self, tickers: List[str]):
        assert not self.Ready()
        self.tickers = tickers
        self.symbols = [ \
                Symbol.Create(t, SecurityType.Equity, Market.USA) \
                for t in self.tickers \
                ]
        self.world.UniverseSettings.Resolution = Resolution.Daily
        self.world.AddUniverseSelection(ManualUniverseSelectionModel(self.symbols))

    def SetWeights(self, weights: np.ndarray):
        self.world.SetHoldings( [ PortfolioTarget(s, w) \
                for s, w in zip(self.symbols, weights) ])

    def Ready(self) -> bool:
        return self.world is not None and \
                self.cash is not None and \
                self.tickers is not None and \
                self.symbols is not None
