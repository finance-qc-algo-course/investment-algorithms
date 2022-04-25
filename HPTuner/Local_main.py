from QCProxy.LocalLauncher import BaseLauncher, SharpeRatioScore, SortinoRatioScore
from QCProxy.LocalPortfolioManager import LocalPortfolioManager
from QCProxy.LocalHistoryManager import LocalHistoryManager
from QCProxy.LocalEventManager import LocalEventManager

from QCProxy.algo.FixedReturnMarkovitz import Algorithm, \
    ALGO_LOOKBACK, ALGO_CASH, ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE, \
    ALGO_HYPERPARAMS

import datetime as dttm

class Launcher(BaseLauncher):
    def __init__(self, score):
        super().__init__(score)
        self.portfolio_manager = LocalPortfolioManager(self)
        self.history_manager = LocalHistoryManager(self)
        self.event_manager = LocalEventManager(self)

        self.algorithm = Algorithm( \
                self.portfolio_manager, \
                self.history_manager, \
                self.event_manager, \
                ALGO_CASH, ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE, \
                ALGO_LOOKBACK, ALGO_HYPERPARAMS)
        pass

if __name__ == "__main__":
    score = SharpeRatioScore()
    launcher = Launcher(score)
    sr = launcher.RunUntil(dttm.date(2008, 1, 1), False)
    sr = launcher.RunUntil(dttm.date(2010, 1, 1), False)
    sr = launcher.RunUntil(dttm.date(2012, 1, 1), False)
    sr = launcher.Run(False)
    print("Sharpe ratio = {}".format(sr))

