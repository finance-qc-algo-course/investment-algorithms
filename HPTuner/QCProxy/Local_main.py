from LocalLauncher import BaseLauncher, SharpeRatioScore # TODO: , SortinoRatioScore
from LocalPortfolioManager import LocalPortfolioManager
from LocalHistoryManager import LocalHistoryManager
from LocalEventManager import LocalEventManager

from algo.FixedReturnMarkovitz import Algorithm, \
    ALGO_HYPERPARAMS, ALGO_CASH, ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE

class Launcher(BaseLauncher):
    def __init__(self, score):
        super().__init__(score)
        self.portfolio_manager = LocalPortfolioManager(self)
        self.history_manager = LocalHistoryManager(self)
        self.event_manager = LocalEventManager(self)

        self.algorithm = Algorithm(ALGO_CASH, ALGO_TICKERS, \
                                   ALGO_START_DATE, ALGO_END_DATE, \
                                   self.portfolio_manager, \
                                   self.history_manager, \
                                   self.event_manager, \
                                   ALGO_HYPERPARAMS)
        pass

if __name__ == "__main__":
    score = SharpeRatioScore()
    launcher = Launcher(score)
    sr = launcher.Run()
    print("Sharpe ratio = {}".format(sr))
