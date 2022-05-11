from QuantPortfolioManager import QuantConnectPortfolioManager
from QuantHistoryManager import QuantConnectHistoryManager
from QuantEventManager import QuantConnectEventManager

from algo.fixed_markovitz_algorithm import Algorithm, \
    ALGO_LOOKBACK, ALGO_CASH, ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE, \
    ALGO_HYPERPARAMS


class QuantConnectLauncher(QCAlgorithm):
    def Initialize(self):
        self.portfolio_manager = QuantConnectPortfolioManager(self)
        self.history_manager = QuantConnectHistoryManager(self)
        self.event_manager = QuantConnectEventManager(self)

        self.algorithm = Algorithm(
            self.portfolio_manager,
            self.history_manager,
            self.event_manager,
            ALGO_CASH, ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE,
            ALGO_LOOKBACK, ALGO_HYPERPARAMS)
        pass

    def OnData(self, data):
        pass

    # TODO:
    def OnPortfolioChanged(self, changes):
        pass
