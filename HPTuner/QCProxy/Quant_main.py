from QuantPortfolioManager import QuantConnectPortfolioManager
from QuantHistoryManager import QuantConnectHistoryManager
from QuantEventManager import QuantConnectEventManager

from algo.fixed_markovitz_algorithm import Algorithm, \
    ALGO_HYPERPARAMS, ALGO_CASH, ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE

class QuantConnectLauncher(QCAlgorithm):
    def Initialize(self):
        self.portfolio_manager = QuantConnectPortfolioManager(self)
        self.history_manager = QuantConnectHistoryManager(self)
        self.event_manager = QuantConnectEventManager(self)

        self.algorithm = Algorithm(ALGO_CASH, ALGO_TICKERS, \
                                   ALGO_START_DATE, ALGO_END_DATE, \
                                   self.portfolio_manager, \
                                   self.history_manager, \
                                   self.event_manager, \
                                   ALGO_HYPERPARAMS)
        pass
    
    def OnData(self, data):
        pass
