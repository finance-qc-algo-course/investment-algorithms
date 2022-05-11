from QCProxy.LocalLauncher import BaseLauncher, BaseScore
from QCProxy.MockDataProvider import MockDataProvider
from QCProxy.LocalPortfolioManager import LocalPortfolioManager
from QCProxy.LocalHistoryManager import LocalHistoryManager
from QCProxy.LocalEventManager import LocalEventManager

from QCProxy.algo.EqualWeightsMarkovitz import Algorithm, \
    ALGO_CASH, ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE, \
    ALGO_HYPERPARAMS

import numpy as np
import datetime as dttm


class AnnualReturnScore(BaseScore):
    def __init__(self):
        super().__init__()
        self.returns = np.array([], dtype=np.float64)

    def Zero(self):
        self.returns = np.array([], dtype=np.float64)

    def Update(self, delta_returns: np.ndarray):
        self.returns = np.append(self.returns, delta_returns)

    def Eval(self) -> np.float64:
        return np.power(np.mean(self.returns) + 1.0, BaseScore.TRADE_DAYS) - 1.0


class SimpleSharpeScore(BaseScore):
    def __init__(self):
        super().__init__()
        self.returns = np.array([], dtype=np.float64)

    def Zero(self):
        self.returns = np.array([], dtype=np.float64)

    def Update(self, delta_returns: np.ndarray):
        self.returns = np.append(self.returns, delta_returns)

    def Eval(self) -> np.float64:
        if len(self.returns) <= 1:
            return np.float64(0.0)
        return np.divide(np.mean(self.returns), np.std(self.returns))


class Launcher(BaseLauncher):
    def __init__(self, score, data_provider=None):
        super().__init__(score, data_provider)
        self.portfolio_manager = LocalPortfolioManager(self)
        self.history_manager = LocalHistoryManager(self)
        self.event_manager = LocalEventManager(self)

        self.algorithm = Algorithm(
            self.portfolio_manager,
            self.history_manager,
            self.event_manager,
            ALGO_CASH, ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE,
            ALGO_HYPERPARAMS)
        pass


def test_return():
    annual_r = 0.20
    daily_r = (1.0 + annual_r) ** (1.0 / BaseScore.TRADE_DAYS) - 1.0
    returns = np.linspace(daily_r * 0.5, daily_r * 1.5, num=len(ALGO_TICKERS))
    data_provider = MockDataProvider(ALGO_TICKERS, returns,
                                     ALGO_START_DATE, ALGO_END_DATE)

    launcher = Launcher(AnnualReturnScore(), data_provider)
    value = launcher.Run(False)
    print("AnnualReturnScore = {}".format(value))
    assert value > annual_r * 0.8 and value < annual_r * 1.25


def test_sharpe():
    annual_r = 0.20
    daily_r = (1.0 + annual_r) ** (1.0 / BaseScore.TRADE_DAYS) - 1.0
    returns = np.linspace(daily_r * 0.5, daily_r * 1.5, num=len(ALGO_TICKERS))
    data_provider = MockDataProvider(ALGO_TICKERS, returns,
                                     ALGO_START_DATE, ALGO_END_DATE)

    launcher = Launcher(SimpleSharpeScore(), data_provider)
    value = launcher.Run(False)
    assert value > 0.0
    print("SimpleSharpeScore = {}".format(value))


if __name__ == "__main__":
    test_return()
    test_sharpe()
