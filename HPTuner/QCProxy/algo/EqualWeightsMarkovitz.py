from ..Interface import Interface

from . import custom_tickers

import numpy as np
import datetime as dttm

from typing import List

ALGO_HYPERPARAMS = {
    "TOP_COUNT": 15, # in 1..=36
}

ALGO_TICKERS = custom_tickers.get_custom_top_tickers(ALGO_HYPERPARAMS["TOP_COUNT"])
ALGO_START_DATE = dttm.date(2006, 1, 1)
ALGO_END_DATE = dttm.date(2014, 1, 1)
ALGO_CASH = 100000

class Algorithm(Interface):
    def __init__(self, portfolio_manager, history_manager, event_manager, \
                 cash: int, tickers: List[str], \
                 start_date: dttm.date, end_date: dttm.date, \
                 hyperparams=ALGO_HYPERPARAMS):
        super().__init__( \
                portfolio_manager, history_manager, event_manager, \
                cash, tickers, start_date, end_date, lookback=0)
        self.InitializeHyperparams(hyperparams)
        self.InitializeAlgorithm()

    def InitializeHyperparams(self, hyperparams):
        self.TOP_COUNT = hyperparams["TOP_COUNT"] # in 1..=36

    def InitializeAlgorithm(self):
        self.count = len(self.GetTickers())
        self.invested = False

        self.SetCallback(self.MarkovitzOnEveryPeriod, period_days=1)

    def MarkovitzOnEveryPeriod(self):
        if not self.invested:
            weights = np.divide(np.ones(self.count), self.count)
            self.SetWeights(weights)
            self.invested = True

