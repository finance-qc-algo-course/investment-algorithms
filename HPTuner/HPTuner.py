import numpy as np
import pandas as pd
import datetime as dttm
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from QCProxy.LocalLauncher import BaseLauncher, BaseScore, SharpeRatioScore
from QCProxy.LocalPortfolioManager import LocalPortfolioManager
from QCProxy.LocalHistoryManager import LocalHistoryManager
from QCProxy.LocalEventManager import LocalEventManager

from QCProxy.algo.FixedReturnMarkovitz import Algorithm, \
    ALGO_HYPERPARAMS, ALGO_CASH, ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE

class LocalLauncher(BaseLauncher):
    def __init__(self, score: BaseScore, \
            start_date: dttm.date, end_date: dttm.date, \
            **hyperparams):
        super().__init__(score)
        self.portfolio_manager = LocalPortfolioManager(self)
        self.history_manager = LocalHistoryManager(self)
        self.event_manager = LocalEventManager(self)

        self.algorithm = Algorithm(ALGO_CASH, ALGO_TICKERS, \
                                   start_date, end_date, \
                                   self.portfolio_manager, \
                                   self.history_manager, \
                                   self.event_manager, \
                                   hyperparams)
        pass


# TODO: predict() - ?
class LocalEstimator(BaseEstimator):
    def __init__(self, metric: BaseScore, \
            global_start_date: dttm.date, global_end_date: dttm.date, \
            WINDOW_SIZE=365, REBALANCE_PERIOD=365, TOP_COUNT=15, \
            TARGET_RETURN=0.0, PREPROC_KIND=None, PREPROC_RATIO=1.0, \
            DIMRED_KIND=None, DIMRED_RATIO=1.0):
        super().__init__()

        self.global_start_date = global_start_date
        self.global_end_date = global_end_date
        self.metric = metric
        self.launcher = None
        self.score_val = 0.0

        self.WINDOW_SIZE = WINDOW_SIZE
        self.REBALANCE_PERIOD = REBALANCE_PERIOD
        self.TOP_COUNT = TOP_COUNT
        self.TARGET_RETURN = TARGET_RETURN
        self.PREPROC_KIND = PREPROC_KIND
        self.PREPROC_RATIO = PREPROC_RATIO
        self.DIMRED_KIND = DIMRED_KIND
        self.DIMRED_RATIO = DIMRED_RATIO

        self.PREPROC_DIMS = int(self.TOP_COUNT * self.PREPROC_RATIO)
        self.PREPROC_PARAMS = {
                'pca': {
                    'kept_components': self.PREPROC_DIMS,
                },
                'to_norm_pca': {
                    'kept_components': self.PREPROC_DIMS,
                },
                'mppca': {
                    'kept_components': self.PREPROC_DIMS,
                    'n_models': 2,
                },
            }
        self.DIMRED_DIMS = int(self.TOP_COUNT * self.DIMRED_RATIO)
        self.DIMRED_PARAMS = {
                'pca': {
                    'n_components': self.DIMRED_DIMS,
                },
                'kpca': {
                    'n_components': self.DIMRED_DIMS,
                    'kernel': 'poly',
                },
                'mcd': {},
            }
    
    def fit(self, data: pd.DatetimeIndex):
        window = data[:self.WINDOW_SIZE]
        start_date = window[0].date()
        end_date = window[-1].date()
        params = {
                k: v for k, v in self.get_params().items() if k in [ \
                        'WINDOW_SIZE', 'REBALANCE_PERIOD', \
                        'TOP_COUNT', 'TARGET_RETURN', \
                        'PREPROC_KIND', 'PREPROC_RATIO', \
                        'DIMRED_KIND', 'DIMRED_RATIO' \
                    ]
            }
        self.launcher = LocalLauncher(self.metric, \
                start_date=start_date, end_date=self.global_end_date, \
                **params)
        self.launcher.RunUntil(end_date) # We don't nee score here Run(w[0], w[-1])
        return self

    def predict(self, data: pd.DatetimeIndex):
        end_date = data[-1].date()
        self.score_val = self.launcher.RunUntil(end_date) # TODO: Run between Run(data[0], data[-1])

    def score(self, data: pd.DatetimeIndex) -> float:
        assert self.launcher is not None
        return self.score_val

if __name__ == "__main__":
    params = { \
        'WINDOW_SIZE': [150, 300, 450, 600, 750], \
        'REBALANCE_PERIOD': [300, 450, 600, 750, 900], \
        'TOP_COUNT': [5, 10, 15, 20, 25], \
        'TARGET_RETURN': [0.0016, 0.0020, 0.0024, 0.0028], \
        'PREPROC_KIND': [None, 'pca', 'to_norm_pca', 'mppca'], \
        'PREPROC_RATIO': [0.2, 0.4, 0.6, 0.8], \
        'DIMRED_KIND': [None, 'pca', 'kpca', 'mcd'], \
        'DIMRED_RATIO': [0.2, 0.4, 0.6, 0.8], \
    }
    default = { name: [vals[0], vals[-1]] for name, vals in params.items() }
    score = SharpeRatioScore(risk_free=0.0)
    runner = LocalEstimator(score, ALGO_START_DATE, ALGO_END_DATE)
    print(runner.get_params())

    # TODO: to choose init params for TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    # TODO: to choose init params for RandomizedSearchCV
    rs = RandomizedSearchCV(\
            estimator=runner, \
            param_distributions=default, \
            cv=tscv)

    data = pd.date_range(start=ALGO_START_DATE, end=ALGO_END_DATE)
    rs.fit(data)
    print(rs.best_params_)

