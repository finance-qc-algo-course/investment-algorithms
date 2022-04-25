import numpy as np
import pandas as pd
import datetime as dttm
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from QCProxy.LocalLauncher import BaseLauncher, BaseScore, SharpeRatioScore
from QCProxy.LocalDataProvider import BaseDataProvider, YahooDataProvider
from QCProxy.LocalPortfolioManager import LocalPortfolioManager
from QCProxy.LocalHistoryManager import LocalHistoryManager
from QCProxy.LocalEventManager import LocalEventManager

from QCProxy.algo.FixedReturnMarkovitz import Algorithm, \
    ALGO_LOOKBACK, ALGO_CASH, ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE, \
    ALGO_HYPERPARAMS

class LocalLauncher(BaseLauncher):
    def __init__(self, score: BaseScore, data_provider: BaseDataProvider, \
            start_date: dttm.date, end_date: dttm.date, \
            **hyperparams):
        super().__init__(score, data_provider)
        self.portfolio_manager = LocalPortfolioManager(self)
        self.history_manager = LocalHistoryManager(self)
        self.event_manager = LocalEventManager(self)

        self.algorithm = Algorithm(self.portfolio_manager, \
                                   self.history_manager, \
                                   self.event_manager, \
                                   ALGO_CASH, ALGO_TICKERS, \
                                   start_date, end_date, ALGO_LOOKBACK, \
                                   hyperparams)
        pass

# TODO: predict() - ?
class LocalEstimator(BaseEstimator):
    def __init__(self, metric: BaseScore, data_provider: BaseDataProvider, \
            global_start_date: dttm.date, global_end_date: dttm.date, \
            WINDOW_SIZE=365, REBALANCE_PERIOD=365, TOP_COUNT=15, \
            TARGET_RETURN=0.0, PREPROC_KIND=None, PREPROC_RATIO=1.0, \
            DIMRED_KIND=None, DIMRED_RATIO=1.0):
        super().__init__()

        self.global_start_date = global_start_date
        self.global_end_date = global_end_date
        self.metric = metric
        self.data_provider = data_provider
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

    def BuildHyperparams(self):
        hyperparams = {}
        hyperparams["WINDOW_SIZE"] = self.WINDOW_SIZE
        hyperparams["REBALANCE_PERIOD"] = self.REBALANCE_PERIOD
        hyperparams["TOP_COUNT"] = self.TOP_COUNT # in 1..=36
        hyperparams["TARGET_RETURN"] = self.TARGET_RETURN
        # {None, 'pca', 'to_norm_pca', 'mppca', 'to_norm_mppca'}
        hyperparams["PREPROC_KIND"] = self.PREPROC_KIND
        # {None, 'pca', 'kpca', 'mcd'}
        hyperparams["DIMRED_KIND"] = self.DIMRED_KIND

        if self.PREPROC_RATIO <= 1:
            hyperparams["PREPROC_DIMS"] = int(self.TOP_COUNT * self.PREPROC_RATIO)
        else:
            hyperparams["PREPROC_DIMS"] = int(self.PREPROC_RATIO)

        hyperparams["DIMRED_DIMS"] = int(self.TOP_COUNT * self.DIMRED_RATIO)

        hyperparams["PREPROC_PARAMS"] = {
                'pca': {
                    'kept_components': hyperparams["PREPROC_DIMS"],
                },
                'to_norm_pca': {
                    'kept_components': hyperparams["PREPROC_DIMS"],
                },
                'mppca': {
                    'kept_components': hyperparams["PREPROC_DIMS"],
                    'n_models': 2,
                },
                'to_norm_mppca': {
                    'kept_components': hyperparams["PREPROC_DIMS"],
                    'n_models': 2,
                }
            }
        hyperparams["DIMRED_PARAMS"] = {
                'pca': {
                    'n_components': hyperparams["DIMRED_DIMS"],
                },
                'kpca': {
                    'n_components': hyperparams["DIMRED_DIMS"],
                    'kernel': 'poly',
                },
                'mcd': {},
            }

        return hyperparams
    
    def fit(self, data: pd.DatetimeIndex):
        start_date = data[0].date()
        end_date = data[-1].date()
        params = self.BuildHyperparams()
        self.launcher = LocalLauncher(self.metric, self.data_provider, \
                start_date=start_date, end_date=self.global_end_date, \
                **params)
        fit_score = self.launcher.RunUntil(end_date)
        print("FIT {}-{} SCORE {}".format(start_date, end_date, fit_score))

    def predict(self, data: pd.DatetimeIndex):
        start_date = data[0].date()
        end_date = data[-1].date()
        self.launcher.AdvanceDays(start_date)
        self.score_val = self.launcher.AdvanceDays(end_date, zero_score=True)
        print("PREDICT {}-{} SCORE {}".format(start_date, end_date, self.score_val))

    def score(self, data: pd.DatetimeIndex) -> np.float64:
        assert self.launcher is not None
        self.predict(data)
        return self.score_val

if __name__ == "__main__":
    params_grid = { \
        'WINDOW_SIZE': [150, 300, 450, 600, 750], \
        'REBALANCE_PERIOD': [300, 450, 600, 750, 900], \
        'TOP_COUNT': [20, 25, 30], \
        'TARGET_RETURN': [0.0016, 0.0020, 0.0024, 0.0028], \
        'PREPROC_KIND': [None], #, 'pca', 'to_norm_pca', 'mppca', 'to_norm_mppca'], \
        'PREPROC_RATIO': [2, 3, 4, 0.2, 0.4, 0.6, 0.8], \
        'DIMRED_KIND': [None], # ['pca'], \
        'DIMRED_RATIO': [None] # [0.2, 0.4, 0.6, 0.8], \
    }
    max_window = max(params_grid['WINDOW_SIZE']) + 1
    metric = SharpeRatioScore(risk_free=0.0)
    data_provider = YahooDataProvider(ALGO_TICKERS, \
            ALGO_START_DATE - dttm.timedelta(days=max_window), ALGO_END_DATE)
    runner = LocalEstimator(metric, data_provider, \
            ALGO_START_DATE, ALGO_END_DATE)

    # TODO: to choose init params for TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    # TODO: to choose init params for RandomizedSearchCV
    restricted_grid = {
            name: [vals[0], vals[-1]] for name, vals in params_grid.items()
        }
    rs = RandomizedSearchCV(\
            estimator=runner, \
            param_distributions=params_grid, \
            cv=tscv, \
            n_iter=10)

    data = pd.date_range(start=ALGO_START_DATE, end=ALGO_END_DATE)
    rs.fit(data)

    print("BEST PARAMS:")
    print(rs.best_params_)

