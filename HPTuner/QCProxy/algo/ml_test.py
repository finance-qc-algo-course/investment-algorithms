import unittest

import numpy as np
import pandas as pd
import scipy.stats as sps
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from QCProxy.algo.FixedReturnMarkovitz import ALGO_TICKERS, ALGO_START_DATE, ALGO_END_DATE
from HPTuner import LocalEstimator
from QCProxy.LocalDataProvider import YahooDataProvider
from QCProxy.LocalLauncher import SharpeRatioScore
from Local_main import Launcher

import datetime as dttm


class DataTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_data_before_model(self):
        score = SharpeRatioScore()
        launcher = Launcher(score)
        launcher.Run(False)
        hist = launcher.algorithm.prices
        self.assertIsInstance(hist, np.ndarray, 'hist is not a np.ndarray')
        self.assertTrue((np.all(hist < 10) & np.all(hist > -1)), 'hist values are not correct')

    def test_data_after_preprocessing(self):
        score = SharpeRatioScore()
        launcher = Launcher(score)
        launcher.Run(False)
        comp_hist = launcher.algorithm.optimized_data
        self.assertIsInstance(comp_hist, np.ndarray, 'comp_hist is not a np.ndarray')
        self.assertTrue((np.all(comp_hist < 10) & np.all(comp_hist > -1)), 'comp_hist values are not correct')


class TestShapesAndWeights(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_sigma_shape(self):
        score = SharpeRatioScore()
        launcher = Launcher(score)
        launcher.Run(False)
        sigma = launcher.algorithm.sigma
        self.assertIsInstance(sigma, pd.DataFrame, 'covariance matrix is not a pd.DataFrame')
        self.assertEqual(sigma.shape[1], launcher.algorithm.DIMRED_DIMS, 'covariance matrix has an incorrect shape')

    def test_components_shape(self):
        score = SharpeRatioScore()
        launcher = Launcher(score)
        launcher.Run(False)
        components = launcher.algorithm.components_
        self.assertIsInstance(components, pd.DataFrame, 'components are not a pd.DataFrame')
        self.assertEqual(components.shape[1], launcher.algorithm.TOP_COUNT, 'num of components is incorrect')
        self.assertEqual(components.shape[0], launcher.algorithm.DIMRED_DIMS, 'components have an incorrect len')

    def test_mu_len(self):
        score = SharpeRatioScore()
        launcher = Launcher(score)
        launcher.Run(False)
        mu = launcher.algorithm.mu
        self.assertIsInstance(mu, np.ndarray, 'average return is not a np.ndarray')
        self.assertEqual(len(mu), launcher.algorithm.DIMRED_DIMS, 'average return has an incorrect len')

    def test_weights(self):
        score = SharpeRatioScore()
        launcher = Launcher(score)
        launcher.Run(False)
        w = launcher.weights
        self.assertIsNotNone(w, 'w is None')
        self.assertIsInstance(w, np.ndarray, 'weights are not a np.ndarray')
        self.assertEqual(len(w), launcher.algorithm.TOP_COUNT, 'weights have an incorrect len')
        self.assertTrue(np.isclose(np.sum(w), 1, atol=1e-4), 'sum of weights is not 1')


class TestFixedReturn(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_fixed_return(self):
        score = SharpeRatioScore()
        launcher = Launcher(score)
        launcher.Run(False)
        fixed_return = launcher.algorithm.fixed_return
        mu = launcher.algorithm.mu
        self.assertIsInstance(fixed_return, float)
        self.assertTrue((np.max(mu) > fixed_return > np.min(mu)), 'fixed_return is not correct')


class TestScore(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.NUM_ITERS = 10

    def test_score(self):
        params_grid = { \
            'WINDOW_SIZE': [300, 450, 600, 750], \
            'REBALANCE_PERIOD': [150, 300, 450, 600, 750, 900], \
            'TOP_COUNT': np.linspace(15, 36, 22, dtype=int), \
            'TARGET_RETURN': [0.2], \
            'TARGET_QUANTILE': sps.beta(8, 3), \
            'PREPROC_KIND': [None, 'pca', 'to_norm_pca', 'mppca', 'to_norm_mppca'], \
            'PREPROC_RATIO': sps.uniform(loc=0.1, scale=0.8), \
            'DIMRED_KIND': [None, 'pca', 'kpca'],  # ['pca'], \
            'DIMRED_RATIO': sps.uniform(loc=0.1, scale=0.8), \
            }
        max_window = max(params_grid['WINDOW_SIZE']) + 1
        metric = SharpeRatioScore(risk_free=0.0)
        data_provider = YahooDataProvider(ALGO_TICKERS, \
                                          ALGO_START_DATE - dttm.timedelta(days=max_window), ALGO_END_DATE)
        runner = LocalEstimator(metric, data_provider, \
                                ALGO_START_DATE, ALGO_END_DATE, THRESHOLD=1e-5)

        # TODO: to choose init params for TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        # TODO: to choose init params for RandomizedSearchCV
        rs = RandomizedSearchCV( \
            estimator=runner, \
            param_distributions=params_grid, \
            cv=tscv, \
            n_iter=self.NUM_ITERS,
            random_state=15,
            n_jobs=-2,
            # verbose=2
        )

        data = pd.date_range(start=ALGO_START_DATE, end=ALGO_END_DATE)
        rs.fit(data)
        score = rs.best_score_
        self.assertIsInstance(score, float)
        self.assertTrue((0 < score < 10), 'score is not correct or algo is bad')


if __name__ == "__main__":
    unittest.main()
