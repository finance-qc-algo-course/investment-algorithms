from Interface import Interface

from . import custom_tickers
from . import cov_matrix_preprocessing

import numpy as np
import cvxpy as cp
import functools
import datetime as dttm

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.covariance import MinCovDet

from typing import List

ALGO_HYPERPARAMS = {
    "WINDOW_SIZE": 735,
    "REBALANCE_PERIOD": 900,
    "TOP_COUNT": 15, # in 1..=46
    "TARGET_RETURN": 0.0028,
    "PREPROC_KIND": None, # {None, 'pca', 'to_norm_pca', 'mppca'}
    "PREPROC_DIMS": 14,
    "PREPROC_PARAMS": {
        "pca": {
            "kept_components": 14, # self.PREPROC_DIMS,
        },
        "to_norm_pca": {
            "kept_components": 14, # self.PREPROC_DIMS,
        },
        "mppca": {
            "kept_components": 14, # self.PREPROC_DIMS,
            "n_models": 2,
        },
    },
    "DIMRED_KIND": 'kpca', # {None, 'pca', 'kpca', 'mcd'}
    "DIMRED_DIMS": 8,
    "DIMRED_PARAMS": {
        "pca": {
            "n_components": 8,
        },
        "kpca": {
            "n_components": 8,
            "kernel": "poly",
        },
        "mcd": {},
    },
}

ALGO_TICKERS = custom_tickers.get_custom_top_tickers(ALGO_HYPERPARAMS["TOP_COUNT"])
ALGO_START_DATE = dttm.date(2006, 1, 1) - \
        max(dttm.timedelta(days=ALGO_HYPERPARAMS["WINDOW_SIZE"]), \
            dttm.timedelta(days=ALGO_HYPERPARAMS["REBALANCE_PERIOD"]))
ALGO_END_DATE = dttm.date(2014, 1, 1)
ALGO_CASH = 100000

class Algorithm(Interface):
    def __init__(self, cash: int, tickers: List[str], \
                 start_date: dttm.date, end_date: dttm.date, \
                 portfolio_manager, history_manager, event_manager, \
                 hyperparams):
        super().__init__(cash, tickers, start_date, end_date, \
                portfolio_manager, history_manager, event_manager)
        self.InitializeHyperparams(hyperparams)
        self.InitializeAlgorithm()

    def InitializeHyperparams(self, hyperparams):
        self.WINDOW_SIZE = ALGO_HYPERPARAMS["WINDOW_SIZE"]
        self.REBALANCE_PERIOD = ALGO_HYPERPARAMS["REBALANCE_PERIOD"]
        self.TOP_COUNT = ALGO_HYPERPARAMS["TOP_COUNT"] # in 1..=46
        self.TARGET_RETURN = ALGO_HYPERPARAMS["TARGET_RETURN"]
        self.PREPROC_KIND = ALGO_HYPERPARAMS["PREPROC_KIND"] # {None, 'pca', 'to_norm_pca', 'mppca'}
        self.PREPROC_DIMS = ALGO_HYPERPARAMS["PREPROC_DIMS"]
        self.PREPROC_PARAMS = ALGO_HYPERPARAMS["PREPROC_PARAMS"]
        self.DIMRED_KIND = ALGO_HYPERPARAMS["DIMRED_KIND"] # {None, 'pca', 'kpca', 'mcd'}
        self.DIMRED_DIMS = ALGO_HYPERPARAMS["DIMRED_DIMS"]
        self.DIMRED_PARAMS = ALGO_HYPERPARAMS["DIMRED_PARAMS"]

    def InitializeAlgorithm(self):
        # Markovitz portfolio data
        # (W.T @ Sigma @ W - RiskTol * W.T @ Mu) -> min
        self.count = len(self.GetTickers())
        self.sigma = np.eye(self.count)
        self.mu = np.ones(self.count)
        self.weights = np.divide(np.ones(self.count), self.count)

        # The following values have to be reevaluated every rebalance step
        self.fixed_return = 1.0 + self.TARGET_RETURN

        self.SetCallback(self.MarkovitzOnEveryPeriod, self.REBALANCE_PERIOD)

    def MarkovitzOnEveryPeriod(self):
        prices = self.MarkovitzGetPrices(days=self.WINDOW_SIZE)
        self.MarkovitzRebalance(prices)

    def MarkovitzRebalance(self, prices):
        self.MarkovitzUpdateParams(prices)
        prices = self.MarkovitzPreprocess(prices, kind=self.PREPROC_KIND)
        self.MarkovitzReduceDimensions(kind=self.DIMRED_KIND)
        self.MarkovitzOptimize()
        self.SetWeights(self.weights)

    def MarkovitzOptimize(self):
        # XXX sum_weight = np.sum(self.weights)
        # XXX self.weights = np.divide(self.weights, sum_weight)
        w = cp.Variable(self.count)
        objective = cp.Minimize(cp.quad_form(w, self.sigma)) # * 0.5
        constraints = [ \
                w.T >= 0.0, \
                w.T @ np.ones(self.count) == 1.0, \
                w.T @ self.mu == self.fixed_return, \
            ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        self.weights = w.value


    def MarkovitzGetPrices(self, days=None):
        days = days or self.WINDOW_SIZE
        hist = self.GetHistory(self.GetTickers(), days)
        symbols = np.unique(hist.index.get_level_values(0).to_numpy())
        idx = functools.reduce(np.intersect1d, \
                        (hist.loc[s].index for s in symbols))
        
        prices = np.array([ \
                ((hist.loc[s].loc[idx]['open'].to_numpy()[1:] - \
                  hist.loc[s].loc[idx]['open'].to_numpy()[:-1]) \
                  / hist.loc[s].loc[idx]['open'].to_numpy()[:-1]) \
                for s in symbols \
                ])
        
        return prices

    def MarkovitzUpdateParams(self, prices):
        self.mu = np.mean(prices, axis=-1)
        self.sigma = np.cov(prices)
        # TODO: adjust quatntiles
        if 1.0 + self.TARGET_RETURN < np.min(self.mu):
            self.fixed_return = np.quantile(self.mu, 0.5)
        elif 1.0 + self.TARGET_RETURN > np.max(self.mu):
            self.fixed_return = np.quantile(self.mu, 0.9)
        else:
            self.fixed_return = 1.0 + self.TARGET_RETURN

    def MarkovitzPreprocess(self, prices, kind=None):
        if kind is None:
            pass # keep `prices` untouched
        elif kind == 'pca':
            prices = cov_matrix_preprocessing \
                .PCA_preprocessing(prices.T, **self.PREPROC_PARAMS["pca"])
            self.sigma = np.cov(prices)
        elif kind == 'to_norm_pca':
            prices = cov_matrix_preprocessing \
                .to_norm_PCA_preprocessing(prices.T, **self.PREPROC_PARAMS["to_norm_pca"])
            self.sigma = np.cov(prices)
        elif kind == 'mppca':
            prices = cov_matrix_preprocessing \
                .MPPCA_preprocessing(prices.T, **self.PREPROC_PARAMS["mppca"])
            self.sigma = np.cov(prices)
        else:
            raise ValueError('{} is not a valid price preprocessing kind'\
                    .format(str(kind)))
        
        return prices

    def MarkovitzReduceDimensions(self, kind=None):
        if kind is None:
            pass # keep `self.sigma` untouched
        elif kind == 'pca':
            pca = PCA(**self.DIMRED_PARAMS["pca"])
            pca.fit(self.sigma)
            self.sigma = pca.get_covariance()
        elif kind == 'kpca':
            kpca = KernelPCA(**self.DIMRED_PARAMS["kpca"])
            kpca.fit(self.sigma)
            # FIXME SKLEARN REQUIRES TO USE eigenvectors_ and eigenvalues_
            # INSTEAD OF alphas_ AND lambdas_ SINCE v1.0
            self.sigma = \
                    kpca.alphas_ @ \
                    np.diag(kpca.lambdas_) @ \
                    kpca.alphas_.T
        elif kind == 'mcd':
            mcd = MinCovDet(**self.DIMRED_PARAMS["mcd"])
            mcd.fit(self.sigma)
            self.sigma = mcd.covariance_
        else:
            raise ValueError('{} is not a valid dimension reduction kind'\
                    .format(str(kind)))
