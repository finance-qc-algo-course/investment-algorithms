from ..Interface import Interface

from . import custom_tickers
from . import cov_matrix_preprocessing
from . import non_negative_preprocessing

import numpy as np
import cvxpy as cp
import functools
import datetime as dttm
import warnings
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.covariance import MinCovDet

from typing import List

ALGO_HYPERPARAMS = {
    "WINDOW_SIZE": 735,
    "REBALANCE_PERIOD": 900,
    "TOP_COUNT": 15, # in 1..=46
    "TARGET_RETURN": 0.0028,
    "TARGET_QUANTILE": 1,
    "NPREPROC_KIND": 'npca', # {None, 'npca', 'nmf'}
    "NPREPROC_DIMS": 2,
    "NPREPROC_FACTOR": 5,
    "NPREPROC_PARAMS": {
        "npca": {
            "n_comp": 10, # self.NPREPROC_DIMS
            "window_size": 30, # self.NPREPROC_FACTOR
        },
        "nmf": {
            "n_comp": 2, # self.NPREPROC_DIMS
            "window_size": 5, # self.NPREPROC_FACTOR
        },
    },
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
    "DIMRED_KIND": 'pca', # {None, 'pca', 'kpca', 'mcd'}
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
    "THRESHOLD": 1e-5,
}

ALGO_TICKERS = custom_tickers.get_custom_top_tickers(ALGO_HYPERPARAMS["TOP_COUNT"])
ALGO_START_DATE = dttm.date(2006, 1, 1)
ALGO_END_DATE = dttm.date(2014, 1, 1)
ALGO_LOOKBACK = ALGO_HYPERPARAMS["WINDOW_SIZE"]
ALGO_CASH = 100000

class Algorithm(Interface):
    def __init__(self, portfolio_manager, history_manager, event_manager, \
                 cash: int, tickers: List[str], \
                 start_date: dttm.date, end_date: dttm.date, lookback: int, \
                 hyperparams=ALGO_HYPERPARAMS):
        super().__init__( \
                portfolio_manager, history_manager, event_manager, \
                cash, tickers, start_date, end_date, lookback)
        self.InitializeHyperparams(hyperparams)
        self.InitializeAlgorithm()

    def InitializeHyperparams(self, hyperparams):
        self.WINDOW_SIZE = hyperparams["WINDOW_SIZE"]
        self.REBALANCE_PERIOD = hyperparams["REBALANCE_PERIOD"]
        self.TOP_COUNT = hyperparams["TOP_COUNT"] # in 1..=36
        self.TARGET_RETURN = hyperparams["TARGET_RETURN"]
        self.TARGET_QUANTILE = hyperparams["TARGET_QUANTILE"]
        # {None, 'npca', 'nmf'}
        self.NPREPROC_KIND = hyperparams["NPREPROC_KIND"]
        self.NPREPROC_DIMS = hyperparams["NPREPROC_DIMS"]
        self.NPREPROC_FACTOR = hyperparams["NPREPROC_FACTOR"]
        self.NPREPROC_PARAMS = hyperparams["NPREPROC_PARAMS"]
        # {None, 'pca', 'to_norm_pca', 'mppca'}
        self.PREPROC_KIND = hyperparams["PREPROC_KIND"]
        self.PREPROC_DIMS = hyperparams["PREPROC_DIMS"]
        self.PREPROC_PARAMS = hyperparams["PREPROC_PARAMS"]
        # {None, 'pca', 'kpca', 'mcd'}
        self.DIMRED_KIND = hyperparams["DIMRED_KIND"]
        self.DIMRED_DIMS = hyperparams["DIMRED_DIMS"]
        self.DIMRED_PARAMS = hyperparams["DIMRED_PARAMS"]
        self.THRESHOLD = hyperparams["THRESHOLD"]

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

    def RegularizeCovarianceMatrix(self, EPS):
        self.sigma = self.sigma + np.eye(self.sigma.shape[0]) * EPS

    def MarkovitzRebalance(self, prices):
        prices = self.MarkovitzNPreprocess(prices.T, kind=self.NPREPROC_KIND)
        returns = self.MarkovitzGetReturns(prices.T).T
        self.MarkovitzUpdateParams(returns)
        self.MarkovitzPreprocess(returns, kind=self.PREPROC_KIND)
        self.MarkovitzReduceDimensions(returns, kind=self.DIMRED_KIND)
        self.SetFixedReturn()
        self.RegularizeCovarianceMatrix(1e-8)
        self.MarkovitzOptimize(self.THRESHOLD)
        self.TransformToTickers()
        self.SetWeights(self.weights)

    def TransformToTickers(self):
        if self.DIMRED_KIND == 'pca' or self.DIMRED_KIND == 'kpca':
            self.weights = self.components_.to_numpy().T.dot(self.weights)

    def MarkovitzOptimize(self, threshold):
        # XXX sum_weight = np.sum(self.weights)
        # XXX self.weights = np.divide(self.weights, sum_weight)
        w = cp.Variable(self.sigma.shape[0])
        objective = cp.Minimize(cp.quad_form(w, self.sigma)) # * 0.5
        constraints = [ \
                w.T >= 0.0, \
                w.T @ np.ones(self.sigma.shape[0]) == 1.0, \
                w.T @ self.mu == self.fixed_return, \
            ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver='SCS')
        except:
            warnings.warn('''SolverError: solver can't solve this task. 
                Trying to solve with another solver''')
            prob.solve(solver='CVXOPT')

        self.weights = w.value
        self.weights[self.weights < threshold] = 0
        self.weights = self.weights / np.sum(self.weights)

    def MarkovitzGetPrices(self, days=None):
        days = days or self.WINDOW_SIZE
        hist = self.GetHistory(self.GetTickers(), days)
        symbols = np.unique(hist.index.get_level_values(0).to_numpy())
        idx = functools.reduce(np.intersect1d, \
                        (hist.loc[s].index for s in symbols))

        prices = np.array([ \
                hist.loc[s].loc[idx]['open'].to_numpy() \
                for s in symbols \
            ])
        return prices

    def MarkovitzGetReturns(self, prices):
        returns = np.divide(prices[:,1:] - prices[:,:-1], prices[:,:-1])
        return returns

    def SetFixedReturn(self):
        # TODO: adjust quantiles
        if self.TARGET_QUANTILE is not None:
            self.fixed_return = np.quantile(self.mu, self.TARGET_QUANTILE)
            return
        if 1.0 + self.TARGET_RETURN < np.min(self.mu):
            self.fixed_return = np.quantile(self.mu, 0.5)
        elif 1.0 + self.TARGET_RETURN > np.max(self.mu):
            self.fixed_return = np.quantile(self.mu, 0.9)
        else:
            self.fixed_return = 1.0 + self.TARGET_RETURN

    def MarkovitzNPreprocess(self, prices, kind=None):
        if kind is None:
            pass # keep `prices` untouched
        elif kind == 'npca':
            prices = non_negative_preprocessing \
                ._NPCA_dim_red(prices, **self.NPREPROC_PARAMS["npca"])
        elif kind == 'nmf':
            prices = non_negative_preprocessing \
                ._NMF_dim_red(prices, **self.NPREPROC_PARAMS["nmf"])
        else:
            raise ValueError('{} is not a valid nonnegative preprocessing kind'\
                    .format(str(kind)))

        return prices

    def MarkovitzUpdateParams(self, prices):
        self.mu = np.mean(prices.T, axis=-1)
        self.sigma = np.cov(prices.T)

    def MarkovitzPreprocess(self, prices, kind=None):
        if kind is None:
            pass # keep `prices` untouched
        elif kind == 'pca':
            self.sigma = cov_matrix_preprocessing \
                .PCA_preprocessing(prices, **self.PREPROC_PARAMS["pca"])
        elif kind == 'to_norm_pca':
            self.sigma = cov_matrix_preprocessing \
                .to_norm_PCA_preprocessing(prices, **self.PREPROC_PARAMS["to_norm_pca"])
        elif kind == 'mppca':
            self.sigma = cov_matrix_preprocessing \
                .MPPCA_preprocessing(prices, **self.PREPROC_PARAMS["mppca"])
        elif kind == 'to_norm_mppca':
            self.sigma = cov_matrix_preprocessing \
                .MPPCA_preprocessing(prices, **self.PREPROC_PARAMS["to_norm_mppca"])
        else:
            raise ValueError('{} is not a valid preprocessing kind'\
                    .format(str(kind)))

    def MarkovitzReduceDimensions(self, prices, kind=None):
        if kind is None:
            pass # keep `self.sigma` untouched
        elif kind == 'pca' or kind == 'kpca':
            if kind == 'pca':
                if self.DIMRED_PARAMS["pca"]['n_components'] >= self.sigma.shape[0]:
                    self.DIMRED_PARAMS["pca"]['n_components'] = self.sigma.shape[0] - 1
                pca = PCA(**self.DIMRED_PARAMS["pca"])
                pca.fit(self.sigma)
                self.components_ = pca.components_
            else:
                if self.DIMRED_PARAMS["kpca"]['n_components'] >= self.sigma.shape[0]:
                    self.DIMRED_PARAMS["kpca"]['n_components'] = self.sigma.shape[0] - 1
                kpca = KernelPCA(**self.DIMRED_PARAMS["kpca"])
                kpca.fit(self.sigma)
                self.components_ = kpca.eigenvectors_.T
            self.components_ = pd.DataFrame(self.components_)
            # remove components less that zero
            self.components_[self.components_ < 0] = 0
            # rebalance componets, so their sum is 1 now 
            self.components_ = self.components_.div(self.components_.sum(axis=1), axis=0)
            self.components_.fillna(1 / len(self.components_.columns.to_numpy()), inplace=True)
            # data of returns by components
            new_data = prices @ self.components_.to_numpy().T
            optimized_data = pd.DataFrame(data=new_data, columns=np.linspace(1, new_data.shape[1], new_data.shape[1]))
            # change mu and sigma

            self.mu = optimized_data[optimized_data.columns].mean().to_numpy()
            self.sigma = optimized_data[optimized_data.columns].cov()
        elif kind == 'mcd':
            mcd = MinCovDet(**self.DIMRED_PARAMS["mcd"])
            mcd.fit(prices)
            self.sigma = mcd.covariance_
        else:
            raise ValueError('{} is not+ a valid dimension reduction kind'\
                    .format(str(kind)))
