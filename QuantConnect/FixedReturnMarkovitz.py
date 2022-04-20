import numpy as np
import cvxpy as cp
import functools

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.covariance import MinCovDet

import custom_tickers
import cov_matrix_preprocessing

class FixedReturnMarkovitz(QCAlgorithm):

    def Initialize(self):
        self.WINDOW_SIZE = 735
        self.REBALANCE_PERIOD = 900
        self.TOP_COUNT = 15 # in bitcoin_lgb..=46
        self.TARGET_RETURN = 0.0028
        self.PREPROC_KIND = 'mppca' # {None, 'pca', 'to_norm_pca', 'mppca'}
        self.PREPROC_DIMS = 14
        self.PREPROC_PARAMS = {
            "pca": {
                "kept_components": self.PREPROC_DIMS,
            },
            "to_norm_pca": {
                "kept_components": self.PREPROC_DIMS,
            },
            "mppca": {
                "kept_components": self.PREPROC_DIMS,
                "n_models": 2,
            },
        }
        self.DIMRED_KIND = None # {None, 'pca', 'kpca', 'mcd'}
        self.DIMRED_DIMS = None
        self.DIMRED_PARAMS = {
            "pca": {
                "n_components": self.DIMRED_DIMS,
            },
            "kpca": {
                "n_components": self.DIMRED_DIMS,
                "kernel": "poly",
            },
            "mcd": {},
        }

        self.tickers = custom_tickers.get_custom_top_tickers(self.TOP_COUNT) # ['AAPL', 'GOOGL', 'IBM']
        self.symbols = [ \
                Symbol.Create(t, SecurityType.Equity, Market.USA) \
                for t in self.tickers \
                ]
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverseSelection(ManualUniverseSelectionModel(self.symbols))

        # Markovitz portfolio data
        # (W.T @ Sigma @ W - RiskTol * W.T @ Mu) -> min
        self.count = len(self.symbols)
        self.sigma = np.eye(self.count)
        self.mu = np.ones(self.count)
        self.weights = np.divide(np.ones(self.count), self.count)

        # The following values have to be reevaluated every rebalance step
        self.fixed_return = 1.0 + self.TARGET_RETURN

        # The following values are internal to the algo logic

        self.SetStartDate(2006,1,1) # Before 2008
        self.SetEndDate(2014,1,1) # After 2008, before 2019
        self.SetCash(100000)  # Set Strategy Cash

        self.days_from_start = 0
        self.Schedule.On(self.DateRules.EveryDay(), \
                         self.TimeRules.At(0, 0), \
                         self.MarkovitzOnEveryDay)

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''

        self.MarkovitzOnData(data)

    def MarkovitzOnData(self, data):
        if not self.Portfolio.Invested:
            self.MarkovitzRebalance(self.MarkovitzGetPrices(days=self.WINDOW_SIZE))

    def MarkovitzOnEveryDay(self):
        if self.days_from_start % self.REBALANCE_PERIOD == 0:
            prices = self.MarkovitzGetPrices(days=self.WINDOW_SIZE)
            self.MarkovitzRebalance(prices)

        self.days_from_start += 1

    def MarkovitzRebalance(self, prices):
        prices = self.MarkovitzPreprocess(prices, kind=self.PREPROC_KIND)
        self.MarkovitzUpdateParams(prices)
        self.MarkovitzReduceDimensions(kind=self.DIMRED_KIND)
        self.MarkovitzOptimize()
        self.SetHoldings( [ PortfolioTarget(s, w) \
                for s, w in zip(self.symbols, self.weights) ])

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
        hist = self.History(self.symbols, days, Resolution.Daily)
        idx = functools.reduce(np.intersect1d, \
                        (hist.loc[str(s.ID)].index \
                        for s in self.symbols) \
                        )

        prices = np.array([ \
                ((hist.loc[str(s.ID)].loc[idx]['open'].to_numpy()[1:] - \
                  hist.loc[str(s.ID)].loc[idx]['open'].to_numpy()[:-1]) \
                  / hist.loc[str(s.ID)].loc[idx]['open'].to_numpy()[:-1]) \
                for s in self.symbols \
                ])

        # XXX: OTHER METHOD (BETTER?)
        """
        prices = np.array([ \
                ((hist.loc[str(s.ID)].loc[idx]['close'] - \
                  hist.loc[str(s.ID)].loc[idx]['open']) \
                  / hist.loc[str(s.ID)].loc[idx]['open']).to_numpy() \
                for s in self.symbols \
                ])
        """

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
        elif kind == 'to_norm_pca':
            prices = cov_matrix_preprocessing \
                .to_norm_PCA_preprocessing(prices.T, **self.PREPROC_PARAMS["to_norm_pca"])
        elif kind == 'mppca':
            prices = cov_matrix_preprocessing \
                .MPPCA_preprocessing(prices.T, **self.PREPROC_PARAMS["mppca"])
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

