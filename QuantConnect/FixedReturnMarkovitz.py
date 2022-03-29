import numpy as np
import cvxpy as cp
import functools

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.covariance import MinCovDet

# import custom_qc500

class FixedReturnMarkovitz(QCAlgorithm):

    def Initialize(self):
        self.HISTORY_DAYS = 360
        self.DIMRED_KIND = 'pca' # 'mcd'
        self.DIMRED_FRACTION = 0.5 # XXX
        self.RETURN_QUANTILE = 0.5 # XXX
        self.tickers = ['AAPL', 'GOOGL', 'IBM'] # custom_qc500.get_custom_qc500_tickers()
        self.symbols = [ \
                Symbol.Create(t, SecurityType.Equity, Market.USA) \
                for t in self.tickers \
                ]
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverseSelection(ManualUniverseSelectionModel(self.symbols))

        # Markowitz portfolio data
        # (W.T @ Sigma @ W - RiskTol * W.T @ Mu) -> min
        self.count = len(self.symbols)
        self.sigma = np.zeros((self.count, self.count)) # XXX ? np.eye(count)
        self.mu = np.ones(self.count)
        self.weights = np.divide(np.ones(self.count), self.count)

        # The following values have to be reevaluated every rebalance step
        self.dims = int(self.count * self.DIMRED_FRACTION)
        self.fixed_return = np.quantile(self.mu, self.RETURN_QUANTILE)

        self.SetStartDate(2006,1,1) # Before 2008
        self.SetEndDate(2014,1,1) # After 2008, before 2019
        self.SetCash(100000)  # Set Strategy Cash

        self.MarkowitzOnMonthStart()
        self.Schedule.On(self.DateRules.MonthStart(), \
                         self.TimeRules.At(0, 0), \
                         self.MarkowitzOnMonthStart)

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''

        self.MarkowitzOnData(data)
        # if not self.Portfolio.Invested:
        #     self.SetHoldings("SPY", 1)

    def MarkowitzRebalance(self):
        self.MarkowitzOptimize()
        self.SetHoldings( [ PortfolioTarget(s, w) \
                for s, w in zip(self.symbols, self.weights) ])

    def MarkowitzOptimize(self):
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

    def MarkowitzOnData(self, data):
        pass

    def MarkowitzOnMonthStart(self):
        self.MarkowitzUpdateParams(days=self.HISTORY_DAYS)
        self.MarkovitzReduceDimensions(kind=self.DIMRED_KIND)
        self.MarkowitzRebalance()

    def MarkowitzUpdateParams(self, days=None):
        days = days or self.HISTORY_DAYS
        hist = self.History(self.symbols, days, Resolution.Daily)
        idx = functools.reduce(np.intersect1d, \
                        (hist.loc[str(s.ID)].index \
                        for s in self.symbols) \
                        )
        prices = [ \
                ((hist.loc[str(s.ID)].loc[idx]['close'] - \
                  hist.loc[str(s.ID)].loc[idx]['open']) \
                  / hist.loc[str(s.ID)].loc[idx]['open']).to_numpy() \
                for s in self.symbols \
                ]
        self.mu = np.mean(prices, axis=-1)
        self.sigma = np.cov(prices)
        self.dims = int(self.count * self.DIMRED_FRACTION)
        self.fixed_return = np.quantile(self.mu, self.RETURN_QUANTILE)

        # XXX pass

    def MarkovitzReduceDimensions(self, kind=None):
        if self.dims == self.count or kind is None:
            pass # keep `self.sigma` untouched
        elif kind == 'pca':
            pca = PCA(n_components=self.dims)
            pca.fit(self.sigma)
            self.sigma = pca.get_covariance()
        elif kind == 'kpca':
            kpca = KernelPCA(n_components=self.dims, kernel='poly')
            kpca.fit(self.sigma)
            # FIXME SKLEARN REQUIRES TO USE eigenvectors_ and eigenvalues_
            # INSTEAD OF alphas_ AND lambdas_ SINCE v1.0
            self.sigma = \
                    kpca.alphas_ @ \
                    np.diag(kpca.lambdas_) @ \
                    kpca.alphas_.T
        elif kind == 'mcd':
            mcd = MinCovDet()
            mcd.fit(self.sigma)
            self.sigma = mcd.covariance_
        else:
            raise ValueError('{} is not a valid dimension reduction kind'\
                    .format(str(self.kind)))


