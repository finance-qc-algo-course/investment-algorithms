import numpy as np
import cvxpy as cp
import functools

class StaticMarkovitz(QCAlgorithm):

    def Initialize(self):
        self.HISTORY_DAYS = 360
        self.tickers = [ "AAPL", "GOOGL", "IBM" ]
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
        self.mu = np.zeros(self.count)
        self.risk_tol = 0.1
        self.weights = np.divide(np.ones(self.count), self.count)

        self.SetStartDate(2006,1,1) # Before 2008
        self.SetEndDate(2014,1,1) # After 2008, before 2019
        self.SetCash(100000)  # Set Strategy Cash

        self.MarkowitzUpdateParams()
        self.MarkowitzRebalance()
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
        E = np.ones(self.count)
        w = cp.Variable(self.count)
        prob = cp.Problem( \
                cp.Minimize( \
                        0.5 * cp.quad_form(w, self.sigma) - \
                        self.risk_tol * w.T @ self.mu \
                    ), \
                    [ \
                        w.T >= 0.0, \
                        E.T @ w == 1.0 \
                    ]
                )

        prob.solve()
        self.weights = w.value
        pass

    def MarkowitzOnData(self, data):
        pass

    def MarkowitzOnMonthStart(self):
        self.MarkowitzUpdateParams(days=self.HISTORY_DAYS)
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

        # XXX pass

