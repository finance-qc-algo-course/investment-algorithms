import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF as NMF_
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import  risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import objective_functions

from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

# import rpy2's package module
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('NMF', 'nsprcomp', 'BiocManager')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

nsprcomp = importr('nsprcomp')

# Allow conversion
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

class NDR(TransformerMixin, BaseEstimator):
    def __init__(self, is_NPCA=False, is_NMF=False, n_comp=None, window_size=None, 
                 n_comp_list=range(1, 6), window_size_list=range(2, 11), risk_free_return=0.01, 
                 metric='sortino', VAR_quantile=0.05, cv=TimeSeriesSplit(), iteration_num=1):
        '''
        init
        '''
        self.is_NPCA = is_NPCA
        self.is_NMF = is_NMF
        self.n_comp = n_comp
        self.window_size = window_size
        self.n_comp_list = n_comp_list
        self.window_size_list = window_size_list
        self.risk_free_return = risk_free_return
        self.metric = metric
        self.VAR_quantile = VAR_quantile
        self.cv = cv
        self.iteration_num = iteration_num
    
    
    def fit(self, X, y=None):
        """
        fit
        """
        if self.n_comp is None or self.window_size is None:
            if self.is_NPCA:
                self.n_comp, self.window_size = self._NPCA_param_search(X)
            
            if self.is_NMF:
                self.n_comp, self.window_size = self._NMF_param_search(X)
            
        return self
    
    
    def transform(self, X, y=None):
        """
        transform
        """
        return self._transform(X)
    
    
    def fit_transform(self, X, y=None):
        """
        fit_transform
        """
        return self.fit(X)._transform(X)
    
    
    def _transform(self, X):
        """
        _transform
        """
        if self.is_NPCA:
            return self._NPCA_dim_red(X, self.n_comp, self.window_size)
        
        if self.is_NMF:
            return self._NMF_dim_red(X, self.n_comp, self.window_size)
    
    
    def _NPCA_dim_red(self, X, n_comp, window_size):
        components_ind = 6
        new_prices = list()

        for i in range(0, len(X.index), window_size):
            var = nsprcomp.nsprcomp(X[i : i + window_size].T, ncomp=n_comp, center=False, scale=True, nneg=True)
            new_prices.append(self._SVP(var[components_ind].T))

        new_prices = np.vstack(new_prices)
        df_new_prices = pd.DataFrame(data=new_prices, index=np.arange(0, len(new_prices)), columns=X.columns)
        new_returns = df_new_prices.pct_change(1).fillna(0)
        new_returns = new_returns.drop(new_returns.index[0])

        return new_returns
    
    
    def _NMF_dim_red(self, X, n_comp, window_size):
        new_prices = list()
        heights = list()

        for i in range(0, len(X.index), window_size):
            model = NMF_(n_components=n_comp)
            W = model.fit_transform(X[i : i + window_size].T)
            H = model.components_
            heights.append(H)
            new_prices.append(self._SVP(W.T))

        new_prices = np.vstack(new_prices)
        df_new_prices = pd.DataFrame(data=new_prices, index=np.arange(0, len(new_prices)), columns=X.columns)
        new_returns = df_new_prices.pct_change(1).fillna(0)
        new_returns = new_returns.drop(new_returns.index[0])
        
        return new_returns
    
    
    def _SVP(self, components):
        r = len(components)
        s = np.zeros(r)   
        for i in range(r):
            s[i] = sum((components[i] - np.mean(components))**2) / (len(components[0]) - 1)

        c = np.zeros(r)
        for i in range(r):
            c[i] = s[i] / sum(s)

        ans = components.T @ c
        return ans
    
    
    def _count_metric(self, portfolio_return):
        N = 255
        
        if self.metric == 'sortino':
            pd_r = pd.DataFrame(portfolio_return)
            pd_r = pd_r.pct_change()
            mean = pd_r.mean() * N - self.risk_free_return
            std_neg = pd_r[pd_r < 0].std() * np.sqrt(N)
            return (mean / std_neg)[pd_r.columns[0]]
        
        if self.metric == 'sharpe':
            pd_r = pd.DataFrame(portfolio_return)
            pd_r = pd_r.pct_change()
            mean = pd_r.mean() * N - self.risk_free_return
            std_neg = pd_r.std() * np.sqrt(N)
            return (mean / std_neg)[pd_r.columns[0]]
        
        if self.metric == 'VAR':
            return np.quantile(portfolio_return, self.VAR_quantile)
    
    
    def _return_weights(self, X):
        mean = expected_returns.mean_historical_return(X, returns_data=True)
        S = risk_models.sample_cov(X, returns_data=True)
        ef = EfficientFrontier(mean, S)
        weights = ef.max_sharpe()
        return weights
    
    
    def _count_portfolio_return(self, weights, test):
        portfolio_return = 0
        for col in test.columns:
            one_company_return = weights[col] * (test[col] + 1).cumprod()
            portfolio_return += one_company_return
        return portfolio_return
        
        
    def _NPCA_param_search(self, prices_df):
        best_ratio = -1
        best_params = [0, 0]
        best_ratio_list = list()
        best_weights = list()

        for n_comp in tqdm(self.n_comp_list):
            for window_size in tqdm(self.window_size_list):
                if n_comp > window_size:
                        continue

                ratio_list = list()
                weights_list = list()

                for i in range(self.iteration_num):               
                    for train_index, test_index in self.cv.split(prices_df):
                        train = prices.loc[prices.index[train_index]]
                        test = prices.loc[prices.index[test_index]]
                        test_returns = test.pct_change(1).fillna(0)

                        try:
                            returns = self._NPCA_dim_red(train, n_comp, window_size)
                            npca_w = self._return_weights(returns)
                            weights_list.append(npca_w)

                            npca_portfolio_return = self._count_portfolio_return(npca_w, test_returns)
                            ratio_list.append(self._count_metric(npca_portfolio_return))
                        except Exception as e:
                            print('Error message: ', e)
                            continue
                
                if len(ratio_list) == 0:
                    continue

                cur_ratio = np.mean(ratio_list)
                if cur_ratio > best_ratio:
                    best_ratio = cur_ratio
                    best_params[0] = n_comp
                    best_params[1] = window_size
                    best_ratio_list = ratio_list
                    best_weights = weights_list

        return best_params
    
    
    def _NMF_param_search(self, prices_df):
        best_ratio = -1
        best_params = [0, 0]
        best_ratio_list = list()
        best_weights = list()

        for n_comp in tqdm(self.n_comp_list):
            for window_size in tqdm(self.window_size_list):
                if n_comp > window_size:
                        continue

                ratio_list = list()
                weights_list = list()

                for i in range(self.iteration_num):               
                    for train_index, test_index in self.cv.split(prices_df):
                        train = prices.loc[prices.index[train_index]]
                        test = prices.loc[prices.index[test_index]]
                        test_returns = test.apply(lambda x: x.pct_change(1).fillna(0), axis=0)

                        try:
                            returns = self._NMF_dim_red(train, n_comp, window_size)
                            nmf_w = self._return_weights(returns)
                            weights_list.append(nmf_w)

                            nmf_portfolio_return = self._count_portfolio_return(nmf_w, test_returns)
                            ratio_list.append(self._count_metric(nmf_portfolio_return))
                        except Exception as e:
                            print('Error message: ', e)
                            continue

                if len(ratio_list) == 0:
                    continue
                
                cur_ratio = np.mean(ratio_list)
                if cur_ratio > best_ratio:
                    best_ratio = cur_ratio
                    best_params[0] = n_comp
                    best_params[1] = window_size
                    best_ratio_list = ratio_list
                    best_weights = weights_list

        return best_params