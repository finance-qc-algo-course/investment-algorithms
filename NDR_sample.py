import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF as NMF_

from rpy2.robjects.packages import importr
base = importr('base')
utils = importr('utils')

import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('NMF', 'nsprcomp', 'BiocManager')

from rpy2.robjects.vectors import StrVector
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# package for NPCA
nsprcomp = importr('nsprcomp')
# Allow conversion
from rpy2.robjects import pandas2ri
pandas2ri.activate()

class NDR(TransformerMixin, BaseEstimator):
    def __init__(self, is_NPCA=False, is_NMF=False, n_comp=None, window_size=None):
        '''
        init
        '''
        self.is_NPCA = is_NPCA
        self.is_NMF = is_NMF
        self.n_comp = n_comp
        self.window_size = window_size
    
    
    def fit(self, X, y=None):
        """
        fit
        """
        # to do
        if self.is_NPCA:
            self._NPCA_dim_red(X, self.n_comp, self.window_size)
        
        if self.is_NMF:
            self._NMF_dim_red(X, self.n_comp, self.window_size)
            
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
    
    
    def _transform(self, X=None):
        """
        _transform
        """
        # to do
        if self.is_NPCA:
            return self.npca_returns
        
        if self.is_NMF:
            return self.nmf_X
        
    
    def score(self, X, y=None):
        #to do
        if self.is_NPCA:
            return 0
        
        if self.is_NMF:
            return -np.mean(self.norm_list)
    
    
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

        self.npca_returns = new_returns
        self.npca_prices = new_prices

        return new_returns
    
    
    def _NMF_dim_red(self, X, n_comp, window_size):
        new_prices = list()
        norm_list = list()

        for i in range(0, len(X.index), window_size):
            model = NMF_(n_components=n_comp)
            W = model.fit_transform(X[i : i + window_size].T)
            H = model.components_
            new_prices.append(self._SVP(W.T))
            norm_list.append(np.linalg.norm(W @ H - X[i : i + window_size].T))
        
        new_prices = np.vstack(new_prices)
        df_new_prices = pd.DataFrame(data=new_prices, index=np.arange(0, len(new_prices)), columns=X.columns)
        new_returns = df_new_prices.pct_change(1).fillna(0)
        new_returns = new_returns.drop(new_returns.index[0])
        
        self.norm_list = norm_list
        self.nmf_returns = new_returns
        self.nmf_prices = new_prices
        
        return new_returns, self.norm_list
    
    
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