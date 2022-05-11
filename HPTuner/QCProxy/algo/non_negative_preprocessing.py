from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages
import pandas as pd
import numpy as np
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF as NMF_

from rpy2.robjects.packages import importr
base = importr('base')
utils = importr('utils')

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

# R package names
packnames = ('NMF', 'nsprcomp', 'BiocManager')

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# package for NPCA
nsprcomp = importr('nsprcomp')
# Allow conversion
pandas2ri.activate()


def _NPCA_dim_red(X, n_comp, window_size):
    if n_comp > window_size:
        return X

    try:
        components_ind = 6
        new_prices = list()

        for i in range(0, X.shape[0], window_size):
            obj = nsprcomp.nsprcomp(
                X[i: i + window_size].T, ncomp=n_comp, center=False, scale=False, nneg=True)
            new_prices.append(_SVP(obj[components_ind].T))

        new_prices = np.vstack(new_prices)

        return new_prices
    except Exception as e:
        return X


def _NMF_dim_red(X, n_comp, window_size):
    if n_comp > window_size:
        return X

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            new_prices = list()

            for i in range(0, X.shape[0], window_size):
                model = NMF_(n_components=n_comp)
                W = model.fit_transform(X[i: i + window_size].T)
                new_prices.append(_SVP(W.T))

            new_prices = np.vstack(new_prices)

            return new_prices
    except Exception as e:
        return X


def _SVP(components):
    r = len(components)
    s = np.zeros(r)
    for i in range(r):
        s[i] = sum((components[i] - np.mean(components))**2) / \
            (len(components[0]) - 1)
    c = np.zeros(r)
    for i in range(r):
        c[i] = s[i] / sum(s)
    ans = components.T @ c
    return ans
