import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.stats as sps

from .mppca import MPPCA

def PCA_preprocessing(returns: np.ndarray, kept_components: int):
    scaler = preprocessing.StandardScaler().fit(returns)
    returns_scaled = scaler.transform(returns)

    pca = PCA(n_components=returns.shape[1])
    pca.fit(returns)
    transformed_top_returns = pca.transform(returns_scaled)

    transformed_top_returns[:,kept_components:] = 0
    simplified_returns = pca.inverse_transform(transformed_top_returns)

    return simplified_returns.T

def to_norm_PCA_preprocessing(returns: np.ndarray, kept_components: int):
    returns_norm = returns.copy()
    for col in range(returns.shape[1]):
        loc, scale = sps.laplace.fit(returns[:, col])
        eps = 0.000001
        returns_norm[:, col] = np.clip(sps.laplace(loc, scale).cdf(returns[:, col]), 0.000001, 1 - eps)
        returns_norm[:, col] = sps.norm().ppf(returns_norm[:, col])
    
    return PCA_preprocessing(returns_norm, kept_components)

def MPPCA_preprocessing(returns: np.ndarray, kept_components: int, n_models: int):
    scaler = preprocessing.StandardScaler().fit(returns)
    returns_scaled = scaler.transform(returns)

    mppca = MPPCA(kept_components, n_models)
    mppca.fit(returns)
    mppca_returns = mppca.transform(returns_scaled)

    return mppca_returns.T
