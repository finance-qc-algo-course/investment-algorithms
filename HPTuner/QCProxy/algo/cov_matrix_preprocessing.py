import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.stats as sps

from .mppca import MPPCA
from .probabilistic_transformations import laplace_to_norm


def PCA_preprocessing(returns: np.ndarray, kept_components: int):
    scaler = preprocessing.StandardScaler().fit(returns)
    returns_scaled = scaler.transform(returns)

    pca = PCA(n_components=returns.shape[1])
    pca.fit(returns_scaled)
    transformed_top_returns = pca.transform(returns_scaled)

    transformed_top_returns[:,kept_components:] = 0
    simplified_returns = pca.inverse_transform(transformed_top_returns)
    
    transformed = scaler.inverse_transform(simplified_returns)

    return np.cov(transformed.T)

def to_norm_PCA_preprocessing(returns: np.ndarray, kept_components: int):
    returns_norm = laplace_to_norm(returns)
    return PCA_preprocessing(returns_norm, kept_components)

def MPPCA_preprocessing(returns: np.ndarray, kept_components: int, n_models: int):
    scaler = preprocessing.StandardScaler().fit(returns)
    returns_scaled = scaler.transform(returns)

    mppca = MPPCA(kept_components, n_models)
    mppca.fit(returns_scaled)
    mppca_returns = mppca.transform(returns_scaled)
    
    transformed = scaler.inverse_transform(mppca_returns)

    return np.cov(transformed.T)

def to_norm_MPPCA_preprocessing(returns: np.ndarray, kept_components: int, n_models: int):
    returns_norm = laplace_to_norm(returns)
    return MPPCA_preprocessing(returns_norm, kept_components, n_models)
    