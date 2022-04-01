import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA


def PCA_preprocessing(returns: np.ndarray, kept_components: int):
    scaler = preprocessing.StandardScaler().fit(returns)
    returns_scaled = scaler.transform(returns)

    pca = PCA(n_components=returns.shape[1])
    pca.fit(returns)
    transformed_top_returns = pca.transform(returns_scaled)

    transformed_top_returns[:,kept_components:] = 0
    simplified_returns = pca.inverse_transform(transformed_top_returns)

    return np.cov(simplified_returns.T)