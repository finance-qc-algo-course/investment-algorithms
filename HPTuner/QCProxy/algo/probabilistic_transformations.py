import scipy.stats as sps
import numpy as np


def laplace_to_norm(data: np.ndarray):
    data_norm = data.copy()
    for col in range(data.shape[1]):
        loc, scale = sps.laplace.fit(data[:, col])
        if scale == 0:
            continue
        eps = 0.000001
        data_norm[:, col] = np.clip(sps.laplace(loc, scale).cdf(data[:, col]), eps, 1 - eps)
        data_norm[:, col] = sps.norm().ppf(data_norm[:, col])

    return data_norm