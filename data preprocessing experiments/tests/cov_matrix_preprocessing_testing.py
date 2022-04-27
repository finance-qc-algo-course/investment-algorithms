import unittest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '..')

from mppca import MPPCA
from probabilistic_transformations import laplace_to_norm
import cov_matrix_preprocessing


def is_pos_def(A):
    if np.array_equal(A, A.T):
        EPS = 1e-6
        if np.all(np.linalg.eigvals(A) > -EPS):
            return True
    return False

def read_data():
    returns = pd.read_csv('../work/data/returns.csv')
    returns = returns.fillna(0)
    returns['date'] = returns['date'].astype(np.datetime64)
    returns = returns.set_index('date')
    return returns

class TestMPPCA(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.data = read_data()
    
    def test_fit(self):
        latent_dim = 5
        n_models = 4

        mppca = MPPCA(latent_dim, n_models)
        mppca.fit(self.data.to_numpy())
        self.assertIsInstance(mppca.mu, np.ndarray)
        self.assertIsInstance(mppca.pi, np.ndarray)
        self.assertIsInstance(mppca.sigma2, np.ndarray)
        self.assertIsInstance(mppca.W, np.ndarray)

    def test_transform(self):
        for n_cols in range(10, 100):
            mean = self.data[self.data.columns].mean()
            top_returns = self.data[mean.sort_values(ascending=False)[:n_cols].index]

            np.seterr(all='raise', under='ignore')

            mppca = MPPCA(5, 4)
            mppca.fit(top_returns.to_numpy())
            mppca_top_returns = mppca.transform(top_returns.to_numpy())

            message = 'MSE is greater than expected'
            self.assertLess(np.mean(mppca_top_returns - top_returns.to_numpy()**2), 3e-3, message)

class TestProbabilisticTransformations(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.data = read_data()

    def test_laplace_to_norm_returns(self):
        normalized_data = laplace_to_norm(self.data.to_numpy())
        self.assertFalse(np.isnan(normalized_data).any())

    def test_laplace_to_norm_equal_data(self):
        data_len = 100
        data = np.zeros(data_len).reshape(1, data_len)
        normalized_data = laplace_to_norm(self.data.to_numpy())
        self.assertFalse(np.isnan(normalized_data).any())

class TestPreprocessingMethods(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.data = read_data()

    def test_PCA(self):
        kept_components = 10
        Sigma = cov_matrix_preprocessing.PCA_preprocessing(
            self.data.to_numpy(), kept_components)
        self.assertFalse(np.isnan(Sigma).any())
        self.assertTrue(is_pos_def(Sigma))
    
    def test_to_norm_PCA(self):
        kept_components = 10
        Sigma = cov_matrix_preprocessing.to_norm_PCA_preprocessing(
            self.data.to_numpy(), kept_components)
        self.assertFalse(np.isnan(Sigma).any())
        self.assertTrue(is_pos_def(Sigma))

    def test_MPPCA(self):
        kept_components = 10
        n_models = 3
        Sigma = cov_matrix_preprocessing.MPPCA_preprocessing(
            self.data.to_numpy(), kept_components, n_models)
        self.assertFalse(np.isnan(Sigma).any())
        self.assertTrue(is_pos_def(Sigma))

    def test_to_norm_MPPCA(self):
        kept_components = 10
        n_models = 3
        Sigma = cov_matrix_preprocessing.to_norm_MPPCA_preprocessing(
            self.data.to_numpy(), kept_components, n_models)
        self.assertFalse(np.isnan(Sigma).any())
        self.assertTrue(is_pos_def(Sigma))

if __name__ == '__main__':
    unittest.main()
