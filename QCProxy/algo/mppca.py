import numpy as np


class MPPCA():
    def __init__(self, latent_dim: int, n_models: int) -> None:
        self.latent_dim = latent_dim
        self.n_models = n_models
    
    def fit(self, data: np.array, n_iter=100) -> None:
        self.data_dim = data.shape[1]

        self.W = np.random.rand(self.n_models, self.data_dim, self.latent_dim)
        self.mu = np.random.rand(self.n_models, self.data_dim)
        self.sigma2 = np.random.rand(self.n_models) + 0.01
        self.pi = np.random.rand(self.n_models) + 0.01

        self.pi, self.mu, self.W, self.sigma2, R, L, sigma2hist = \
            self._mppca_gem(data, self.pi, self.mu, self.W, self.sigma2, n_iter)

    def _mppca_gem(self, X, pi, mu, W, sigma2, niter):
        # taken from https://github.com/michelbl/MPPCA

        N, d = X.shape
        p = len(sigma2)
        _, q = W[0].shape

        sigma2hist = np.zeros((p, niter))
        M = np.zeros((p, q, q))
        Minv = np.zeros((p, q, q))
        Cinv = np.zeros((p, d, d))
        logR = np.zeros((N, p))
        R = np.zeros((N, p))
        M[:] = 0.
        Minv[:] = 0.
        Cinv[:] = 0.

        L = np.zeros(niter)
        for i in range(niter):
            # print('.', end='')
            for c in range(p):
                sigma2hist[c, i] = sigma2[c]

                # M
                M[c, :, :] = sigma2[c]*np.eye(q) + np.dot(W[c, :, :].T, W[c, :, :])
                Minv[c, :, :] = np.linalg.inv(M[c, :, :])

                # Cinv
                Cinv[c, :, :] = (np.eye(d)
                    - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
                    ) / sigma2[c]

                # R_ni
                deviation_from_center = X - mu[c, :]
                logR[:, c] = ( np.log(pi[c])
                    + 0.5*np.log(
                        np.linalg.det(
                            np.eye(d) - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
                        )
                    )
                    - 0.5*d*np.log(sigma2[c])
                    - 0.5*(deviation_from_center * np.dot(deviation_from_center, Cinv[c, :, :].T)).sum(1)
                    )

            myMax = logR.max(axis=1).reshape((N, 1))
            L[i] = (
                (myMax.ravel() + np.log(np.exp(logR - myMax).sum(axis=1))).sum(axis=0)
                - N*d*np.log(2*3.141593)/2.
                )

            logR = logR - myMax - np.reshape(np.log(np.exp(logR - myMax).sum(axis=1)), (N, 1))

            myMax = logR.max(axis=0)
            logpi = myMax + np.log(np.exp(logR - myMax).sum(axis=0)) - np.log(N)
            logpi = logpi.T
            pi = np.exp(logpi)
            R = np.exp(logR)
            for c in range(p):
                mu[c, :] = (R[:, c].reshape((N, 1)) * X).sum(axis=0) / R[:, c].sum()
                deviation_from_center = X - mu[c, :].reshape((1, d))

                SW = ( (1/(pi[c]*N))
                    * np.dot((R[:, c].reshape((N, 1)) * deviation_from_center).T,
                        np.dot(deviation_from_center, W[c, :, :]))
                    )

                Wnew = np.dot(SW, np.linalg.inv(sigma2[c]*np.eye(q) + np.dot(np.dot(Minv[c, :, :], W[c, :, :].T), SW)))

                sigma2[c] = (1/d) * (
                    (R[:, c].reshape(N, 1) * np.power(deviation_from_center, 2)).sum()
                    /
                    (N*pi[c])
                    -
                    np.trace(np.dot(np.dot(SW, Minv[c, :, :]), Wnew.T))
                    )

                W[c, :, :] = Wnew

        return pi, mu, W, sigma2, R, L, sigma2hist

    def transform(self, data: np.array):
        errors = np.zeros((self.n_models, data.shape[0]))
        transformed = np.zeros((self.n_models, data.shape[0], data.shape[1]))
        for i in range(self.n_models):
            transformed[i, :, :] = self._single_transform(data, i)
            errors[i] = np.mean((data - transformed[i])**2, axis=1)
        
        best_components = np.argmin(errors.T, axis=1)

        mppca_top_returns = np.zeros(data.shape)
        for i in range(data.shape[0]):
            mppca_top_returns[i, :] = transformed[best_components[i], i, :]

        return mppca_top_returns

    def _single_transform(self, data: np.array, model_index: int):
        mu = self.mu[model_index].reshape((1, self.mu.shape[1])) 
        W = self.W[model_index]

        return (data - mu) @ W @ (np.linalg.inv(W.T @ W)) @ W.T + mu


##########################################################################


def mppca_gem(X, pi, mu, W, sigma2, niter):
    N, d = X.shape
    p = len(sigma2)
    _, q = W[0].shape

    sigma2hist = np.zeros((p, niter))
    M = np.zeros((p, q, q))
    Minv = np.zeros((p, q, q))
    Cinv = np.zeros((p, d, d))
    logR = np.zeros((N, p))
    R = np.zeros((N, p))
    M[:] = 0.
    Minv[:] = 0.
    Cinv[:] = 0.

    L = np.zeros(niter)
    for i in range(niter):
        print('.', end='')
        for c in range(p):
            sigma2hist[c, i] = sigma2[c]

            # M
            M[c, :, :] = sigma2[c]*np.eye(q) + np.dot(W[c, :, :].T, W[c, :, :])
            Minv[c, :, :] = np.linalg.inv(M[c, :, :])

            # Cinv
            Cinv[c, :, :] = (np.eye(d)
                - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
                ) / sigma2[c]

            # R_ni
            deviation_from_center = X - mu[c, :]
            logR[:, c] = ( np.log(pi[c])
                + 0.5*np.log(
                    np.linalg.det(
                        np.eye(d) - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
                    )
                )
                - 0.5*d*np.log(sigma2[c])
                - 0.5*(deviation_from_center * np.dot(deviation_from_center, Cinv[c, :, :].T)).sum(1)
                )

        myMax = logR.max(axis=1).reshape((N, 1))
        L[i] = (
            (myMax.ravel() + np.log(np.exp(logR - myMax).sum(axis=1))).sum(axis=0)
            - N*d*np.log(2*3.141593)/2.
            )

        logR = logR - myMax - np.reshape(np.log(np.exp(logR - myMax).sum(axis=1)), (N, 1))

        myMax = logR.max(axis=0)
        logpi = myMax + np.log(np.exp(logR - myMax).sum(axis=0)) - np.log(N)
        logpi = logpi.T
        pi = np.exp(logpi)
        R = np.exp(logR)
        for c in range(p):
            mu[c, :] = (R[:, c].reshape((N, 1)) * X).sum(axis=0) / R[:, c].sum()
            deviation_from_center = X - mu[c, :].reshape((1, d))

            SW = ( (1/(pi[c]*N))
                * np.dot((R[:, c].reshape((N, 1)) * deviation_from_center).T,
                    np.dot(deviation_from_center, W[c, :, :]))
                )

            Wnew = np.dot(SW, np.linalg.inv(sigma2[c]*np.eye(q) + np.dot(np.dot(Minv[c, :, :], W[c, :, :].T), SW)))

            sigma2[c] = (1/d) * (
                (R[:, c].reshape(N, 1) * np.power(deviation_from_center, 2)).sum()
                /
                (N*pi[c])
                -
                np.trace(np.dot(np.dot(SW, Minv[c, :, :]), Wnew.T))
                )

            W[c, :, :] = Wnew

    return pi, mu, W, sigma2, R, L, sigma2hist


def mppca_predict(X, pi, mu, W, sigma2):
    N, d = X.shape
    p = len(sigma2)
    _, q = W[0].shape

    M = np.zeros((p, q, q))
    Minv = np.zeros((p, q, q))
    Cinv = np.zeros((p, d, d))
    logR = np.zeros((N, p))
    R = np.zeros((N, p))

    for c in range(p):
        # M
        M[c, :, :] = sigma2[c] * np.eye(q) + np.dot(W[c, :, :].T, W[c, :, :])
        Minv[c, :, :] = np.linalg.inv(M[c, :, :])

        # Cinv
        Cinv[c, :, :] = (np.eye(d)
            - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
            ) / sigma2[c]

        # R_ni
        deviation_from_center = X - mu[c, :]
        logR[:, c] = ( np.log(pi[c])
            + 0.5*np.log(
                np.linalg.det(
                    np.eye(d) - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
                )
            )
            - 0.5*d*np.log(sigma2[c])
            - 0.5*(deviation_from_center * np.dot(deviation_from_center, Cinv[c, :, :].T)).sum(1)
            )

    myMax = logR.max(axis=1).reshape((N, 1))
    logR = logR - myMax - np.reshape(np.log(np.exp(logR - myMax).sum(axis=1)), (N, 1))
    R = np.exp(logR)

    return R

