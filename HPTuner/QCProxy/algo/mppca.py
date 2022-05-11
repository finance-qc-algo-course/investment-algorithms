import numpy as np


class MPPCA():
    def __init__(self, latent_dim: int, n_models: int) -> None:
        self.latent_dim = latent_dim
        self.n_models = n_models

        self.EPS = 10e-6

    def fit(self, data: np.array, n_iter=100) -> None:
        self.data_dim = data.shape[1]

        self.W = np.random.rand(self.n_models, self.data_dim, self.latent_dim)
        self.mu = np.random.rand(self.n_models, self.data_dim)
        self.sigma2 = np.random.rand(self.n_models) + 0.01
        self.pi = np.random.rand(self.n_models) + 0.01

        self.pi, self.mu, self.W, self.sigma2, R, likelihood_hist, sigma2hist = \
            self._mppca_gem(data, self.pi, self.mu,
                            self.W, self.sigma2, n_iter)

    def _mppca_gem(self, X, pi, mu, W, sigma2, niter):
        # использован код отсюда https://github.com/michelbl/MPPCA
        '''
        Генерирует параметры для mppca.

        Вычисления обоснованы в статье Mixtures of Probabilistiс Principal Component Analysers
        Michael E. Tipping and Christopher M. Bishop.

        Возвращает tuple (pi, mu, W, sigma2, R, L, sigma2hist). pi, mu, W, sigma2, R параметры mppca,
        L - история функции правдоподобия, sigma2hist - история sigma^2.
        '''

        # N строк размерности d хотим уменьшить до латентной размерности q

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
                # для каждой компоненты считаем

                sigma2hist[c, i] = sigma2[c]

                # M = sigma^2 * I_q + W.T @ W
                M[c, :, :] = sigma2[c] * np.eye(q) + W[c, :, :].T @ W[c, :, :]
                Minv[c, :, :] = np.linalg.inv(M[c, :, :])

                # Cinv (страница 29 статьи)
                Cinv[c, :, :] = (np.eye(d) - W[c, :, :] @ Minv[c, :, :] @ W[c, :, :].T) / (
                    sigma2[c] + self.EPS)

                # R_ni, считаем только pi_i * p(t_n | i), (страница 9 статьи и 5)
                deviation_from_center = X - mu[c, :]
                detCinv = np.abs(np.linalg.det(Cinv[c, :, :] * sigma2[c]))
                if detCinv < self.EPS:
                    detCinv = self.EPS
                logR[:, c] = (np.log(pi[c])
                              # вычитаем ln(C) <=> прибавляем ln(C^-1)
                              + 0.5 * np.log(detCinv)
                              - 0.5 * d * np.log(np.pi)
                              - 0.5 * \
                              (deviation_from_center * (deviation_from_center @
                               Cinv[c, :, :].T)).sum(axis=1)
                              )

            # для каждого вектора выбираем лучшую по его мнению модель
            my_max = logR.max(axis=1).reshape((N, 1))

            MAX_FOR_EXP = 700
            # отношение R_ni к best R_n (по i)
            shifted_R = np.exp(
                np.clip(logR - my_max, -MAX_FOR_EXP, MAX_FOR_EXP))

            # подсчет функции правдоподобия, которую максимизируем
            L[i] = (my_max.ravel() + np.log(shifted_R.sum(axis=1))).sum(axis=0)

            # для каждого n (n - нумерует данные) считаем:
            # заметим, что shifted_R.sum(axis=1) - это p(t_n) / best R_n, где best R_n = e^(my_max)
            # тогда в итоге logR - my_max - (log(p(t_n)) - my_max) =
            # = logR - log(p(t_n), что нам и нужно
            logR = logR - my_max - \
                np.reshape(np.log(shifted_R.sum(axis=1)), (N, 1))

            # аналогично считаем pi_i (страница 9 статьи)
            my_max = logR.max(axis=0)
            shifted_R = np.exp(
                np.clip(logR - my_max.reshape((1, p)), -MAX_FOR_EXP, MAX_FOR_EXP))
            logpi = my_max + np.log(shifted_R.sum(axis=0)) - np.log(N)

            pi = np.exp(np.clip(logpi, -MAX_FOR_EXP, MAX_FOR_EXP))
            R = np.exp(np.clip(logR, -MAX_FOR_EXP, MAX_FOR_EXP))

            # пересчитаем mu_i и W_i (страница 30 статьи)
            for c in range(p):
                mu[c, :] = (R[:, c].reshape((N, 1)) *
                            X).sum(axis=0) / R[:, c].sum()
                deviation_from_center = X - mu[c, :].reshape((1, d))

                SW = ((1 / (pi[c] * N))
                      * np.dot((R[:, c].reshape((N, 1)) * deviation_from_center).T,
                               np.dot(deviation_from_center, W[c, :, :]))
                      )

                Wnew = SW @ np.linalg.inv(sigma2[c] * np.eye(q) +
                                          Minv[c, :, :] @ W[c, :, :].T @ SW)

                sigma2[c] = (1 / d) * (
                    (R[:, c].reshape(N, 1) *
                     np.power(deviation_from_center, 2)).sum()
                    /
                    (N * pi[c])
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

        return (data - mu) @ W @ (np.linalg.inv(W.T @ W + self.EPS * np.eye(W.shape[1]))) @ W.T + mu


##########################################################################
# some old version support
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
                             - np.dot(np.dot(W[c, :, :],
                                      Minv[c, :, :]), W[c, :, :].T)
                             ) / sigma2[c]

            # R_ni
            deviation_from_center = X - mu[c, :]
            logR[:, c] = (np.log(pi[c])
                          + 0.5*np.log(
                np.linalg.det(
                    np.eye(
                        d) - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
                )
            )
                - 0.5*d*np.log(sigma2[c])
                - 0.5*(deviation_from_center *
                       np.dot(deviation_from_center, Cinv[c, :, :].T)).sum(1)
            )

        myMax = logR.max(axis=1).reshape((N, 1))
        L[i] = (
            (myMax.ravel() + np.log(np.exp(logR - myMax).sum(axis=1))).sum(axis=0)
            - N*d*np.log(2*3.141593)/2.
        )

        logR = logR - myMax - \
            np.reshape(np.log(np.exp(logR - myMax).sum(axis=1)), (N, 1))

        myMax = logR.max(axis=0)
        logpi = myMax + np.log(np.exp(logR - myMax).sum(axis=0)) - np.log(N)
        logpi = logpi.T
        pi = np.exp(logpi)
        R = np.exp(logR)
        for c in range(p):
            mu[c, :] = (R[:, c].reshape((N, 1)) *
                        X).sum(axis=0) / R[:, c].sum()
            deviation_from_center = X - mu[c, :].reshape((1, d))

            SW = ((1/(pi[c]*N))
                  * np.dot((R[:, c].reshape((N, 1)) * deviation_from_center).T,
                           np.dot(deviation_from_center, W[c, :, :]))
                  )

            Wnew = np.dot(SW, np.linalg.inv(
                sigma2[c]*np.eye(q) + np.dot(np.dot(Minv[c, :, :], W[c, :, :].T), SW)))

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
                         - np.dot(np.dot(W[c, :, :],
                                  Minv[c, :, :]), W[c, :, :].T)
                         ) / sigma2[c]

        # R_ni
        deviation_from_center = X - mu[c, :]
        logR[:, c] = (np.log(pi[c])
                      + 0.5*np.log(
            np.linalg.det(
                np.eye(
                    d) - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
            )
        )
            - 0.5*d*np.log(sigma2[c])
            - 0.5*(deviation_from_center *
                   np.dot(deviation_from_center, Cinv[c, :, :].T)).sum(1)
        )

    myMax = logR.max(axis=1).reshape((N, 1))
    logR = logR - myMax - \
        np.reshape(np.log(np.exp(logR - myMax).sum(axis=1)), (N, 1))
    R = np.exp(logR)

    return R
