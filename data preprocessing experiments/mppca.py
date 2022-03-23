import numpy as np


# class MPPCA():
#     def __init__(self, data_dim, latent_dim, components_number) -> None:
#         self.data_dim = data_dim
#         self.latent_dim = latent_dim
#         self.components_number = components_number
    
#     def fit(self, data, n_iterations=10):
#         assert data.shape[1] == self.data_dim
#         self.W = np.random.rand(self.components_number, self.data_dim, self.latent_dim)
#         self.mu = np.random.rand(self.components_number, self.data_dim)
#         self.sigma2 = np.random.rand(1)
#         self.pi = np.random.rand(self.components_number)
#         self.pi /= np.sum(self.pi)

#         old_W = self.W.copy()
#         old_mu = self.mu.copy()
#         old_sigma2 = self.sigma2.copy()
#         old_pi = self.pi.copy()

#         self.logR = np.zeros((self.data_dim, self.latent_dim))

#         for iter in range(n_iterations):
#             logR = self.__update_log_R()
            
    
#     def __update_log_R(self, data):
#         pass
#         # logR = np.zeros((self.data_dim, self.latent_dim))
#         # for i in range(self.components_number):
#         #     deviation_from_center = data - mu[c, :]
#         #     logR[:, c] = ( np.log(pi[c])
#         #         + 0.5*np.log(
#         #             np.linalg.det(
#         #                 np.eye(d) - np.dot(np.dot(W[c, :, :], Minv[c, :, :]), W[c, :, :].T)
#         #             )
#         #         )
#         #         - 0.5*d*np.log(sigma2[c])
#         #         - 0.5*(deviation_from_center * np.dot(deviation_from_center, Cinv[c, :, :].T)).sum(1)
#         #         )


##########################################################################

def initialization_kmeans(X, p, q, variance_level=None):
    """
    X : dataset
    p : number of clusters
    q : dimension of the latent space
    variance_level
    pi : proportions of clusters
    mu : centers of the clusters in the observation space
    W : latent to observation matricies
    sigma2 : noise
    """

    N, d = X.shape

    # initialization
    init_centers = np.random.randint(0, N, p)
    while (len(np.unique(init_centers)) != p):
        init_centers = np.random.randint(0, N, p)

    mu = X[init_centers, :]
    distance_square = np.zeros((N, p))
    clusters = np.zeros(N, dtype=np.int32)

    D_old = -2
    D = -1

    while(D_old != D):
        D_old = D

        # assign clusters
        for c in range(p):
            distance_square[:, c] = np.power(X - mu[c, :], 2).sum(1)
        clusters = np.argmin(distance_square, axis=1)

        # compute distortion
        distmin = distance_square[range(N), clusters]
        D = distmin.sum()

        # compute new centers
        for c in range(p):
            mu[c, :] = X[clusters == c, :].mean(0)

    #for c in range(p):
    #    plt.scatter(X[clusters == c, 0], X[clusters == c, 1], c=np.random.rand(3,1))

    # parameter initialization
    pi = np.zeros(p)
    W = np.zeros((p, d, q))
    sigma2 = np.zeros(p)
    for c in range(p):
        if variance_level:
            W[c, :, :] = variance_level * np.random.randn(d, q)
        else:
            W[c, :, :] = np.random.randn(d, q)

        pi[c] = (clusters == c).sum() / N
        if variance_level:
            sigma2[c] = np.abs((variance_level/10) * np.random.randn())
        else:
            sigma2[c] = (distmin[clusters == c]).mean() / d

    return pi, mu, W, sigma2, clusters


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

def transform(data, W, mu, sigma2, type='optimal'):
    latent_dim = W.shape[1]         

    # compute M = W^T * W + sigma^2 * I
    M = W.T @ W + np.diagflat(np.ones(latent_dim) * sigma2)

    # compute the inverse of M
    Minv = np.linalg.inv(M)

    # compute encoding of the training images
    train_encoding = (data - mu) @ W @ Minv  

    if type == 'optimal':
        return (data - mu) @ W @ (np.linalg.inv(W.T @ W)) @ W.T + mu
    if type == 'simple':
        return train_encoding @ W.T + mu