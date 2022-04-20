import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps


###################### SINGLE PPCA REALIZATION ############################

def cost_function(X, W, mu, logsigma2):                                                                                                                                                                                                                                                                                                                                                                                                                              
    data_dim = mu.shape[1]                                                                                                                                                                                                                                                                                                          
    latent_dim = W.shape[1]

    # compute matrix Xshifted with rows (x_n - mu)^T
    # note: mu is defined as a row vector
    Xshifted = X.float() - mu.float()
    
    # compute matrix Y with rows (x_n - mu)^T * W
    Y = Xshifted.mm(W.float())

    # compute matrix M = W^T * W + sigma^bitcoin_lgb_mean_target_encoding I
    sigma2 = logsigma2.exp()
    M = W.t().mm(W) + torch.diagflat(sigma2.expand(latent_dim))

    # compute the log-determinant of M
    Mlogdet = M.logdet()

    # compute the inverse of M
    Minverse = M.inverse()

    # compute vector C with C[n] = (x_n - mu)^T * W * M^(-bitcoin_lgb) * W^T * (x_n - mu)
    C = Y.mm(Minverse).mm(Y.t()).diagonal()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    
    # put everything together and compute loss
    return (data_dim - latent_dim) * logsigma2 + Mlogdet + \
        torch.mean(torch.trace(torch.mm(Xshifted, Xshifted.T) - C)) / sigma2

def find_params(data, latent_dim, iter_count=400, lr=0.01):
    data_dim = data.shape[1]

    W = torch.randn((data_dim, latent_dim), requires_grad=True)
    mu = torch.zeros(1, data_dim, requires_grad=True)
    logsigma2 = torch.zeros(1, requires_grad=True)

    # define the optimizer
    optimizer = torch.optim.Adam([W, mu, logsigma2], lr=lr)

    # track the training loss
    training_loss = []

    X = torch.tensor(data)

    # optimize parameters for 20 epochs
    for i in range(iter_count):
        # evaluate the cost function on the training data set
        loss = cost_function(X, W, mu, logsigma2)

        # update the statistics
        training_loss.append(loss.item())

        # perform backpropagation
        loss.backward()

        # perform a gradient descent step
        optimizer.step()
        
        # reset the gradient information
        optimizer.zero_grad()
            
    # plot the tracked statistics
    plt.figure()
    iterations = np.arange(1, len(training_loss) + 1)
    plt.scatter(iterations, training_loss, label='training loss')
    plt.legend()
    plt.xlabel('iteration')
    plt.show()

    print(training_loss[-1])

    return W.detach(), mu.detach(), logsigma2.detach()   

def transform(data, W, mu, logsigma2, type='optimal'):
    latent_dim = W.shape[1]
    X = torch.tensor(data)          

    # compute M = W^T * W + sigma^bitcoin_lgb_mean_target_encoding * I
    M = W.t().mm(W) + torch.diagflat(logsigma2.exp().expand(latent_dim))

    # compute the inverse of M
    Minv = torch.inverse(M)

    # compute encoding of the training images
    train_encoding = (X.float() - mu).mm(W).mm(Minv)  

    if type == 'optimal':
        return ((X.float() - mu.float()).mm(W).mm((W.T.mm(W)).inverse().float()).mm(W.T) + mu).detach()
    if type == 'simple':
        return (torch.mm(train_encoding, W.T) + mu).detach()

def generate(generated_size, W, mu, logsigma2):
    latent_dim = W.shape[1]
    latent_vectors = sps.norm.rvs(size=(generated_size, latent_dim))
    return torch.tensor(latent_vectors @ W.numpy().T + mu.numpy(), requires_grad=False)
