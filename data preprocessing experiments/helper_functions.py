import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as pof
import scipy.stats as sps
sns.set()

plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.rc('axes', labelsize=30)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('legend', fontsize=30)
plt.rc('figure', titlesize=30)

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.covariance import MinCovDet
from tqdm import tqdm
import cvxpy as cvx

from portfolio_optimizer import PortfolioOptimizer


def plot_results(names, optimizers, df_train, df_test, title, sp=None):
    fig = go.Figure()
    
    for name, optimizer in tqdm(zip(names, optimizers)):
        po = optimizer.best_estimator_
        po.fit(df_train)
        cum_predict = po.predict(df_test)
        fig.add_trace(go.Scatter(x=cum_predict.index, y=cum_predict, name=name))
        
    if sp is not None:
        fig.add_trace(go.Scatter(x=sp.index, y=sp, name='Индекс S&P 500'))
            
    fig.update_layout(title=title, xaxis_title='time', 
                  yaxis_title='return')
    fig.show()
    pof.plot(fig, filename=f'{title}.html', auto_open=False)
