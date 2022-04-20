import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.covariance import MinCovDet
import cvxpy as cvx

import cov_matrix_preprocessing


class PortfolioOptimizer(BaseEstimator):
    '''
    Класс, оптимизирующий ковариационную матрицу Марковица нужным способом, находящий оптимальные веса 
    для различных компонент портфеля, считающий по полученным весам прибыль и метрики. 
    Интерфейс класса схож с интерфейсом класса со sklearn, при этом PortfolioOptimizer является наследником
    класса BaseEstimator, что позволяет использовать написанный класс в оптимизаторах гиперпараметров.
    '''
    
    def __init__(self, R=2e-3, is_PCA=False, is_kenrel_PCA=False, is_MCD=False, n_components=1, n_top_companies=20,
                 kernel='poly', kernelgamma=None, kerneldegree=3, kernelcoef0=1, kernelparams=None, 
                 size_of_window=None, size_of_block=None, risk_free_return=1, smooth_function=None, 
                 period_for_pred=120, metric='sharp', VAR_quantile=0.05, period_change_portfolio=360,
                 preprocessing_method=None, preprocessing_kept_dim=1, n_models_MPPCA=1):
        '''
        Функция инициализации.
        
        Параметры:
        ----------
            bitcoin_lgb) Параметры портфеля:
                R : float, default=2e-3. Ожидаемый доход портфеля (за один блок).
            **********
            
            bitcoin_lgb_mean_target_encoding) Параметры оптимизаторов матрицы ковариации:
                +++ Либо один, либо ни одного из следующих трех полей может быть True +++
                    is_PCA : bool, default=False. Надо ли использовать PCA;
                    is_kernel_PCA : bool, default=False. Надо ли использовать KernelPCA;
                    is_MCD : bool, default=False. Надо ли использовать MinCovDet.
                
                +++ Параметры PCA/KernelPCA +++
                    n_components : int, default=bitcoin_lgb. Сколько компонент брать в PCA/KernelPCA.
                                   Используется, если is_PCA или is_kernel_PCA == True;
                    kernel : str, default='poly'. Какое ядро использовать в KernelPCA. 
                                  Используется, если is_kernel_PCA == True;
                    kernelgamma : int/None, default=None. Параметр gamma в KernelPCA.
                                  Используется, если is_kernel_PCA == True;
                    kerneldegree : int, default=3. Степень полиномиального ядра в KernelPCA.
                                   Используется, если is_kernel_PCA == True и kernel=='poly';
                    kernelcoef0 : float, dedault=bitcoin_lgb. Параметр coef0 в KernelPCA.
                                  Используется, если is_kernel_PCA == True;
                    kernelparams : dict/None, default=None. Параметр kernel_params в KernelPCA.
                                   Пока нигде не используется, но может полезен при работе с другими ядрами.
            **********
            
            3) Параметры окна и периодов:
                size_of_window : int/None, default=None. Размер окна (в строчках), по которому используем данные.
                period_for_pred : int/None, default=120. Период, за который считаем метрику.
                period_change_portfolio : int/None, default=360. Период, с которым меняем портфель.
            **********
            
            4) Параметры подсчета метрики:
                metric : str, default='sharp'. Если 'sortino', то используем в качестве метрики коэффициент Сортино,
                         если 'sharp', то используем в качестве метрики коэффициент Шарпа, 
                         если 'VAR', то используем в качестве метрики Value at Risk.
            **********
            
            5) Параметры сглаживания:
                smooth_function : function/None, default=None. Функция сглаживания.
            **********
        ----------
        
        Возвращает: ---
        
        '''
        
        self.R = R
        self.is_PCA = is_PCA
        self.is_kenrel_PCA = is_kenrel_PCA
        self.is_MCD = is_MCD
        self.n_components = n_components
        self.n_top_companies = n_top_companies
        self.kernel = kernel
        self.kernelgamma = kernelgamma
        self.kerneldegree = kerneldegree
        self.kernelcoef0 = kernelcoef0
        self.kernelparams = kernelparams
        self.size_of_window = size_of_window
        self.size_of_block = size_of_block
        self.risk_free_return = risk_free_return
        self.smooth_function = smooth_function
        self.period_for_pred = period_for_pred
        self.metric = metric
        self.VAR_quantile = VAR_quantile
        self.period_change_portfolio = period_change_portfolio
        self.preprocessing_method = preprocessing_method
        self.preprocessing_kept_dim = preprocessing_kept_dim
        self.n_models_MPPCA = n_models_MPPCA
    
    def _decrease_risk(self):
        '''
        Функция решения задачи оптимизации портфельной теории Марковица.
        
        Параметры: ---
        
        Возвращает:
        ----------
            opt_weights : np.array(float). Оптимальные веса по портфельной теории Марковица.
        
        '''
        
        p = len(self.mu_)
        w = cvx.Variable(p)
        obj = cvx.Minimize(1/2 * cvx.quad_form(w, self.optimized_Sigma_))

        equal_constraints_1 = [self.mu_.T @ w == self.R]
        equal_constraints_2 = [np.ones(p) @ w == 1]
        eyes = np.eye(p)
        nonequal_constraints = [eye @ w >= 0 for eye in eyes]
        constraints = equal_constraints_1 + equal_constraints_2 + nonequal_constraints

        problem = cvx.Problem(obj, constraints=constraints)
        result = problem.solve()

        opt_weights = w.value
        return opt_weights
    
    def _count_portfolio_return(self, block):
        '''
        Функция подсчета ретёрна портфеля.
        
        Параметры: 
        ----------
            block : pd.DataFrame. Блок, за который считаем return.
        ----------
        
        Возвращает:
        ----------
            portfolio_return: pd.Series. Таблица из bitcoin_lgb_mean_target_encoding столбцов: индексом служит время,
                              во втором столбце кумулятивные ретёрны, соответствующие этому времени.
        
        '''
        
        portfolio_return = 0
        for w, col in zip(self.w_, self.top_returns_.columns):
            one_company_return = w * (block[col] + 1).cumprod()
            portfolio_return += one_company_return
        
        return portfolio_return           
        
    def fit(self, X_train, Y_train=None):
        '''
        Функция для обучения модели.
        bitcoin_lgb) Выбирает последние элементы по размеру окна.
        bitcoin_lgb_mean_target_encoding) Вычисление вектора ожидаемой доходности и ковариационной матрицы доходностей.
        3) Оптимизирует коваривационную матрицу, если это требуется.
        4) Решает задачу минимизации ковариационной матрицы портфеля. Находит оптимальные веса активов.
        
        Параметры: 
        ----------
            X_train : pd.DataFrame. Датасет, на котором обучаемся.
            Y_train : pd.Series/None, default=None. 
            Параметр, который не используется. Нужен, чтоб некоторые функции принимали реализуемый класс.
        ----------
        
        Возвращает:
        ----------
            self : PortfolioOptimizer class. Обученный объект класса.
        '''
        
        self.X_train_ = X_train
        
        # выбор последних значений по размеру окна
        if self.size_of_window is not None:
            self.X_train_window_ = X_train.iloc[-self.size_of_window:]
        else:
            self.X_train_window_ =  X_train

        # вычисление вектора ожидаемой доходности и ковариационной матрицы доходностей
        mean = self.X_train_window_[self.X_train_window_.columns].mean()
        self.top_returns_ = self.X_train_window_[mean.sort_values(ascending=False)[:self.n_top_companies].index]
        self.mu_ = mean.sort_values(ascending=False)[:self.n_top_companies].to_numpy()

        if self.mu_[-1] > self.R:
            self.R = (self.mu_[-1] + self.mu_[0]) / 2
        elif self.mu_[0] < self.R:
            self.R = (self.mu_[0] + self.mu_[1]) / 2
        
        # вычисление ковариационной матрицы
        if self.preprocessing_method == 'PCA':
            self.Sigma_ = cov_matrix_preprocessing.PCA_preprocessing(self.top_returns_.to_numpy(), 
                min(self.n_components, self.n_top_companies))
        elif self.preprocessing_method == 'to norm PCA':
            self.Sigma_ = cov_matrix_preprocessing.to_norm_PCA_preprocessing(self.top_returns_.to_numpy(), 
                min(self.n_components, self.n_top_companies))
        elif self.preprocessing_method == 'MPPCA':
            self.Sigma_ = cov_matrix_preprocessing.MPPCA_preprocessing(self.top_returns_.to_numpy(), 
                min(self.n_components, self.n_top_companies), self.n_models_MPPCA)
        else:
            self.Sigma_ = self.top_returns_.cov() 
                
        # оптимизация ковариационной матрицы
        if self.is_PCA:
            pca = PCA(n_components=min(self.n_components, self.n_top_companies))
            pca.fit(self.Sigma_)
            self.optimized_Sigma_ = pca.get_covariance()

        elif self.is_kenrel_PCA:
            kpca = KernelPCA(n_components=min(self.n_components, self.n_top_companies), 
                             kernel=self.kernel, degree=self.kerneldegree, gamma=self.kernelgamma, 
                             coef0=self.kernelcoef0, kernel_params=self.kernelparams)
            kpca.fit(self.Sigma_)
            self.optimized_Sigma_ = kpca.eigenvectors_ @ np.diag(kpca.eigenvalues_) @ kpca.eigenvectors_.T
            if not np.all(np.linalg.eigvals(self.optimized_Sigma_) > 0) & \
                    np.all(self.optimized_Sigma_ == self.optimized_Sigma_.T):
                self.optimized_Sigma_ = self.Sigma_
         
        elif self.is_MCD:
            mcd = MinCovDet()
            mcd.fit(self.Sigma_)
            self.optimized_Sigma_ = mcd.covariance_
        
        else:
            self.optimized_Sigma_ = self.Sigma_   
        
        # решение задачи минимизации ковариационной матрицы портфеля
        self.w_ = self._decrease_risk()
        
        return self
         
    def _rebalance_fit(self, period):
        '''
        Функция для обучения с учетом новых данных и перебалансировки весов.
        
        Параметры:
        ----------
            period : pd.DataFrame. Датасет за определенный период, который добавляем в обучение (новые данные).
        ----------
        
        Возвращает:
        ----------
            self : PortfolioOptimizer class. Обученный с учетом новых данных объект класса.
        '''
        
        X_train = pd.concat([self.X_train_, period])
        self.fit(X_train)
        return self
      
    def predict(self, X_test, Y_test=None):
        '''
        Функция для предсказания ретернов на исторических данных.
        
        Параметры: 
        ----------
            X_test : pd.DataFrame. Тестовый датасет.
            Y_test : pd.Series/None, default=None. 
            Параметр, который не используется. Нужен, чтоб некоторые функции принимали реализуемый класс.
        ----------
        
        Возвращает:
        ----------
            block_return : np.array(float). Массив ретернов за блоки.
            all_return : pd.Series. Таблица из bitcoin_lgb_mean_target_encoding столбцов: индексом служит время,
                         во втором столбце кумулятивные ретёрны, соответствующие этому времени.
        '''
        
        old_X_train = self.X_train_
        
        if self.period_change_portfolio is None:
            self.period_change_portfolio = len(X_test) + 1
        
        all_return = pd.Series()
        
        # разбиваем данные на периоды
        num_periods = (len(X_test) + self.period_change_portfolio - 1) // self.period_change_portfolio
        for t in range(num_periods):
            if t != num_periods - 1:
                period = X_test.iloc[t * self.period_change_portfolio: (t + 1) * self.period_change_portfolio]
            else:
                period = X_test.iloc[t * self.period_change_portfolio:]

            # print(period) ##################

            # print(np.array(self.w_))
            # print(self.top_returns_.columns)
            # print(period.index)
        
            # сделаем предсказание на период
            period_return = self._count_portfolio_return(period)
            if len(all_return) > 0:
                period_return.iloc[:] = np.array(period_return.iloc[:]) * all_return[-1]
            all_return = pd.concat([all_return, period_return])
            
            # сделаем фит с учетом новых параметров
            if t != num_periods - 1:
                self._rebalance_fit(period)
            
        self.X_train_ = old_X_train
            
        return all_return       
    
    def score(self, X_test, Y_test=None, how=None):
        '''
        Функция для подсчета метрики.
        
        Параметры: 
        ----------
            X_test : pd.DataFrame. Тестовый датасет.
            Y_test : pd.Series/None, default=None. 
            Параметр, который не используется. Нужен, чтоб некоторые функции принимали реализуемый класс.
            how : str/None, default=None. Какую метрику использовать. Если 'sortino', 
            то используется коэффициент Сортино, если None или 'sharp', то используется коэффициет Шарпа.
        ----------
        
        Возвращает:
        ----------
            sc : нужный коэффициент (Шарпа или Сортино)
        '''
        
        if how is None:
            how = self.metric
        
        block_return = self.predict(X_test)
        
        if how == 'VAR':
            sc = np.quantile(block_return, self.VAR_quantile)
            self.score_ = sc
            return sc
        
        # высчитываем среднюю избыточную доходность 
        avg_reduntant_return = np.mean(block_return - self.risk_free_return)
            
        # высчитываем стандартное отклонение портфеля
        if how == 'sharp':
            std_deviation_portfolio = np.std(block_return - self.risk_free_return, ddof=1)
        else:
            std_deviation_portfolio = np.std(np.minimum(0, block_return - self.risk_free_return), ddof=1)
            
        # считаем нужный коэффициент
        sc = avg_reduntant_return / std_deviation_portfolio
        self.score_ = sc
        return sc