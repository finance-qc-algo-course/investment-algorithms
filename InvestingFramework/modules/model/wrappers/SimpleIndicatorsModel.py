from backtesting.lib import SignalStrategy
from catboost import CatBoostRegressor

from ..base import Model

class SimpleIndicatorsModel(Model):
    def __init__(self, train_data):
        self.model = CatBoostRegressor()