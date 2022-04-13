import pandas as pd
from stockstats import StockDataFrame
from catboost import CatBoostClassifier
from trader import Trader
from broker import Broker


def get_features(dataset: pd.DataFrame) -> StockDataFrame:
    INDICATORS = [
        # "volume_10_ema", "volume_10_std", "volume_10_sma",
        "macd", "macd_xu_macds", "macd_xd_macds",
        "boll", "high_x_boll_ub", "low_x_boll_lb",
        "rsi_14", "chop", "mfi",
        "volume_10_ema", "volume_10_sma", "volume_10_mstd"
    ]
    stock_df = StockDataFrame(dataset.copy())
    return stock_df[INDICATORS]

    return stock_df


class CorrelatedStrategy(Trader):
    def __init__(self, broker: Broker):
        super(CorrelatedStrategy, self).__init__(broker)
        self.classifier =

    def process_data(self, data: pd.Series, current_money: float, price: float) -> dict:
