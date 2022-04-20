import numpy as np
import pandas as pd
from stockstats import StockDataFrame
import talib
import joblib

from trader import Trader
from broker import Broker, Action


def get_features(dataset: pd.DataFrame) -> pd.DataFrame:
    indicators = [
        "macd", "macds",
        "macd_5_ema", "macd_5_mstd", "macd_10_ema", "macd_10_mstd",
        "macds_5_ema", "macds_10_ema", "macds_20_ema", "macds_5_mstd",
        "rsi", "rsi_6",
        "volume_5_ema", "volume_10_ema", "volume_20_ema", "volume_10_mstd",
        "close_5_mstd", "close_15_mstd", "close_20_mstd", "close_30_mstd",
        "trix", "middle_10_trix",
        "wr", "wr_6",
        "atr", "atr_5",
        "dma",
        "ppo", "ppos",
        "volume", "close",
    ]

    result_dataset = StockDataFrame(dataset.copy())

    result_dataset["apo"] = talib.APO(result_dataset["close"])
    result_dataset["bop"] = talib.BOP(result_dataset["open"], result_dataset["high"], result_dataset["low"],
                                      result_dataset["close"])
    result_dataset["cmo"] = talib.CMO(result_dataset["close"])
    result_dataset["minus_dm"] = talib.MINUS_DM(result_dataset["high"], result_dataset["low"])
    result_dataset["mom"] = talib.MOM(result_dataset["close"])
    result_dataset["willr"] = talib.WILLR(result_dataset["high"], result_dataset["low"],
                                          result_dataset["close"])

    additional_columns = ["apo", "bop", "cmo", "minus_dm", "mom", "willr"]

    return result_dataset[indicators + additional_columns]


class BitcoinTrader(Trader):
    DECISION_BOUNDARY = 50
    MAX_AMOUNT = 20

    def __init__(self, broker: Broker, model_path: str, warmup: int = 20):
        super(BitcoinTrader, self).__init__(broker)

        self.warmup = warmup
        self.iterations = 0
        self.model = joblib.load(open(model_path, "rb"))
        self.stock_data = pd.DataFrame()

    def process_data(self, data: pd.Series, current_money: float, price: float) -> None:
        self.stock_data = self.stock_data.append(data, ignore_index=True)
        self.iterations += 1
        if self.iterations <= self.warmup:
            return
        indicators_dataset = get_features(self.stock_data).to_numpy().copy()
        prediction = self.model.predict(indicators_dataset[-1].reshape(1, -1))
        amount = min(self.MAX_AMOUNT, int(np.abs((prediction[0] / self.DECISION_BOUNDARY) ** 2)))
        if prediction > self.DECISION_BOUNDARY:
            self.broker.make_deal(data["ticker"], amount, Action.BUY, price + 50, price - 30)
        elif prediction < -self.DECISION_BOUNDARY:
            self.broker.make_deal(data["ticker"], amount, Action.SELL, price + 30, price - 50)
