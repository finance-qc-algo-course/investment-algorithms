import numpy as np
import pandas as pd
from stockstats import StockDataFrame
import talib
import joblib
import pickle

from trader import Trader
from broker import Broker, Action

MEAN_TARGET_ENCODING_PATH = "./mean_target_encoding"


def estimate_mean_target_regularization(category, global_mean: float, category_means: pd.Series,
                                        category_counts: pd.Series, alpha: float) -> float:
    try:
        return (category_counts[category] * category_means[category] + global_mean * alpha) / (category_counts[category] * alpha)
    except:
        return global_mean


def mean_target_regularization(dataset: pd.DataFrame, column: str, target_name: str, train: bool = True) -> pd.Series:
    encoding_file_path = MEAN_TARGET_ENCODING_PATH + "_" + column + ".pickle"
    if train:
        encoding = {
            "global_mean": dataset[target_name].mean(),
            "counts": dataset.groupby(dataset[column]).count()[target_name],
            "category_means": dataset.groupby(dataset[column]).mean()[target_name]
        }
        with open(encoding_file_path, "wb") as output_file:
            pickle.dump(encoding, output_file)
    else:
        with open(encoding_file_path, "rb") as input_file:
            encoding = pickle.load(input_file)

    global_mean = encoding["global_mean"]
    counts = encoding["counts"]
    category_means = encoding["category_means"]

    target_mean = dataset[column].apply(
        lambda category: estimate_mean_target_regularization(
            category,
            global_mean=global_mean,
            category_means=category_means,
            category_counts=counts,
            alpha=10
        )
    )

    return target_mean


def get_features(dataset: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    indicators = [
        "macd", "macds", "macdh",
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
        "volume", "close"
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

    rsi_bins = [0, 20, 40, 60, 80, 100]
    rsi_labels = [0, 1, 2, 3, 4]
    wr_bins = [-100, -80, -60, -40, -20, 0]
    wr_labels = [0, 1, 2, 3, 4]
    macdh_bins = np.linspace(-1000, 1000, 6)
    macdh_labels = [0, 1, 2, 3, 4]

    result_dataset["rsi_bins"] = pd.cut(result_dataset["rsi"], rsi_bins, labels=rsi_labels).astype(float)
    result_dataset["rsi_6_bins"] = pd.cut(result_dataset["rsi_6"], rsi_bins, labels=rsi_labels).astype(float)
    result_dataset["wr_bins"] = pd.cut(result_dataset["wr"], wr_bins, labels=wr_labels).astype(float)
    result_dataset["wr_6_bins"] = pd.cut(result_dataset["wr_6"], wr_bins, labels=wr_labels).astype(float)
    result_dataset["macdh_bins"] = pd.cut(result_dataset["macdh"], macdh_bins, labels=macdh_labels).astype(float)

    result_dataset["rsi_mean_target"] = mean_target_regularization(result_dataset, "rsi_bins", "target", train)
    result_dataset["rsi_6_mean_target"] = mean_target_regularization(result_dataset, "rsi_6_bins", "target", train)
    result_dataset["wr_mean_target"] = mean_target_regularization(result_dataset, "wr_bins", "target", train)
    result_dataset["wr_6_mean_target"] = mean_target_regularization(result_dataset, "wr_6_bins", "target", train)
    result_dataset["macdh_mean_target"] = mean_target_regularization(result_dataset, "macdh_bins", "target", train)

    additional_columns = ["apo", "bop", "cmo", "minus_dm", "mom", "willr", "rsi_mean_target",
                          "rsi_6_mean_target", "wr_mean_target", "wr_6_mean_target", "macdh_mean_target"]

    if train:
        additional_columns += ["target"]

    return result_dataset[indicators + additional_columns]


class BitcoinTrader(Trader):
    MODEL_PATH = "./bitcoin_trader.pickle"
    DECISION_BOUNDARY = 50
    MAX_AMOUNT = 20

    def __init__(self, broker: Broker, warmup: int = 20):
        super(BitcoinTrader, self).__init__(broker)

        self.warmup = warmup
        self.iterations = 0
        self.model = joblib.load(open(self.MODEL_PATH, "rb"))
        self.stock_data = pd.DataFrame()

    def process_data(self, data: pd.Series, current_money: float, price: float) -> None:
        self.stock_data = self.stock_data.append(data, ignore_index=True)
        self.iterations += 1
        if self.iterations <= self.warmup:
            return
        indicators_dataset = get_features(self.stock_data, train=False).to_numpy().copy()
        prediction = self.model.predict(indicators_dataset[-1].reshape(1, -1))
        amount = min(self.MAX_AMOUNT, int(np.abs((prediction[0] / self.DECISION_BOUNDARY) ** 2)))
        if prediction > self.DECISION_BOUNDARY:
            self.broker.make_deal(data["ticker"], amount, Action.BUY, price + 50, price - 30)
        elif prediction < -self.DECISION_BOUNDARY:
            self.broker.make_deal(data["ticker"], amount, Action.SELL, price + 30, price - 50)
