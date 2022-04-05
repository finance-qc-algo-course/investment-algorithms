import pandas as pd
import os

from broker import Broker
from dummy_strategy import DummyTrader

INITIAL_MONEY = 1000000.0
COMMISSION = 0.004

PRICES_DATA_PATH = "../data/prices/"
DATASET_NAME = "AAPL_prices_15minute_test.csv"

if __name__ == "__main__":
    data = pd.read_csv(os.path.join(PRICES_DATA_PATH, DATASET_NAME), index_col=0)

    trader = DummyTrader(Broker(initial_money=INITIAL_MONEY, commission=COMMISSION, data=data))
    trader.start_trading()
