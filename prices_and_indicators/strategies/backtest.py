import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

from broker import Broker
from dummy_strategy import DummyTrader

INITIAL_MONEY = 1000000.0
COMMISSION = 0.004

# MODEL_DIRECTORY_PATH = "./bitcoin_lgb_mean_target_encoding/"
MODEL_DIRECTORY_PATH = "./bitcoin_lgb/"
PRICES_DATA_PATH = "../data/prices/"
DATASET_NAME = "BTC_15minute_test.csv"

sys.path.append(MODEL_DIRECTORY_PATH)

from smart_trader import BitcoinTrader

if __name__ == "__main__":
    data = pd.read_csv(os.path.join(PRICES_DATA_PATH, DATASET_NAME), index_col=0)
    sample_data = data.iloc[:]

    broker = Broker(initial_money=INITIAL_MONEY, commission=COMMISSION, data=sample_data)
    DummyTrader(broker).start_trading()
    dummy_trader_money_history = broker.money_history

    broker = Broker(initial_money=INITIAL_MONEY, commission=COMMISSION, data=sample_data)
    BitcoinTrader(broker, MODEL_DIRECTORY_PATH + "model.pkl").start_trading()
    bitcoin_trader_money_history = broker.money_history

    plt.figure(figsize=(12, 8))

    bins = np.arange(len(dummy_trader_money_history))
    plt.plot(bins, dummy_trader_money_history, label="Dummy trader")
    plt.plot(bins, bitcoin_trader_money_history, label="Bitcoin trader")

    plt.title("Comparison of algorithms", fontsize=20)
    plt.xlabel("Timestamp", fontsize=18)
    plt.ylabel("Balance", fontsize=18)

    plt.legend()
    plt.savefig("bitcoin_strategy_results.png")
