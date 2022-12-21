import os
import json
from backtesting import Backtest

from modules.data_load import TiingoLoader
from modules.model.wrappers import SimpleIndicatorsModel
from modules.preprocess import SimpleIndicatorsPreprocess

TIINGO_API_TOKEN_PATH = "./InvestingFramework/resources/api_token.txt"
CONFIG_PATH = "./InvestingFramework/config.json"

if __name__ == "__main__":
    with open(CONFIG_PATH, 'r') as config_file:
        configs = json.load(config_file)
    tiingo_loader = TiingoLoader(TIINGO_API_TOKEN_PATH)
    data = tiingo_loader.load_data(configs["train_tickers"][0], configs["train_from_date"], configs["train_to_date"])

    bt = Backtest(data, SimpleIndicatorsModel, commission=configs["backtest_comission"], cash=configs["backtest_initial_cash"])
    stats = bt.run()
    
    print(stats)