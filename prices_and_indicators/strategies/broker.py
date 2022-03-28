import pandas as pd
from strategy import Strategy


class Asset:
    def __init__(self, ticker: str, buy_price: float, buy_timestamp: pd.Timestamp) -> None:
        self.ticker = ticker
        self.buy_price = buy_price
        self.buy_timestamp = buy_timestamp


class Portfolio:
    def __init__(self) -> None:
        self.assets = []

    def add_asset(self, asset: Asset) -> None:
        self.assets.append(asset)


class Order:
    def __init__(self, asset: Asset, stop_loss: float, take_profit: float):
        assert asset.buy_price > stop_loss, "Buy price should be higher than stop loss"
        assert asset.buy_price < take_profit, "Buy price should be lower than take profit"

        self.asset = asset
        self.stop_loss = stop_loss
        self.take_profit = take_profit


class Broker:
    def __init__(self, initial_money: float, commission: float, strategy: Strategy) -> None:
        self.money = initial_money
        self.commission = commission
        self.strategy = strategy
        self.portfolio = Portfolio()
        self.orders = list[Order]

    def start_session(self, data: pd.DataFrame) -> None:
        for row_idx, row in data.iterrows():
            order = self.strategy.process_data(row)

            if order["action"] == "buy":
                raise NotImplementedError
            elif order["action"] == "sell":
                raise NotImplementedError
            elif order["action"] == "wait":
                pass
            else:
                raise NotImplementedError
