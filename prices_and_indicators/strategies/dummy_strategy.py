import pandas as pd
from trader import Trader
from broker import Broker, Action


class DummyTrader(Trader):
    def __init__(self, broker: Broker):
        super(DummyTrader, self).__init__(broker)

    def process_data(self, data: pd.Series, current_money: float, price: float) -> None:
        amount = current_money // (price * (1 + self.broker.get_commission()))
        if amount > 0:
            self.broker.make_deal(data["ticker"], amount, Action.BUY)
