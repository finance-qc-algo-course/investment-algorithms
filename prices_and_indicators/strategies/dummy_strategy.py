import pandas as pd
from trader import Trader
from broker import Broker, Action


class DummyTrader(Trader):
    def __init__(self, broker: Broker):
        super(DummyTrader, self).__init__(broker)

    def process_data(self, data: pd.Series, current_money: float, price: float) -> dict:
        buy_amount = current_money // (price * (1 + self.broker.get_commission()))
        if buy_amount > 0:
            return {
                "action": Action.BUY,
                "amount": buy_amount,
                "upper_price_bound": None,
                "lower_price_bound": None}
        else:
            return {"action": Action.WAIT}
