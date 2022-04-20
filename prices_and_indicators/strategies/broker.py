import numpy as np
import pandas as pd
from itertools import count
from enum import IntEnum


class Action(IntEnum):
    BUY = 0
    SELL = 1


class Order:
    _order_id = count(0)

    def __init__(self, ticker: str, price: float, amount: int, action: Action,
                 lower_price_bound: float = None, upper_price_bound: float = None) -> None:
        self.id = next(self._order_id)
        self.ticker = ticker
        self.price = price
        self.amount = amount
        self.action = action

        self.lower_price_bound = lower_price_bound
        self.upper_price_bound = upper_price_bound

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __repr__(self):
        if self.action == Action.BUY:
            action = "BUY"
        elif self.action == Action.SELL:
            action = "SELL"
        return f"Ticker: {self.ticker}; price: {self.price}; amount: {self.amount}; action: {action}"


class Portfolio:
    def __init__(self) -> None:
        self.__orders = set()

    def add_order(self, order: Order) -> None:
        self.__orders.add(order)

    def collect_all_orders(self):
        for order in self.__orders:
            yield order

        self.__orders = set()

    def collect_orders(self, price):
        orders_to_remove = set()
        for order in self.__orders:
            if (order.upper_price_bound is not None and order.upper_price_bound < price) or \
                    (order.lower_price_bound is not None and order.lower_price_bound > price):
                orders_to_remove.add(order)
                yield order

        self.__orders -= orders_to_remove

    def get_stocks_amount(self, ticker: str):
        amount = 0
        for order in self.__orders:
            if order.ticker == ticker:
                if order.action == Action.BUY:
                    amount += order.amount
                else:
                    amount -= order.amount

        return amount


class Broker:
    MONEY_LOG_RATIO = 100

    def __init__(self, initial_money: float, commission: float, data: pd.DataFrame) -> None:
        self.__session_starts = 0
        self.__money = initial_money
        self.__commission = commission
        self.__portfolio = Portfolio()
        self.__data = data
        self.__current_price = None
        self.order_history = []
        self.money_history = []

    def estimate_balance(self):
        return self.__money + self.__current_price * self.__portfolio.get_stocks_amount("X:BTCUSD")

    def get_commission(self) -> float:
        return self.__commission

    def deal_cost(self, price: float, amount: int) -> float:
        return price * amount * (1 + self.__commission)

    def close_deal(self, ticker: str, amount: int, action: Action,
                   lower_price_bound: float = None, upper_price_bound: float = None) -> Order:
        order = Order(ticker, self.__current_price, amount, action, lower_price_bound, upper_price_bound)
        deal_cost = self.deal_cost(order.price, order.amount)
        if action == Action.SELL:
            deal_cost *= -1
        if deal_cost > self.__money:
            return None
        self.order_history.append(order)
        self.__money -= deal_cost

        return order

    def make_deal(self, ticker: str, amount: int, action: Action,
                  lower_price_bound: float = None, upper_price_bound: float = None) -> Order:
        order = self.close_deal(ticker, amount, action, lower_price_bound, upper_price_bound)
        if order is None:
            return None
        self.__portfolio.add_order(order)
        # print(order)

        return order

    def process_data(self):
        self.__session_starts += 1
        assert self.__session_starts < 2, "You can start only one session"

        print("Starting new session with")
        print("- initial money:", self.__money)
        print("- commission:", self.__commission)

        for row_idx, row in self.__data.iterrows():
            # Generates price of current timestamp randomly near close price of previous one.
            self.__current_price = row["close"] + np.random.normal()

            # Check for stop losses and take profits
            for closed_order in self.__portfolio.collect_orders(self.__current_price):
                if closed_order.action == Action.BUY:
                    action = Action.SELL
                else:
                    action = Action.BUY
                self.close_deal(closed_order.ticker, closed_order.amount, action)

            if row_idx % self.MONEY_LOG_RATIO == 0:
                print(f"Date: {row['date']}. Current money: {self.__money}")

            self.money_history.append(self.estimate_balance())
            yield row, self.__money, self.__current_price

        for closed_order in self.__portfolio.collect_all_orders():
            if closed_order.action == Action.BUY:
                action = Action.SELL
            else:
                action = Action.BUY
            self.close_deal(closed_order.ticker, closed_order.amount, action)

        print("Final money:", self.__money)
