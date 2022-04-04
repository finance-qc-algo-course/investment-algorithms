import numpy as np
import pandas as pd
from itertools import count
from bisect import bisect_left, insort
from enum import IntEnum
from typing import Any


class Action(IntEnum):
    BUY = 0
    SELL = 1
    WAIT = 2


class Order:
    _order_id = count(0)

    def __init__(self, ticker: str, price: float, amount: int, buy_timestamp: pd.Timestamp, action: Action,
                 lower_price_bound: float = None, upper_price_bound: float = None) -> None:
        self.id = next(self._order_id)
        self.ticker = ticker
        self.price = price
        self.amount = amount
        self.buy_timestamp = buy_timestamp
        self.action = action

        self.lower_price_bound = lower_price_bound
        self.upper_price_bound = upper_price_bound

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id


class OrderUpperBoundWrapper:
    def __init__(self, order: Order):
        self.order = order

    def __lt__(self, other):
        return self.order.upper_price_bound < other.order.upper_price_bound


class OrderLowerBoundWrapper:
    def __init__(self, order: Order):
        self.order = order

    def __lt__(self, other):
        return self.order.lower_price_bound > other.order.lower_price_bound


class Portfolio:
    def __init__(self) -> None:
        self.__all_orders = set()
        self.__upper_price_bounds = list()
        self.__lower_price_bounds = list()

    def add_asset(self, order: Order) -> None:
        self.__all_orders.add(order)
        if order.lower_price_bound is not None:
            insort(self.__lower_price_bounds, OrderLowerBoundWrapper(order))
        if order.upper_price_bound is not None:
            insort(self.__upper_price_bounds, OrderUpperBoundWrapper(order))

    def close_order(self, removing_order: Order, lower_bounds=False):
        if lower_bounds:
            from_idx = bisect_left(self.__lower_price_bounds, OrderLowerBoundWrapper(removing_order))
            for idx in range(from_idx, len(self.__lower_price_bounds)):
                order = self.__lower_price_bounds[idx].order
                if order.id == removing_order.id:
                    self.__lower_price_bounds.pop(idx)
                    break
                if order.lower_price_bound != removing_order.lower_price_bound:
                    break
        else:
            from_idx = bisect_left(self.__upper_price_bounds, OrderUpperBoundWrapper(removing_order))
            for idx in range(from_idx, len(self.__upper_price_bounds)):
                order = self.__upper_price_bounds[idx].order
                if order.id == removing_order.id:
                    self.__upper_price_bounds.pop(idx)
                    break
                if order.upper_price_bound != removing_order.upper_price_bound:
                    break

    def close_all_orders(self):
        for order in self.__all_orders:
            yield order

        self.__all_orders = set()
        self.__lower_price_bounds = list()
        self.__upper_price_bounds = list()

    def collect_orders(self, price):
        while self.__upper_price_bounds and price >= self.__upper_price_bounds[0].order.upper_price_bound:
            order = self.__upper_price_bounds.pop(0).order
            self.close_order(order, lower_bounds=True)
            if order in self.__all_orders:
                self.__all_orders.remove(order)

            yield order
        while self.__lower_price_bounds and price <= self.__lower_price_bounds[0].order.lower_price_bound:
            order = self.__lower_price_bounds.pop(0).order
            self.close_order(order, lower_bounds=False)

            yield order


class Broker:
    __brokers_count = count(0)
    MONEY_LOG_RATIO = 100

    def __init__(self, initial_money: float, commission: float, data: pd.DataFrame) -> None:
        current_broker_id = next(self.__brokers_count)
        assert current_broker_id < 1, "You can create only one broker"

        self.__session_starts = 0
        self.__money = initial_money
        self.__commission = commission
        self.__portfolio = Portfolio()
        self.__data = data
        self.__previous_timestamp = None
        self.__current_price = None

    def get_commission(self):
        return self.__commission

    def deal_cost(self, price: float, amount: int) -> float:
        return price * amount * (1 + self.__commission)

    def make_deal(self, order: dict[str, Any]):
        row = self.__data.loc[self.__previous_timestamp]
        if order["action"] != Action.WAIT:
            deal_cost = self.deal_cost(self.__current_price, order["amount"])
            self.__money -= deal_cost

            if order["action"] == Action.BUY:
                print("BUY order:")
            else:
                print("SELL order")
            print("- price:", self.__current_price)
            print("- amount:", order["amount"])

            self.__portfolio.add_asset(Order(
                ticker=row["ticker"],
                price=self.__current_price,
                amount=order["amount"],
                buy_timestamp=pd.Timestamp(row["date"]),
                action=order["action"],
                lower_price_bound=order["lower_price_bound"],
                upper_price_bound=order["upper_price_bound"]
            ))

    def process_data(self):
        self.__session_starts += 1
        assert self.__session_starts < 2, "You can start only one session"

        print("Starting new session with")
        print("- initial money:", self.__money)
        print("- commission:", self.__commission)

        for row_idx, row in self.__data.iterrows():
            # Generates price of current timestamp randomly near close price of previous one.
            self.__current_price = row["close"] + np.random.normal()
            self.__previous_timestamp = row_idx

            # Check for stop losses and take profits
            for closed_order in self.__portfolio.collect_orders(self.__current_price):
                if closed_order.action == Action.BUY:
                    if closed_order.upper_price_bound >= self.__current_price:
                        self.__money += self.deal_cost(closed_order.upper_price_bound, closed_order.amount)
                        print("Reaching take profit!")
                    elif closed_order.lower_price_bound <= self.__current_price:
                        self.__money += self.deal_cost(closed_order.lower_price_bound, closed_order.amount)
                        print("Reaching stop loss!")
                else:
                    if closed_order.upper_price_bound >= self.__current_price:
                        self.__money += self.deal_cost(closed_order.upper_price_bound, closed_order.amount)
                        print("Reaching stop loss!")
                    elif closed_order.lower_price_bound <= self.__current_price:
                        self.__money += self.deal_cost(closed_order.lower_price_bound, closed_order.amount)
                        print("Reaching take profit!")

            yield row, self.__money, self.__current_price

            if row_idx % self.MONEY_LOG_RATIO == 0:
                print(f"Timestamp: {row_idx}. Current money: {self.__money}")

        for order in self.__portfolio.close_all_orders():
            self.__money += order.amount * self.__current_price

        print("Final money:", self.__money)