import abc
from abc import ABC
import pandas as pd


class Strategy(ABC):
    def process_data(self, data: pd.Series) -> dict:
        raise NotImplementedError
