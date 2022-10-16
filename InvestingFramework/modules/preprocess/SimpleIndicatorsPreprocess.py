from stockstats import StockDataFrame
import pandas as pd

from .Preprocessor import Preprocessor

class SimpleIndicatorsPreprocess(Preprocessor):
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        indicators = ["macd", "rsi"]
        stocks_data = StockDataFrame.retype(data)

        columns = list(data.columns) + indicators
        return pd.DataFrame(stocks_data[columns])
