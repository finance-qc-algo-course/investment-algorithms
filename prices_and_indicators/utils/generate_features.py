import pandas as pd
import numpy as np
from stockstats import StockDataFrame

DROP_COLUMNS = ["t", "n", "vw"]
RENAME_COLUMNS = {"c": "close", "h": "high", "l": "low", "o": "open", "v": "volume"}


def indicators_with_rowwise_ewm(dataset: pd.DataFrame, ewm_alpha: float = 0.3, ewm_period: int = 10) -> pd.DataFrame:
    INDICATORS = [
        "volume_26_ema", "volume_42_ema", "volume_86_ema", "volume_174_ema", "volume_363_ema", "volume_42_smma",
        "volume_6_mstd", "volume_14_mstd", "volume_26_mstd", "volume_42_mstd", "volume_86_mstd", "volume_174_mstd", "volume_363_mstd",
        "high_42_mstd", "high_86_mstd", "high_174_mstd", "high_363_mstd",
        "low_42_mstd", "low_86_mstd", "low_174_mstd", "low_363_mstd",
        "wt1", "wt2",
        "vr_6", "vr_14", "vr_26", "vr_42", "vr_86", "vr_174", "vr_363",
        "atr_6", "atr_3", "atr_2",
        "adxr",
        "kdjk_6",
        "cr", "cr-ma1", "cr-ma2", "cr-ma3",
        "macd", "boll",
        "chop_6", "chop_14", "chop_26", "chop_42", "chop_86", "chop_174", "chop_363",
        "mfi_6", "mfi_14", "mfi_26", "mfi_42", "mfi_86", "mfi_174", "mfi_363"
    ]

    result_dataset = dataset.copy()
    
    result_dataset.drop(DROP_COLUMNS, axis=1, inplace=True)
    result_dataset.reset_index(drop=True, inplace=True)
    result_dataset.rename(RENAME_COLUMNS, axis=1, inplace=True)

    stock_df = StockDataFrame(result_dataset)
    stock_df[INDICATORS]
    
    result_dataset = result_dataset.join(result_dataset.ewm(alpha=ewm_alpha, min_periods=ewm_period).mean(), rsuffix="_ewm")
    result_dataset.reset_index(drop=True, inplace=True)

    return result_dataset


def lstm_indicators(dataset: pd.DataFrame) -> pd.DataFrame:
    INDICATORS = [
        "volume_6_ema", "volume_14_ema", "volume_26_ema", "volume_42_ema", 
        "volume_6_smma", "volume_14_smma", "volume_26_smma", "volume_42_smma",
        "volume_6_mstd", "volume_14_mstd", "volume_26_mstd", "volume_42_mstd",
        "high_6_mstd", "high_14_mstd", "high_26_mstd", "high_42_mstd",
        "low_6_mstd", "low_14_mstd", "low_26_mstd", "low_42_mstd",
        "wt1", "wt2",
        # "vr_6",
        "atr_6",
        "adxr",
        "kdjk_6",
        "cr", "cr-ma1", "cr-ma2", "cr-ma3",
        "macd", "boll",
        "chop_14", "chop_26", "chop_42",
        "mfi_6", "mfi_14", "mfi_26", "mfi_42",
    ]
    stock_df = StockDataFrame(dataset.copy())
    stock_df[INDICATORS]

    return stock_df


def simple_indicators(dataset: pd.DataFrame) -> pd.DataFrame:
    INDICATORS = [
        # "volume_10_ema", "volume_10_std", "volume_10_sma",
        "macd", "macd_xu_macds", "macd_xd_macds",
        "boll", "high_x_boll_ub", "low_x_boll_lb",
        "rsi", "chop", "mfi"
    ]
    stock_df = StockDataFrame(dataset.copy())
    stock_df[INDICATORS]

    return stock_df
