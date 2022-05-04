# QuantBook Analysis Tool 
# For more information see [https://www.quantconnect.com/docs/research/overview]

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
import json
import stockstats

def apply_indicators(dataset):
    INDICATORS = [
        "volume_26_ema", "volume_42_ema", "volume_86_ema", "volume_174_ema", "volume_363_ema", 
        "volume_42_smma",
        "volume_14_mstd", "volume_26_mstd", "volume_42_mstd", "volume_86_mstd", "volume_174_mstd", 
        "high_42_mstd", "high_86_mstd", "high_174_mstd", 
        "low_42_mstd", "low_86_mstd", "low_174_mstd", 
        "close_42_mstd", "close_86_mstd", "close_174_mstd", "close_363_mstd",
        "vr_26", "vr_86", "vr_174", "vr_363",
        "atr_6", "atr_14",
        "cr", "cr-ma1", "cr-ma2", "cr-ma3",
        "chop_14", "chop_42", "chop_174", "chop_363",
        "mfi_42", "mfi_174", 
        'rsi_42', 'rsi_86', 'rsi_174', 
        'wt2', 
        'close_12_trix', 'close_174_trix', 'close_363_trix',  
        'volume_12_trix', 'volume_174_trix', 'volume_363_trix', 
        'close_12_tema', 'close_174_tema', 'close_363_tema',  
        'volume_12_tema', 'volume_174_tema', 'volume_363_tema',  
        'boll', 'boll_ub', 'boll_lb', 'macdh', 'ppo',
        'vwma', 'vwma_174', 'vwma_363',
        'close_10_kama', 'close_174_kama',
        'supertrend', 'supertrend_ub', 'supertrend_lb'
    ]
    # DROP_COLUMNS = ["t", "n", "vw"]
    # RENAME_COLUMNS = {"c": "close", "h": "high", "l": "low", "o": "open", "v": "volume"}

    result_dataset = dataset.copy()
    
    # result_dataset.drop(DROP_COLUMNS, axis=1, inplace=True)
    # result_dataset.reset_index(drop=True, inplace=True)
    # result_dataset.rename(RENAME_COLUMNS, axis=1, inplace=True)

    # result_dataset = StockDataFrame(result_dataset)
    result_dataset = stockstats.StockDataFrame(result_dataset)
    
    result_dataset['boll_dif'] = result_dataset['boll_ub'] - result_dataset['boll_lb']
    # result_dataset['pdi_mdi'] = result_dataset['pdi'] - result_dataset['mdi']
    result_dataset['boll_ub_close'] = result_dataset['boll_ub'] - result_dataset['close']
    result_dataset['boll_lb_close'] = result_dataset['close'] - result_dataset['boll_lb']

    # result_dataset = result_dataset.join(result_dataset.ewm(alpha=ewm_alpha, min_periods=ewm_period).mean(), rsuffix="_ewm")
    INDICATORS.extend(['close', 'high', 'low', 'open', 'volume', 'boll_dif', 'boll_ub_close', 'boll_lb_close'])
    result_dataset = result_dataset[INDICATORS]

    return result_dataset
    
class Model:
    def __init__(self):
        self.load_model()
    
    def load_model(self):
        qb = QuantBook()
        # model_str = qb.Download('https://drive.google.com/uc?export=download&id=1kR7_CG7glpD3H7DjeOhzoD6eiAEItqy5')
        # model_str = qb.Download('https://drive.google.com/uc?export=download&id=13EwMaSOT-LrWxFdTQhXMjxulmSgUodkV')
        model_str = qb.Download('https://drive.google.com/uc?export=download&id=1Gq-9w4LDyEy-fRReJ_ImxojsrCU9STmZ')
        self.model = lgb.Booster(model_str=model_str)
        
    def predict_valotile(self, data):
        data = apply_indicators(data)
        return self.model.predict(data[-1:])
