import numpy as np
import pandas as pd
import json
import stockstats
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pickle
import codecs


def apply_indicators(dataset):
    K = [6, 14, 26, 42]
    columns = ['open', 'close', 'volume']
    
    INDICATORS_K = [[
        f'rsi_{k}', f'stochrsi_{k}', f'vr_{k}',
        f'wr_{k}', f'cci_{k}', f'atr_{k}', f'vwma_{k}', f'chop_{k}',
        f'macd_{k}_ema', f'macds_{k}_ema', f'macdh_{k}_ema', 
        f'macd_{k}_mstd', f'macds_{k}_mstd', f'macdh_{k}_mstd', 
        f'mfi_{k}', f'kdjk_{k}', f'kdjd_{k}', f'kdjj_{k}', 
    ] for k in K]
    
    INDICATORS_COLUMNS = [[
        f'{column}_{k}_ema', f'{column}_{k}_mstd', 
        f'{column}_{k}_smma', f'{column}_{k}_sma', 
        f'{column}_{k}_mvar', 
        f'{column}_{k}_trix', f'{column}_{k}_tema',
    ] for column in columns for k in K]
    
    INDICATORS = sum(INDICATORS_K, []) + sum(INDICATORS_COLUMNS, []) + [
        'wt1', 'wt2', 'dma',
        'supertrend', 'supertrend_ub', 'supertrend_lb',
        'pdi', 'mdi', 'dx', 'adx', 'adxr',
        'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3',
        'boll', 'boll_ub', 'boll_lb',
        'macd', 'macds', 'macdh',
        'ppo', 'ppos', 'ppoh',
        'volume', 'open', 'close', 'high', 'low',
    ]
    # DROP_COLUMNS = ["t", "n", "vw"]
    # RENAME_COLUMNS = {"c": "close", "h": "high", "l": "low", "o": "open", "v": "volume"}

    result_dataset = dataset.copy()
    
    # result_dataset.drop(DROP_COLUMNS, axis=1, inplace=True)
    # result_dataset.reset_index(drop=True, inplace=True)
    # result_dataset.rename(RENAME_COLUMNS, axis=1, inplace=True)

    # result_dataset = StockDataFrame(result_dataset)
    result_dataset = stockstats.StockDataFrame(result_dataset)
    result_dataset = result_dataset[INDICATORS]
    result_dataset = result_dataset.join(result_dataset.ewm(alpha=0.3, min_periods=10).mean(), rsuffix="_ewm")
    result_dataset = result_dataset.fillna(method="ffill").dropna() \
                                   .replace([np.inf, -np.inf], method='ffill') \
                                   .resample('1min').ffill()

    return result_dataset
    
    
def get_data(indicators, scaler):
    X = indicators.copy()
    values = [X.index.month, X.index.dayofweek, X.index.hour, X.index.minute // 5]
    names = ['month', 'dayofweek', 'hour', '5minute']
    n_uniques = [12, 7, 24, 60 // 5]
    
    for value, name, n_unique in zip(values, names, n_uniques):
        onehot_enc = np.zeros((X.shape[0], n_unique))
        onehot_enc[np.arange(X.shape[0]), value - 1] = 1
        
        columns = [f'time_{name}_{i}' for i in range(n_unique)]
        X = pd.concat([X, 
                       pd.DataFrame(onehot_enc, 
                                    columns=columns,
                                    index=X.index,
                                    dtype='int64')], 
                      axis=1)
    
    TIMES = ['time_month_0', 'time_month_1', 'time_month_2', 'time_month_3', 
             'time_month_4', 'time_month_5', 'time_month_6', 'time_month_7', 
             'time_month_8', 'time_month_9', 'time_month_10', 'time_month_11', 
             'time_dayofweek_0', 'time_dayofweek_1', 'time_dayofweek_2', 
             'time_dayofweek_3', 'time_dayofweek_4', 'time_dayofweek_5', 
             'time_dayofweek_6', 
             'time_hour_0', 'time_hour_1', 'time_hour_2', 'time_hour_3', 
             'time_hour_4', 'time_hour_5', 'time_hour_6', 'time_hour_7', 
             'time_hour_8', 'time_hour_9', 'time_hour_10', 'time_hour_11', 
             'time_hour_12', 'time_hour_13', 'time_hour_14', 'time_hour_15', 
             'time_hour_16', 'time_hour_17', 'time_hour_18', 'time_hour_19', 
             'time_hour_20', 'time_hour_21', 'time_hour_22', 'time_hour_23', 
             'time_5minute_1', 'time_5minute_2', 'time_5minute_5', 
             'time_5minute_8', 'time_5minute_11']

    assert (len(indicators.columns) + len(TIMES)) == 418, 'number columns != 418'
    
    X = pd.concat(
        [
            pd.DataFrame(scaler.transform(X[indicators.columns]),
                         index=X.index,
                         columns=indicators.columns), 
            X[TIMES]
        ],
        axis=1
    )

    return X
    

def load_state(model):
    load_data = torch.load(MODELS_DIR + name + '.data', map_location=torch.device('cpu'))

    epoch = load_data['epoch'] - 1

    history = defaultdict(lambda: defaultdict(list))
    for k, v in load_data['history'].items():
        for k2, v2 in v.items():
            history[k][k2] = v2

    model.load_state_dict(load_data['model_state_dict'])
    optimizer.load_state_dict(load_data['optimizer_state_dict'])
    
    return model, optimizer, epoch, history
    
    
class ModelSimpleSmall(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.5):
        super(ModelSimpleSmall, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_first=True,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, X, news=None, len_news=None, titles=None, len_titles=None):
        # X.shape = (BATCH_SIZE, WINDOW_SIZE, N_FEATURES)
        
        out, (hidden, _) = self.lstm(X)
        # out.shape = (BATCH_SIZE, WINDOW_SIZE, HIDDEN_SIZE)
        # hidden.shape = (N_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
        
        output = self.fc1(self.dropout(hidden[-1]))
        # output = self.norm(output)
        output = self.fc2(F.relu(output))
        return output
    
    
class IndicatorsNewsDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, shifts=4*12, *, news_dates=None, news=None, 
                 titles=None, vocab=None):
        self.y = None
        if y is not None:
            self.y = y.astype(np.float32)
        self.X = X.astype(np.float32)
        
        if isinstance(shifts, int):
            self.shifts = np.arange(shifts)
        else:
            self.shifts = np.array(shifts).sort()
        self.shifts = self.shifts[::-1]
        self.shift = self.shifts.max()
        
        self.news, self.len_news = pd.DataFrame(), pd.DataFrame()
        self.titles, self.len_titles = pd.DataFrame(), pd.DataFrame()
        
        if news is not None and vocab is not None and news_dates is not None and titles is not None:
            self.len_news = pd.DataFrame([len(text) for text in news], 
                                         index=news_dates)
            self.news = pd.DataFrame([vocab(text) + [0] * (MAX_LEN - len(text)) for text in news], 
                                     index=news_dates)
            
            self.len_titles = pd.DataFrame([len(title) for title in titles], 
                                           index=news_dates)
            self.titles = pd.DataFrame([vocab(title) + [0] * (MAX_TITLE_LEN - len(title)) for title in titles], 
                                       index=news_dates)

    def __len__(self):
        return self.X.shape[0] - self.shift

    def __getitem__(self, index):
        if index > 0:
            begin_news = self.X.index[index - 1]
            end_news = self.X.index[index]
            index_news = (begin_news < self.news.index) & (self.news.index <= end_news)
        else:
            index_news = []
            
        return np.array(self.X.iloc[index + self.shift - self.shifts]), \
               np.array(self.y.iloc[index + self.shift] if self.y is not None else np.array([])), \
               np.array(self.news[index_news]), np.array(self.len_news[index_news]), \
               np.array(self.titles[index_news]), np.array(self.len_titles[index_news]) 

    
class Model:
    def __init__(self):
        self.load_model()
    
    def load_model(self):
        qb = QuantBook()
        model_file = qb.Download('https://drive.google.com/uc?export=download&id=1eR1Lgc5XOZwK9f4fhYiMY7ClvKsUB-KG')
        state_dict = pickle.loads(codecs.decode(model_file.encode(), "base64"))
        
        model = ModelSimpleSmall(input_size=418, hidden_size=64, num_layers=1, dropout=0.3)
        model.load_state_dict(state_dict)
        
        self.model = model
        
        scaler_str = qb.Download('https://drive.google.com/uc?export=download&id=15FRJD5YgNNeln3bDbRrAKOx0OudGfJs1')
        scaler = pickle.loads(codecs.decode(scaler_str.encode(), "base64"))
        self.scaler = scaler
        
    def predict(self, data):
        try:
            # print('start_predict')
            
            indicators = apply_indicators(data)
            
            X = get_data(indicators, self.scaler).resample('15min').first()
    
            test_dataset = IndicatorsNewsDataset(X, shifts=4*12)
            
            input_data = torch.tensor(test_dataset[-1][0]).unsqueeze(0)
            pred = self.model(input_data)[0][0].item() / 100
            
            # print('end_predict')
            return pred
        except Exception:
            return 0

