import numpy as np
import pandas as pd

from stockstats import StockDataFrame
import warnings

from sklearn.preprocessing import StandardScaler

import torch
import torchtext
from torch.utils.data import DataLoader, Dataset

from .paths import NEWS_DIR

MAX_LEN = 224
MAX_TITLE_LEN = 24


def get_prices(tickers, prices_dir):
    return {ticker: pd.read_csv(prices_dir/f'{ticker}.csv', parse_dates=['date'], index_col='date')
            for ticker in tickers}


def get_indicator_names(ks=None, columns=None):
    if ks is None:
        ks = [6, 14, 26, 42]
    if columns is None:
        columns = ['open', 'close', 'volume']

    indicators_k = [[
        f'rsi_{k}', f'stochrsi_{k}', f'vr_{k}',
        f'wr_{k}', f'cci_{k}', f'atr_{k}', f'vwma_{k}', f'chop_{k}',
        f'macd_{k}_ema', f'macds_{k}_ema', f'macdh_{k}_ema',
        f'macd_{k}_mstd', f'macds_{k}_mstd', f'macdh_{k}_mstd',
        f'mfi_{k}', f'kdjk_{k}', f'kdjd_{k}', f'kdjj_{k}',
    ] for k in ks]

    indicators_columns = [[
        f'{column}_{k}_ema', f'{column}_{k}_mstd',
        f'{column}_{k}_smma', f'{column}_{k}_sma',
        f'{column}_{k}_mvar',
        f'{column}_{k}_trix', f'{column}_{k}_tema',
    ] for column in columns for k in ks]

    indicators = sum(indicators_k, []) + sum(indicators_columns, []) + [
        'wt1', 'wt2', 'dma',
        'supertrend', 'supertrend_ub', 'supertrend_lb',
        'pdi', 'mdi', 'dx', 'adx', 'adxr',
        'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3',
        'boll', 'boll_ub', 'boll_lb',
        'macd', 'macds', 'macdh',
        'ppo', 'ppos', 'ppoh',
        'volume', 'open', 'close', 'high', 'low',
    ]

    return indicators


def calculate_indicators(prices, indicator_names, filename=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result_dataset = prices.drop_duplicates().copy()
        result_dataset = StockDataFrame(result_dataset)
        result_dataset = result_dataset[indicator_names]
        result_dataset = result_dataset.join(result_dataset.ewm(alpha=0.3, min_periods=10).mean(), rsuffix="_ewm")
        result_dataset = result_dataset.fillna(method="ffill").dropna() \
                                       .replace([np.inf, -np.inf], method='ffill') \
                                       .resample('1min').ffill()

    if filename is not None:
        result_dataset.reset_index().to_feather(filename)
    return result_dataset


def encode_date(dates):
    values = [dates.month, dates.dayofweek, dates.hour, dates.minute // 5]
    names = ['month', 'dayofweek', 'hour', '5minute']
    n_uniques = [12, 7, 24, 60 // 5]

    result = pd.DataFrame()

    for value, name, n_unique in zip(values, names, n_uniques):
        onehot_enc = np.zeros((dates.shape[0], n_unique))
        onehot_enc[np.arange(dates.shape[0]), value - 1] = 1

        columns = [f'time_{name}_{i}' for i in range(n_unique)]
        result = pd.concat([result,
                            pd.DataFrame(onehot_enc, columns=columns, index=dates, dtype='int64')],
                           axis=1)

    return result


def calculate_returns(open_prices, shifts):
    minute = pd.Timedelta('1min')
    dates = open_prices.index[:-max(shifts)]

    y = pd.DataFrame()

    current_price = open_prices.loc[dates].reset_index()['open']
    for shift in shifts:
        shift_price = open_prices.loc[dates + shift * minute].reset_index()['open']
        y[f'{shift}min'] = (shift_price - current_price) / current_price

    y.index = dates
    return y


def prepare_features(ticker, shifts, prices_dir, output_dir, start_date='2017-06-01', suffix_filename='',
                     resample_period=None):
    prices = get_prices([ticker], prices_dir)[ticker].loc[start_date:]

    prices.drop(["unix", "symbol", "Volume USD"], axis=1, inplace=True)
    prices.rename({f"Volume {ticker}": "volume"}, axis=1, inplace=True)

    indicator_names = get_indicator_names()
    indicators = calculate_indicators(prices, indicator_names, filename=output_dir/f'{ticker}_indicators.feather')
    encoded_date = encode_date(indicators.index)

    features = pd.concat([indicators, encoded_date], axis=1)
    target = calculate_returns(features['open'], shifts=shifts)
    features = features.loc[target.index]

    features.reset_index().to_feather(output_dir/f'X_data{suffix_filename}.feather')
    target.reset_index().to_feather(output_dir/f'y_data{suffix_filename}.feather')

    if resample_period is not None:
        features_period = features.resample(resample_period).first()
        target_period = target.resample(resample_period).first()

        features_period.reset_index().to_feather(output_dir/f'X_data_{resample_period}{suffix_filename}.feather')
        target_period.reset_index().to_feather(output_dir/f'y_data_{resample_period}{suffix_filename}.feather')


class IndicatorsNewsDataset(Dataset):
    def __init__(self, X, y, shifts=4 * 12, *, news=None, vocab=None, max_len=MAX_LEN, max_title_len=MAX_TITLE_LEN):
        self.y = y.astype(np.float32)
        self.X = X.astype(np.float32)

        if isinstance(shifts, int):
            self.shifts = np.arange(shifts)
        else:
            self.shifts = shifts
            np.array(self.shifts).sort()
        self.shifts = self.shifts[::-1]
        self.shift = self.shifts.max()

        self.news, self.len_news = pd.DataFrame(), pd.DataFrame()
        self.titles, self.len_titles = pd.DataFrame(), pd.DataFrame()
        self.bos = 2
        self.eos = 3
        self.default_news = [self.bos, self.eos] + [0] * (max_len - 2)
        self.default_title = [self.bos, self.eos] + [0] * (max_title_len - 2)

        if news is not None and vocab is not None:
            self.bos = vocab['<bos>']
            self.eos = vocab['<eos>']

            self.len_news = pd.Series(
                [len(text) for text in news['body']],
                index=news.index
            )
            self.news = pd.Series(
                [vocab(text.tolist()) + [0] * (max_len - len(text)) for text in news['body']],
                index=news.index
            )

            self.len_titles = pd.Series(
                [len(title) for title in news['title']],
                index=news.index
            )
            self.titles = pd.Series(
                [vocab(title.tolist()) + [0] * (max_title_len - len(title)) for title in news['title']],
                index=news.index
            )

    def __len__(self):
        return self.X.shape[0] - self.shift

    def __getitem__(self, index):
        begin_news = self.X.index[index - 1] if index > 0 else self.news.index[0]
        end_news = self.X.index[index]
        index_news = (begin_news < self.news.index) & (self.news.index <= end_news)

        if index_news.sum() != 0:
            return np.array(self.X.iloc[index + self.shift - self.shifts]), \
                   np.array(self.y.iloc[index + self.shift]), \
                   np.array(self.news[index_news].tolist()), np.array(self.len_news[index_news]), \
                   np.array(self.titles[index_news].tolist()), np.array(self.len_titles[index_news])
        else:
            return np.array(self.X.iloc[index + self.shift - self.shifts]), \
                   np.array(self.y.iloc[index + self.shift]), \
                   np.array([self.default_news]), np.array([2]), \
                   np.array([self.default_title]), np.array([2])


def collate_fn(batch):
    X_batch, y_batch, news, len_news, titles, len_titles = list(zip(*batch))

    return torch.tensor(np.array(X_batch)), torch.tensor(np.array(y_batch)), \
           news, len_news, titles, len_titles


def read_data(data_dir, features_name, targets_name):
    features = pd.read_feather(data_dir / features_name).set_index('date')
    targets = pd.read_feather(data_dir / targets_name).set_index('date')
    return features, targets


def resample_data(features, targets, period, train_start_date, train_end_date, test_end_date, coef):
    features = features.resample(period).first()
    features.drop(columns=features.columns[(features != 0).sum() == 0], inplace=True)
    targets = targets[[period]].resample(period).first()

    features_train = features.loc[train_start_date:train_end_date]
    features_test = features.loc[train_end_date:test_end_date]

    targets_train = targets.loc[train_start_date:train_end_date] * coef
    targets_test = targets.loc[train_end_date:test_end_date] * coef

    return features_train, features_test, targets_train, targets_test


def scale_data(features_train, features_test, times_columns, indicators_columns, only_indicators=True):
    scaler = StandardScaler()

    features_train = pd.concat(
        [
            pd.DataFrame(scaler.fit_transform(features_train[indicators_columns]),
                         index=features_train.index,
                         columns=features_train[indicators_columns].columns),
            features_train[times_columns]
        ],
        axis=1
    )
    features_test = pd.concat(
        [
            pd.DataFrame(scaler.fit_transform(features_test[indicators_columns]),
                         index=features_test.index,
                         columns=features_test[indicators_columns].columns),
            features_test[times_columns]
        ],
        axis=1
    )

    if only_indicators:
        return features_train[indicators_columns], features_test[indicators_columns]
    else:
        return features_train, features_test


def get_resampled_data(data_dir, period, train_start_date, train_end_date, test_end_date,
                       coef=100, file_period=None, suffix_filename=''):
    suffix = suffix_filename
    if file_period is not None:
        suffix = f'_{file_period}' + suffix
    features, targets = read_data(data_dir, f'X_data{suffix}.feather', f'y_data{suffix}.feather')

    return resample_data(features, targets, period, train_start_date, train_end_date, test_end_date, coef)


def get_dataloaders(features_train, features_test, targets_train, targets_test, shifts=2*12, batch_size=256):
    times_columns = [c for c in features_train.columns if c.startswith('time_')]
    indicators_columns = [c for c in features_train.columns if not c.startswith('time_')]

    features_train, features_test = scale_data(features_train, features_test, times_columns, indicators_columns)

    news_train = pd.read_feather(NEWS_DIR / 'news_train.feather').set_index('date')
    news_test = pd.read_feather(NEWS_DIR / 'news_test.feather').set_index('date')

    vocab = torchtext.vocab.build_vocab_from_iterator(news_train['body'], specials=['<pad>', '<unk>', '<bos>', '<eos>'],
                                                      min_freq=15)
    vocab.set_default_index(vocab['<unk>'])

    train_dataset = IndicatorsNewsDataset(features_train, targets_train, shifts=shifts, news=news_train, vocab=vocab)
    test_dataset = IndicatorsNewsDataset(features_test, targets_test, shifts=shifts, news=news_test, vocab=vocab)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)

    return train_dataset, test_dataset, train_dataloader, test_dataloader, vocab


def get_buy_and_hold_returns(features, dataset):
    buy_and_hold = features['open'].iloc[-len(dataset) - 1:]
    buy_and_hold = (buy_and_hold / buy_and_hold.iloc[0] - 1)[1:]
    return buy_and_hold
