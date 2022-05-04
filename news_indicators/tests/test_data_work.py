import numpy as np

import pytest

from src.paths import PRICES_DIR
from src.prepare_data import get_prices, prepare_features, read_data, get_resampled_data, get_dataloaders

DEBUG = False


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("currency", ['BTC', 'ETH', 'LTC'])
def test_initial_data_valid(currency):
    prices = get_prices([currency], PRICES_DIR)[currency]
    numeric_columns = ['open', 'close', 'high', 'low', f'Volume {currency}', 'Volume USD']

    assert not (prices.index < '2000-01-01').any()
    assert not (prices.index > '2023-01-01').any()
    assert not (prices[numeric_columns] < 0).any().any()
    assert not prices.isna().any().any()

    assert (prices['symbol'] == f'{currency}/USD').all()

    assert (prices.dtypes[numeric_columns] == np.float64).all()

    # check index (dates) not string
    assert prices.index.dtype != object


CASES = [('LTC', 15), ('BTC', 5), ('ETH', 30)][:2]
SHIFTS = [5, 10, 15, 20, 30, 60, 120, 240]


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("currency, period", CASES)
def test_data_simple_preprocessing(currency, period, tmp_path):
    prepare_features(currency, SHIFTS, PRICES_DIR, PRICES_DIR, start_date='2021-01-01', resample_period=f'{period}min')

    features, targets = read_data(PRICES_DIR, f'X_data_{period}min.feather', f'y_data_{period}min.feather')

    assert features.shape[0] == targets.shape[0]
    assert (features.index == targets.index).all()
    assert targets.shape[1] == len(SHIFTS)

    assert not features.isna().any().any()
    assert not targets.isna().any().any()

    assert not features.isin([np.inf, -np.inf]).any().any()
    assert not targets.isin([np.inf, -np.inf]).any().any()

    assert len(features) == len(features.resample(f'{period}min').ffill())
    assert (targets >= -1).all().all()


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("currency, period", CASES)
def test_data_preprocessing_split(currency, period, tmp_path):
    start_date = '2021-01-01'
    prepare_features(currency, SHIFTS, PRICES_DIR, PRICES_DIR, start_date=start_date, resample_period=f'{period}min')

    features_train, features_test, targets_train, targets_test = \
        get_resampled_data(data_dir=PRICES_DIR, period=f'{period}min', train_start_date=start_date,
                           train_end_date='2021-10-31 23:59:59', test_end_date='2022-02-01 00:00:00',
                           coef=100, file_period=f'{period}min')

    assert (features_train.index == targets_train.index).all()
    assert (features_test.index == targets_test.index).all()
    assert targets_train.shape[1] == 1 and targets_test.shape[1] == 1

    assert not features_train.isna().any().any()
    assert not features_test.isna().any().any()
    assert not targets_train.isna().any().any()
    assert not targets_test.isna().any().any()

    assert not features_train.isin([np.inf, -np.inf]).any().any()
    assert not features_test.isin([np.inf, -np.inf]).any().any()
    assert not targets_train.isin([np.inf, -np.inf]).any().any()
    assert not targets_test.isin([np.inf, -np.inf]).any().any()

    assert (targets_train >= -100).all().all()
    assert (targets_test >= -100).all().all()

    assert features_train.index.max() < features_test.index.min()


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("currency, period", CASES)
def test_get_data_loaders(currency, period, tmp_path):
    start_date = '2021-01-01'
    prepare_features(currency, SHIFTS, PRICES_DIR, PRICES_DIR, start_date=start_date, resample_period=f'{period}min')

    features_train, features_test, targets_train, targets_test = \
        get_resampled_data(data_dir=PRICES_DIR, period=f'{period}min', train_start_date=start_date,
                           train_end_date='2021-10-31 23:59:59', test_end_date='2022-02-01 00:00:00',
                           coef=100, file_period=f'{period}min')

    shifts_count = 42
    batch_size = 512
    train_dataset, test_dataset, train_dataloader, test_dataloader = \
        get_dataloaders(features_train, features_test, targets_train, targets_test,
                        shifts=shifts_count, batch_size=batch_size)

    assert len(train_dataset) == len(features_train) - shifts_count + 1
    assert len(test_dataset) == len(features_test) - shifts_count + 1

    assert len(train_dataset[0]) == 6
    assert len(test_dataset[0]) == 6

    assert train_dataset[0][0].shape == (shifts_count, features_train.shape[1])
    assert train_dataset[0][1].shape == (1, )

    assert test_dataset[0][0].shape == (shifts_count, features_test.shape[1])
    assert test_dataset[0][1].shape == (1, )

    train_batch = next(iter(train_dataloader))
    assert len(train_batch) == 6
    assert train_batch[0].shape == (batch_size, shifts_count, features_train.shape[1])
    assert train_batch[1].shape == (batch_size, 1)

    test_batch = next(iter(test_dataloader))
    assert len(test_batch) == 6
    assert test_batch[0].shape == (batch_size, shifts_count, features_test.shape[1])
    assert test_batch[1].shape == (batch_size, 1)

    assert len(train_dataloader) == pytest.approx(np.ceil(len(train_dataset) / batch_size))
    assert len(test_dataloader) == pytest.approx(np.ceil(len(test_dataset) / batch_size))
