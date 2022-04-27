import os

import numpy as np

import pytest
import torch
from torch import nn

from src.paths import PRICES_DIR, MODELS_DIR, PLOTS_DIR
from src.prepare_data import get_resampled_data, get_dataloaders, get_buy_and_hold_returns
from src.torch_models import ModelSimple, ModelLSTM_CNN
from src.torch_utils import train, load_state, predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS = [ModelSimple, ModelLSTM_CNN]

DEBUG = False


@pytest.fixture(scope="module")
def data():
    features_train, features_test, targets_train, targets_test = \
        get_resampled_data(data_dir=PRICES_DIR, period='15min', train_start_date='2021-04-01',
                           train_end_date='2021-10-31 23:59:59', test_end_date='2022-01-31 00:00:00',
                           coef=100, file_period='15min')

    train_dataset, test_dataset, train_dataloader, test_dataloader = \
        get_dataloaders(features_train, features_test, targets_train, targets_test,
                        shifts=42, batch_size=256)

    buy_and_hold = get_buy_and_hold_returns(features_test, test_dataset)
    input_size = features_train.shape[-1]

    yield train_dataloader, test_dataloader, buy_and_hold, input_size


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("model", MODELS)
def test_train_artifacts(model, data, tmp_path):
    train_dataloader, test_dataloader, buy_and_hold, input_size = data

    model_obj = model(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    model_name = 'test_model'

    _, history = train(
        model_obj,
        criterion,
        optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        num_epochs=1,
        buy_and_hold=buy_and_hold,
        model_dir=tmp_path,
        plots_dir=tmp_path,
        save_name=model_name,
        step_plotting=1,
    )

    assert f'{model_name}_last.data' in os.listdir(tmp_path)
    assert f'{model_name}_loss.data' in os.listdir(tmp_path)
    assert f'{model_name}_mae.data' in os.listdir(tmp_path)
    assert f'{model_name}_profit.data' in os.listdir(tmp_path)
    assert f'{model_name}_profit_ths.data' in os.listdir(tmp_path)
    assert f'{model_name}_profit_fee.data' in os.listdir(tmp_path)
    assert f'{model_name}_profit_fee_ths.data' in os.listdir(tmp_path)

    assert f'{model_name}' in os.listdir(tmp_path)
    assert 'profits_1epoch.png' in os.listdir(tmp_path / model_name)
    assert 'metrics_1epoch.png' in os.listdir(tmp_path / model_name)


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("model", MODELS)
def test_metric_correctness(model, data):
    train_dataloader, test_dataloader, buy_and_hold, input_size = data
    num_epochs = 5

    model_obj = model(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    _, history = train(
        model_obj,
        criterion,
        optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        num_epochs=num_epochs,
        buy_and_hold=buy_and_hold,
        step_plotting=100,
    )

    assert len(history) == 6

    for metric_name in history:
        assert len(history[metric_name]) == 2
        for data_name in history[metric_name]:
            assert len(history[metric_name][data_name]) == num_epochs

            assert float('nan') not in history[metric_name][data_name]
            assert float('inf') not in history[metric_name][data_name]
            assert -float('inf') not in history[metric_name][data_name]


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("model", MODELS)
def test_model_over_fit_small_data(model, tmp_path):
    features_train, features_test, targets_train, targets_test = \
        get_resampled_data(data_dir=PRICES_DIR, period='15min', train_start_date='2021-08-01 00:00:00',
                           train_end_date='2021-08-01 22:00:00', test_end_date='2021-08-02 12:00:00',
                           coef=100, file_period='15min')

    train_dataset, test_dataset, train_dataloader, test_dataloader = \
        get_dataloaders(features_train, features_test, targets_train, targets_test,
                        shifts=42, batch_size=16)

    buy_and_hold = get_buy_and_hold_returns(features_test, test_dataset)
    input_size = features_train.shape[-1]

    model_obj = model(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    _, history = train(
        model_obj,
        criterion,
        optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        num_epochs=50,
        buy_and_hold=buy_and_hold,
        step_plotting=100,
    )

    assert history['loss']['train'][-1] / history['loss']['val'][-1] < 1/5


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("model", MODELS)
def test_loss_decrease(model, data):
    train_dataloader, test_dataloader, buy_and_hold, input_size = data
    num_epochs = 5

    model_obj = model(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    _, history = train(
        model_obj,
        criterion,
        optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        num_epochs=num_epochs,
        buy_and_hold=buy_and_hold,
        step_plotting=100,
    )

    assert history['loss']['train'][0] > history['loss']['train'][num_epochs - 1]
    assert history['loss']['val'][0] > history['loss']['val'][num_epochs - 1]


MODEL_NAMES = ['ModelSimple1', 'Model_LSTM_CNN']


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("model, model_name", zip(MODELS, MODEL_NAMES))
def test_correctness_trained_model(model, model_name, data):
    train_dataloader, test_dataloader, buy_and_hold, input_size = data

    model_obj = model(input_size=input_size, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=1e-3)

    model_obj, optimizer, epoch, history = load_state(model_obj, optimizer, MODELS_DIR, f'{model_name}_profit_fee')

    assert epoch > 10

    assert len(history) == 6
    for metric_name in history:
        assert len(history[metric_name]) == 2
        for data_name in history[metric_name]:
            assert len(history[metric_name][data_name]) == epoch

            assert float('nan') not in history[metric_name][data_name]
            assert float('inf') not in history[metric_name][data_name]
            assert -float('inf') not in history[metric_name][data_name]


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("model, model_name", zip(MODELS, MODEL_NAMES))
def test_correctness_predict_artifacts(model, model_name, data, tmp_path, capsys):
    train_dataloader, test_dataloader, buy_and_hold, input_size = data

    model_obj = model(input_size=input_size, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model_obj, optimizer, epoch, history = load_state(model_obj, optimizer, MODELS_DIR, f'{model_name}_profit_fee')

    loss, mae, profit = predict(
        model_obj,
        criterion,
        val_dataloader=test_dataloader,
        threshold=0.0005,
        fee=0.001,
        buy_and_hold=buy_and_hold,
        short=True,
        long=True,
        plots_dir=tmp_path,
        save_name='test_predict'
    )

    assert 'test_predict' in os.listdir(tmp_path)
    assert 'profits_final.png' in os.listdir(tmp_path / 'test_predict')

    captured = capsys.readouterr()
    output = captured.out

    assert 'val loss:' in output
    assert f'{np.floor(loss*100)/100:.2f}' in output
    assert 'val mae:' in output
    assert f'{np.floor(mae*100)/100:.2f}' in output

    assert 'val profit:' in output
    assert f'{np.floor(np.abs(profit["profit"]*100)):.0f}' in output
    assert 'val profit ths:' in output
    assert f'{np.floor(np.abs(profit["profit_threshold"]*100)):.0f}' in output

    assert 'val profit fee:' in output
    assert f'{np.floor(np.abs(profit["profit_fee"]*100)):.0f}' in output
    assert 'val profit fee ths:' in output
    assert f'{np.floor(np.abs(profit["profit_fee_threshold"]*100)):.0f}' in output


MODEL_SUFFIXES = ['profit', 'profit_fee', 'profit_ths', 'profit_fee_ths']


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("model, model_name", zip(MODELS, MODEL_NAMES))
@pytest.mark.parametrize("best_profit", MODEL_SUFFIXES)
def test_positive_profit_trained_model(model, model_name, best_profit, data, tmp_path):
    train_dataloader, test_dataloader, buy_and_hold, input_size = data

    model_obj = model(input_size=input_size, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model_obj, optimizer, epoch, history = load_state(model_obj, optimizer, MODELS_DIR, f'{model_name}_{best_profit}')

    loss, mae, profit = predict(
        model_obj,
        criterion,
        val_dataloader=test_dataloader,
        threshold=0.0005,
        fee=0.001,
        buy_and_hold=buy_and_hold,
        short=True,
        long=True,
        plots_dir=tmp_path,
        save_name='test_predict'
    )

    assert profit[best_profit] > 0


@pytest.mark.skipif(DEBUG, reason="long test")
@pytest.mark.parametrize("model, model_name", zip(MODELS, MODEL_NAMES))
@pytest.mark.parametrize("best_profit", MODEL_SUFFIXES)
def test_better_buy_hold_profit_trained_model(model, model_name, best_profit, data, tmp_path):
    train_dataloader, test_dataloader, buy_and_hold, input_size = data

    model_obj = model(input_size=input_size, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=5e-3)
    criterion = nn.MSELoss()

    model_obj, optimizer, epoch, history = load_state(model_obj, optimizer, MODELS_DIR, f'{model_name}_{best_profit}')

    loss, mae, profit = predict(
        model_obj,
        criterion,
        val_dataloader=test_dataloader,
        threshold=0.0005,
        fee=0.001,
        buy_and_hold=buy_and_hold,
        short=True,
        long=True,
        plots_dir=tmp_path,
        save_name='test_predict'
    )

    assert profit[best_profit] > buy_and_hold.iloc[-1]


# def test_attack_model():
#     features_train, features_test, targets_train, targets_test = \
#         get_resampled_data(data_dir=PRICES_DIR, period='15min', train_start_date='2021-01-01',
#                            train_end_date='2021-06-30 23:59:59', test_end_date='2021-08-30 00:00:00',
#                            coef=100, file_period='15min')
#
#     train_dataset, test_dataset, train_dataloader, test_dataloader = \
#         get_dataloaders(features_train, features_test, targets_train, targets_test,
#                         shifts=42, batch_size=256)
#
#     buy_and_hold = get_buy_and_hold_returns(features_test, test_dataset)
#     input_size = features_train.shape[-1]
