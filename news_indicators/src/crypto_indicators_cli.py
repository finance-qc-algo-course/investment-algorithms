import string
import datetime
import warnings
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
import plotly

from tqdm.auto import tqdm

import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from stockstats import StockDataFrame

import click

from .prepare_data import prepare_features, get_resampled_data, get_dataloaders, get_buy_and_hold_returns
from .paths import PRICES_DIR, MODELS_DIR, PLOTS_DIR
from .torch_utils import train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option('--input', '-i', 'input_dir', default=PRICES_DIR, help='Path to input dir (with prices)')
@click.option('--output', '-o', 'output_dir', default=PRICES_DIR, help='Path to output dir')
@click.option('--ticker', '-t', 'ticker', default=PRICES_DIR, help='Crypto ticker')
@click.option('--period', '-p', 'period', default='5min', help='Resample period')
def prepare_data(input_dir, output_dir, ticker, period):
    prepare_features(ticker, shifts=[5, 10, 15, 20, 30, 45, 60, 90, 2 * 60, 3 * 60, 4 * 60, 6 * 60],
                     prices_dir=input_dir, output_dir=output_dir, resample_period=period)


@cli.command()
@click.option('--input', '-i', 'input_dir', default=PRICES_DIR, help='Path to input dir (with features files)')
@click.option('--output', '-o', 'output_model', default=MODELS_DIR, help='Path to output model file')
@click.option('--period', '-p', 'period', default='15min', help='Resample period')
@click.option('--file_period', '-fp', 'file_period', default=None, help='Resample period')
@click.option('--train_start', '-train', 'train_start_date', default='2017-06-01', help='Train period start')
@click.option('--train_end', '-train', 'train_end_date', default='2021-07-31 23:59:59', help='Train period end')
@click.option('--test_end', '-test', 'test_end_date', default='2022-01-31 23:59:59', help='Test period end')
@click.option('--shifts_count', '-sh', 'shifts_count', default=4*12, help='Count previous samples')
@click.option('--batch_size', '-bs', 'batch_size', default=256, help='Batch size')
@click.option('--model_name', '-m', 'model_name', help='Training model name')
@click.option('--fee', '-fee', 'fee', default=0.001, help='Fee')
@click.option('--threshold', '-tsh', 'threshold', default=0.0005, help='Threshold')
@click.option('--num_epochs', '-n_epochs', 'num_epochs', default=25, help='Number epochs')
def train_model(input_dir, output_model, period, file_period, train_start_date, train_end_date, test_end_date,
                shifts_count, batch_size, model_name, fee, threshold, num_epochs):
    features_train, features_test, targets_train, targets_test = \
        get_resampled_data(data_dir=input_dir, period=period, train_start_date=train_start_date,
                           train_end_date=train_end_date, test_end_date=test_end_date,
                           coef=100, file_period=file_period)

    train_dataset, test_dataset, train_dataloader, test_dataloader = \
        get_dataloaders(features_train, features_test, targets_train, targets_test,
                        shifts=shifts_count, batch_size=batch_size)

    buy_and_hold = get_buy_and_hold_returns(features_test, test_dataset)
    input_size = train_dataset.shape[-1]

    ModelClass = getattr(importlib.import_module(".torch_models"), model_name)
    model = ModelClass(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.2)

    model, history = train(
        model,
        criterion,
        optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        # scheduler=scheduler,
        num_epochs=num_epochs,
        threshold=threshold,
        fee=fee,
        buy_and_hold=buy_and_hold,
        model_dir=output_model,
        plots_dir=PLOTS_DIR,
        save_name=model_name,
        step_plotting=1,
    )


if __name__ == "__main__":
    cli()
