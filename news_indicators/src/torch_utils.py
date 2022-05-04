import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

import torch
from IPython import display
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
import time
import os
from ruamel.yaml import YAML

os.environ["WANDB_SILENT"] = "true"
import wandb
import logging


logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

with open('wandb_token') as f:
    wandb.login(key=f.read())

sns.set(style='darkgrid', font_scale=1.3, palette='Set2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_params(params_file):
    yaml = YAML(typ="safe")
    with open(params_file) as file:
        params = yaml.load(file)
    return params


def save_state(model, optimizer, epoch, history, model_dir, name, log_wandb=False):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': dict(history),
    }, model_dir / f'{name}.data')

    if log_wandb:
        artifact = wandb.Artifact(name, type='model')
        artifact.add_file(model_dir / f'{name}.data')
        wandb.log_artifact(artifact)


def load_state(model, optimizer, model_dir, name):
    load_data = torch.load(model_dir / f'{name}.data', map_location=torch.device(device))

    epoch = load_data['epoch']

    history = defaultdict(lambda: defaultdict(list))
    for k, v in load_data['history'].items():
        for k2, v2 in v.items():
            history[k][k2] = v2

    model.load_state_dict(load_data['model_state_dict'])
    optimizer.load_state_dict(load_data['optimizer_state_dict'])

    return model, optimizer, epoch, history


def plot_history(history, start=0, plots_dir=None, save_name=None):
    num_plots = len(history)
    n_rows = int(np.ceil(num_plots / 2))

    plt.figure(figsize=(20, 7 * n_rows))

    for i, (name, metric) in enumerate(history.items()):
        plt.subplot(n_rows, 2, i + 1)

        grid = np.arange(start, len(metric['train']))
        plt.plot(grid, metric['train'][start:], label='train')
        plt.plot(grid, metric['val'][start:], label='val')

        plt.title(name)
        plt.ylabel(name)
        plt.xlabel('Эпоха')
        plt.legend()

    if save_name is not None and plots_dir is not None:
        plt.savefig(plots_dir / save_name)
    plt.show()


class ProfitPercent:
    def __init__(self, fee=0.001, coef=100, buy_and_hold=None, threshold=None, short=True, long=True):
        self.fee = fee
        self.coef = coef

        self.threshold = threshold
        if threshold is None:
            self.threshold = 2 * self.fee
        self.profit = 0

        self.history_preds = []
        self.buy_and_hold = buy_and_hold

        self.current_idx = 0

        self.short = short
        self.long = long

    def update_state(self, y_true, y_pred):
        y_true = torch.squeeze(y_true) / self.coef
        y_pred = torch.squeeze(y_pred) / self.coef
        y_pred_copy = y_pred.clone()

        idxs = ((y_pred > 2 * self.threshold) & self.long) | ((y_pred < -2 * self.threshold) & self.short)
        y_true = y_true[idxs]
        y_pred = y_pred[idxs]

        fee_sold = (1 + y_true) * (1 - self.fee) * self.fee
        profit = y_true * torch.sign(y_pred) * (1 - self.fee) - fee_sold

        y_pred_copy[~idxs] = 0
        y_pred_copy[idxs] = profit
        self.history_preds.extend(y_pred_copy.detach().cpu().tolist())

        profit = profit.detach().cpu()
        self.profit += profit.numpy().sum()

    def result(self):
        return self.profit

    def plot(self, create_figure=False, plots_dir=None, save_name=None):
        profit = np.array(self.history_preds).cumsum()
        grid = np.arange(len(profit))

        if create_figure:
            plt.figure(figsize=(15, 8))

        plt.plot(grid, profit, label='Предсказания')
        if self.buy_and_hold is not None:
            plt.plot(grid, self.buy_and_hold, label='Купить и держать')

        plt.legend()
        plt.xlabel('Номер сделки')
        plt.ylabel('Кумулятивная доходность')
        add = ''
        if not self.short and self.long:
            add = ' only long'
        if not self.long and self.short:
            add = ' only short'

        plt.title(f'Комиссия {self.fee * 100}%, Граница {self.threshold * 100}%{add}')

        if save_name is not None and plots_dir is not None:
            plt.savefig(plots_dir / save_name)


def batch_step(model, batch, criterion, optimizer=None, prefix='val_', profit_percents=None):
    features_batch, target_batch, news, len_news, titles, len_titles = batch

    features_batch = features_batch.to(device)
    target_batch = target_batch.to(device)

    pred_ret = model(features_batch, news, len_news, titles, len_titles)

    loss = criterion(pred_ret, target_batch)

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    train_loss = loss.detach().cpu().numpy()
    train_mae = mean_absolute_error(target_batch.detach().cpu().numpy(), pred_ret.detach().cpu().numpy())
    wandb.log({f"{prefix}batch/loss": train_loss, f"{prefix}batch/mae": train_mae})

    if profit_percents is not None:
        for k, profit in profit_percents.items():
            profit.update_state(target_batch, pred_ret)
            wandb.log({f'{prefix}batch/profit_{k}': profit.result()})

    del loss, pred_ret
    del features_batch, target_batch, news, len_news, titles, len_titles

    return train_loss, train_mae


def train(
        model,
        criterion,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler=None,
        num_epochs=50,
        start_epoch=0,
        history=None,
        threshold=0.0005,
        fee=0.001,
        buy_and_hold=None,
        model_dir=None,
        plots_dir=None,
        save_name=None,
        step_plotting=1,
        model_params=None,
):
    best_profits = defaultdict(lambda: -float('inf'))

    if history is None:
        history = defaultdict(lambda: defaultdict(list))
        best_loss = float('inf')
        best_mae = float('inf')
    else:
        best_loss = min(history['loss']['val'])
        best_mae = min(history['mae']['val'])
        for name, value in history.items():
            if name.startswith('profit_'):
                best_profits[name.strip('profit_')] = value['val']

    draw_start = 0

    if plots_dir is not None and save_name is not None:
        plots_dir = plots_dir / save_name
        os.makedirs(plots_dir, exist_ok=True)

    wandb.init(
        project='Indicators',
        name=save_name,
        entity="investment-portfolio-optimization",
        config={
            'num_epochs': num_epochs,
            'loss': type(criterion).__name__,
            'optimizer': type(optimizer).__name__,
            'threshold': threshold,
            'batch_size': train_dataloader.batch_size,
            # 'shifts': shifts,
            'model_params': model_params,
        }
    )
    wandb.watch(model, criterion=criterion, log='all', log_graph=True, log_freq=512)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss, val_loss = 0.0, 0.0
        train_mae, val_mae = 0.0, 0.0

        profits = {
            'train': {
                'simple': ProfitPercent(fee=0.0),
                'ths': ProfitPercent(fee=0.0, threshold=threshold),
                'fee': ProfitPercent(fee=fee),
                'fee_ths': ProfitPercent(fee=fee, threshold=2 * fee + threshold),
            },
            'val': {
                'simple': ProfitPercent(fee=0.0, buy_and_hold=buy_and_hold),
                'ths': ProfitPercent(fee=0.0, buy_and_hold=buy_and_hold, threshold=threshold),
                'fee': ProfitPercent(fee=fee, buy_and_hold=buy_and_hold),
                'fee_ths': ProfitPercent(fee=fee, buy_and_hold=buy_and_hold, threshold=2 * fee + threshold),
            }
        }

        # Тренируем:
        start_time = time.time()
        model.train()

        for batch in tqdm(train_dataloader):
            loss, mae = batch_step(model, batch, criterion, optimizer,
                                   prefix='train_', profit_percents=profits['train'])
            train_loss += loss
            train_mae += mae

        if scheduler is not None:
            scheduler.step()

        train_loss /= len(train_dataloader)
        train_mae /= len(train_dataloader)

        history['loss']['train'].append(train_loss)
        history['mae']['train'].append(train_mae)
        for key, profit in profits['train'].items():
            history[f'profit_{key}']['train'].append(profit.result())

        wandb.log({"train/loss": train_loss, "train/mae": train_mae})
        wandb.log({f'train/profit_{k}': p.result() for k, p in profits['train'].items()})

        # Валидируем:
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                loss, mae = batch_step(model, batch, criterion, prefix='val_', profit_percents=profits['val'])
                val_loss += loss
                val_mae += mae

        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)

        history['loss']['val'].append(val_loss)
        history['mae']['val'].append(val_mae)
        for key, profit in profits['val'].items():
            history[f'profit_{key}']['val'].append(profit.result())

        wandb.log({"val/loss": val_loss, "val/mae": val_mae})
        wandb.log({f'val/profit_{k}': p.result() for k, p in profits['val'].items()})

        if save_name is not None and model_dir is not None:
            save_state(model, optimizer, epoch, history, model_dir, save_name + '_last', log_wandb=False)

            if best_loss > val_loss:
                best_loss = val_loss
                save_state(model, optimizer, epoch, history, model_dir, save_name + '_loss', log_wandb=True)

            if best_mae > val_mae:
                best_mae = val_mae
                save_state(model, optimizer, epoch, history, model_dir, save_name + '_mae', log_wandb=True)

            for name, profit in profits['val'].items():
                if best_profits[name] < profit.result():
                    best_profits[name] = profit.result()
                    save_state(model, optimizer, epoch, history, model_dir,
                               save_name + f'_profit_{name}', log_wandb=True)

        # Выводим метрики и строим графики
        if epoch % step_plotting == step_plotting - 1 or epoch == num_epochs - 1:
            display.clear_output()

            print(f'Epoch {epoch + 1} of {start_epoch + num_epochs} took {time.time() - start_time:.3f}s')

            print(f'  training loss: \t{train_loss:.6f}')
            print(f'  training mae: \t{train_mae:.6f}')
            for name, profit in profits['train'].items():
                print(f'  training profit {name}: \t{profit.result() * 100:.3f} %')

            print(f'  val loss: \t{val_loss:.6f} \t({best_loss:.6f})')
            print(f'  val mae: \t{val_mae:.6f} \t({best_mae:.6f})')
            for name, profit in profits['val'].items():
                print(f'  val profit {name}: \t{profit.result() * 100:.3f} % \t({best_profits[name] * 100:.3f} %)')

            # if len(history['loss']['train']) > 10:
            #     delta_begin = history['loss']['train'][draw_start] - history['loss']['train'][draw_start + 5]
            #     delta_end = history['loss']['train'][-6] - history['loss']['train'][-1]
            #     if delta_begin / delta_end > 10:
            #         draw_start = min(draw_start + 5, len(history['loss']['train']) - 2)

            plot_history(history, start=draw_start, plots_dir=plots_dir, save_name=f'metrics_{epoch + 1}epoch.png')

            for i, (name, profit) in enumerate(profits['val'].items()):
                fig = plt.figure(figsize=(10, 6))
                profit.plot()
                wandb.log({f'profit_plots/{name}': fig})

            plt.figure(figsize=(20, 14))
            for i, (name, profit) in enumerate(profits['val'].items()):
                plt.subplot(2, 2, i + 1)
                profit.plot()

            if plots_dir is not None:
                plt.savefig(plots_dir / f'profits_{epoch + 1}epoch.png')
            plt.show()

    wandb.finish()
    return model, history


def predict(
        model,
        criterion,
        val_dataloader,
        threshold=0.0005,
        fee=0.001,
        buy_and_hold=None,
        short=True,
        long=True,
        plots_dir=None,
        save_name=None,
):
    val_loss = 0.0
    val_mae = 0.0

    val_profit = ProfitPercent(fee=0.0, buy_and_hold=buy_and_hold, short=short, long=long)
    val_profit_threshold = ProfitPercent(fee=0.0, buy_and_hold=buy_and_hold, threshold=threshold, short=short,
                                         long=long)

    val_profit_fee = ProfitPercent(fee=fee, buy_and_hold=buy_and_hold, short=short, long=long)
    val_profit_fee_threshold = ProfitPercent(fee=fee, buy_and_hold=buy_and_hold, threshold=2 * fee + threshold,
                                             short=short, long=long)

    if plots_dir is not None and save_name is not None:
        plots_dir = plots_dir / save_name
        os.makedirs(plots_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            loss, mae = batch_step(model, batch, criterion,
                                   profit_percents=[val_profit, val_profit_threshold,
                                                    val_profit_fee, val_profit_fee_threshold])

            val_loss += loss
            val_mae += mae

    val_loss /= len(val_dataloader)
    val_mae /= len(val_dataloader)

    # Выводим метрики и строим графики
    display.clear_output()

    print(f'  val loss: \t{val_loss:.6f}')
    print(f'  val mae: \t{val_mae:.6f}')
    print(f'  val profit: \t{val_profit.result() * 100:.3f} %')
    print(f'  val profit ths: \t{val_profit_threshold.result() * 100:.3f} %')
    print(f'  val profit fee: \t{val_profit_fee.result() * 100:.3f} %')
    print(f'  val profit fee ths: \t{val_profit_fee_threshold.result() * 100:.3f} %')

    plt.figure(figsize=(20, 14))

    plt.subplot(2, 2, 1)
    val_profit.plot()
    plt.subplot(2, 2, 2)
    val_profit_fee.plot()

    plt.subplot(2, 2, 3)
    val_profit_threshold.plot()
    plt.subplot(2, 2, 4)
    val_profit_fee_threshold.plot()

    if plots_dir is not None:
        plt.savefig(plots_dir / f'profits_final.png')
    plt.show()

    profit = {'profit': val_profit.result(), 'profit_ths': val_profit_threshold.result(),
              'profit_fee': val_profit_fee.result(), 'profit_fee_ths': val_profit_fee_threshold.result()}

    return val_loss, val_mae, profit
