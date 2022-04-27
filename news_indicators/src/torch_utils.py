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

sns.set(style='darkgrid', font_scale=1.3, palette='Set2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_state(model, optimizer, epoch, history, model_dir, name):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': dict(history),
    }, model_dir / f'{name}.data')


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


def batch_step(model, batch, criterion, optimizer=None, profit_percents=None):
    features_batch, target_batch, news, len_news, titles, len_titles = batch

    features_batch = features_batch.to(device)
    target_batch = target_batch.to(device)
    news = news.to(device)
    titles = titles.to(device)

    pred_ret = model(features_batch, news, len_news, titles, len_titles)

    loss = criterion(pred_ret, target_batch)

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    train_loss = loss.detach().cpu().numpy()
    train_mae = mean_absolute_error(target_batch.detach().cpu().numpy(), pred_ret.detach().cpu().numpy())

    if profit_percents is not None:
        for profit in profit_percents:
            profit.update_state(target_batch, pred_ret)

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
):
    if history is None:
        history = defaultdict(lambda: defaultdict(list))
        best_loss = float('inf')
        best_mae = float('inf')
        best_profit = -float('inf')
        best_profit_threshold = -float('inf')
        best_profit_fee = -float('inf')
        best_profit_fee_threshold = -float('inf')
    else:
        best_loss = min(history['loss']['val'])
        best_mae = min(history['mae']['val'])
        best_profit = max(history['profit']['val'])
        best_profit_threshold = max(history['profit_ths']['val'])
        best_profit_fee = max(history['profit_fee']['val'])
        best_profit_fee_threshold = max(history['profit_fee_ths']['val'])

    draw_start = 0

    if plots_dir is not None and save_name is not None:
        plots_dir = plots_dir / save_name
        os.makedirs(plots_dir, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss, val_loss = 0.0, 0.0
        train_mae, val_mae = 0.0, 0.0

        train_profit = ProfitPercent(fee=0.0)
        train_profit_threshold = ProfitPercent(fee=0.0, threshold=threshold)
        val_profit = ProfitPercent(fee=0.0, buy_and_hold=buy_and_hold)
        val_profit_threshold = ProfitPercent(fee=0.0, buy_and_hold=buy_and_hold, threshold=threshold)

        train_profit_fee = ProfitPercent(fee=fee)
        train_profit_fee_threshold = ProfitPercent(fee=fee, threshold=threshold)
        val_profit_fee = ProfitPercent(fee=fee, buy_and_hold=buy_and_hold)
        val_profit_fee_threshold = ProfitPercent(fee=fee, buy_and_hold=buy_and_hold, threshold=2 * fee + threshold)

        # Тренируем:
        start_time = time.time()
        model.train()

        for batch in tqdm(train_dataloader):
            loss, mae = batch_step(
                model, batch, criterion, optimizer,
                profit_percents=[train_profit, train_profit_threshold, train_profit_fee, train_profit_fee_threshold]
            )
            train_loss += loss
            train_mae += mae

        if scheduler is not None:
            scheduler.step()

        train_loss /= len(train_dataloader)
        train_mae /= len(train_dataloader)

        history['loss']['train'].append(train_loss)
        history['mae']['train'].append(train_mae)
        history['profit']['train'].append(train_profit.result())
        history['profit_ths']['train'].append(train_profit_threshold.result())
        history['profit_fee']['train'].append(train_profit_fee.result())
        history['profit_fee_ths']['train'].append(train_profit_fee_threshold.result())

        # Валидируем:
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                loss, mae = batch_step(
                    model, batch, criterion,
                    profit_percents=[val_profit, val_profit_threshold, val_profit_fee, val_profit_fee_threshold]
                )
                val_loss += loss
                val_mae += mae

        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)

        history['loss']['val'].append(val_loss)
        history['mae']['val'].append(val_mae)
        history['profit']['val'].append(val_profit.result())
        history['profit_ths']['val'].append(val_profit_threshold.result())
        history['profit_fee']['val'].append(val_profit_fee.result())
        history['profit_fee_ths']['val'].append(val_profit_fee_threshold.result())

        if save_name is not None and model_dir is not None:
            save_state(model, optimizer, epoch, history, model_dir, save_name + '_last')

            if best_loss > val_loss:
                best_loss = val_loss
                save_state(model, optimizer, epoch, history, model_dir, save_name + '_loss')

            if best_mae > val_mae:
                best_mae = val_mae
                save_state(model, optimizer, epoch, history, model_dir, save_name + '_mae')

            if best_profit < val_profit.result():
                best_profit = val_profit.result()
                save_state(model, optimizer, epoch, history, model_dir, save_name + '_profit')

            if best_profit_threshold < val_profit_threshold.result():
                best_profit_threshold = val_profit_threshold.result()
                save_state(model, optimizer, epoch, history, model_dir, save_name + '_profit_ths')

            if best_profit_fee < val_profit_fee.result():
                best_profit_fee = val_profit_fee.result()
                save_state(model, optimizer, epoch, history, model_dir, save_name + '_profit_fee')

            if best_profit_fee_threshold < val_profit_fee_threshold.result():
                best_profit_fee_threshold = val_profit_fee_threshold.result()
                save_state(model, optimizer, epoch, history, model_dir, save_name + '_profit_fee_ths')

        # Выводим метрики и строим графики
        if epoch % step_plotting == step_plotting - 1 or epoch == num_epochs - 1:
            display.clear_output()

            print(f'Epoch {epoch + 1} of {start_epoch + num_epochs} took {time.time() - start_time:.3f}s')

            print(f'  training loss: \t{train_loss:.6f}')
            print(f'  training mae: \t{train_mae:.6f}')
            print(f'  training profit: \t{train_profit.result() * 100:.3f} %')
            print(f'  training profit ths: \t{train_profit_threshold.result() * 100:.3f} %')
            print(f'  training profit fee: \t{train_profit_fee.result() * 100:.3f} %')
            print(f'  training profit fee ths: \t{train_profit_fee_threshold.result() * 100:.3f} %')

            print(f'  val loss: \t{val_loss:.6f} \t({best_loss:.6f})')
            print(f'  val mae: \t{val_mae:.6f} \t({best_mae:.6f})')
            print(f'  val profit: \t{val_profit.result() * 100:.3f} % \t({best_profit * 100:.3f} %)')
            print(f'  val profit ths: \t{val_profit_threshold.result() * 100:.3f} % '
                  f'\t({best_profit_threshold * 100:.3f} %)')
            print(f'  val profit fee: \t{val_profit_fee.result() * 100:.3f} % \t({best_profit_fee * 100:.3f} %)')
            print(f'  val profit fee ths: \t{val_profit_fee_threshold.result() * 100:.3f} % '
                  f'\t({best_profit_fee_threshold * 100:.3f} %)')

            # if len(history['loss']['train']) > 10:
            #     delta_begin = history['loss']['train'][draw_start] - history['loss']['train'][draw_start + 5]
            #     delta_end = history['loss']['train'][-6] - history['loss']['train'][-1]
            #     if delta_begin / delta_end > 10:
            #         draw_start = min(draw_start + 5, len(history['loss']['train']) - 2)

            plot_history(history, start=draw_start, plots_dir=plots_dir, save_name=f'metrics_{epoch + 1}epoch.png')

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
                plt.savefig(plots_dir / f'profits_{epoch + 1}epoch.png')
            plt.show()

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
