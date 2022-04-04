import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)


class StockDataset(Dataset):
    def __init__(self, dataset: torch.tensor, targets: torch.tensor):
        assert dataset.shape[0] == targets.shape[0]
        
        self.dataset = dataset.float()
        self.targets = targets.long()
        
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, idx):
        return self.dataset[idx], self.targets[idx]


class StockSequenceDataset(Dataset):
    def __init__(self, dataset: torch.tensor, targets: torch.tensor, sequence_length: int):
        assert dataset.shape[0] == targets.shape[0]

        self.dataset = dataset.float()
        self.targets = targets.long()
        self.sequence_length = sequence_length

    def __len__(self):
        return self.dataset.shape[0] // self.sequence_length

    def __getitem__(self, idx):
        from_idx = self.sequence_length * idx
        to_idx = self.sequence_length * (idx + 1)
        return self.dataset[from_idx:to_idx], self.targets[from_idx:to_idx]


class StockConvolutionalModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(StockConvolutionalModel, self).__init__()
        DROPOUT_PROBA = 0.3
        KERNEL_SIZE = 3
        
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=1),
            nn.Dropout(p=DROPOUT_PROBA),
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2),
            nn.AvgPool1d(kernel_size=KERNEL_SIZE),
            nn.BatchNorm1d(num_features=8),
            nn.Dropout(p=DROPOUT_PROBA),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2),
            nn.AvgPool1d(kernel_size=KERNEL_SIZE),
            nn.BatchNorm1d(num_features=16),
            nn.Dropout(p=DROPOUT_PROBA),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2),
            nn.AvgPool1d(kernel_size=KERNEL_SIZE),
            nn.BatchNorm1d(num_features=32),
            nn.Dropout(p=DROPOUT_PROBA)
        )

        self.linear = nn.Linear(in_features=32, out_features=num_classes)
        
    def forward(self, X):
        conv_out = self.conv(X)
        return self.linear(conv_out.squeeze().mean(axis=2))


class StockDenseModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(StockDenseModel, self).__init__()
        DROPOUT_PROBA = 0.3
        
        self.sequential = nn.Sequential(
            # nn.BatchNorm1d(num_features=input_size),
            # nn.Dropout(p=DROPOUT_PROBA),
            nn.Linear(in_features=input_size, out_features=num_classes)
        )
        
    def forward(self, X):
        return X


class StockLSTMModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int, sequence_length: int,
                 hidden_size: int = 128, num_layers: int = 3):
        super(StockLSTMModel, self).__init__()

        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)
        
    def forward(self, X):
        lstm_output, _ = self.lstm(X)
        return self.linear(lstm_output)

    def get_features(self, X):
        output, _ = self.lstm(X)
        return output[self.num_layers]


def train_batch(model, criterion, optimizer, X_batch, y_batch):
    optimizer.zero_grad()
    logits = model(X_batch.to(device)).cpu()
    loss = criterion(logits.view(-1, logits.shape[-1]), y_batch.flatten())
    loss.backward()
    optimizer.step()
    
    return loss.item()


def validate_batch(model, criterion, X_batch, y_batch):
    logits = model(X_batch.to(device)).cpu()
    loss = criterion(logits.view(-1, logits.shape[-1]), y_batch.flatten())
    
    return loss.item()


def print_losses(train_loss, validation_loss, epoch):
    print("////////////////////////////////////////")
    print("// Epoch:", epoch)
    print("// Train loss:", np.mean(train_loss))
    print("// Validation loss:", np.mean(validation_loss))
    print("////////////////////////////////////////")


def train_model(model, criterion, optimizer, train_loader, validation_loader, num_epoch):
    PRINT_PERIOD = 1
    model = model.to(device)
    
    for epoch in range(1, num_epoch + 1):
        model.train()
        train_loss = []
        for X_batch, y_batch in train_loader:
            train_loss.append(train_batch(model, criterion, optimizer, X_batch, y_batch))
        
        model.eval()
        validation_loss = []
        for X_batch, y_batch in validation_loader:
            validation_loss.append(validate_batch(model, criterion, X_batch, y_batch))
            
        if epoch % PRINT_PERIOD == 0:
            print_losses(train_loss, validation_loss, epoch)


def get_test_logits(model, test_loader):
    model = model.to(device)
    model.eval()

    all_logits = []
    all_targets = []
    for X_batch, y_batch in test_loader:
        logits = model(X_batch.to(device)).cpu()
        all_logits.append(logits)
        all_targets.append(y_batch)
    return torch.concat(all_logits), torch.concat(all_targets)
