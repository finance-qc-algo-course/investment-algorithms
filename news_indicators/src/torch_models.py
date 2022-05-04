import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
import plotly

import string
import datetime
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler
from stockstats import StockDataFrame
import warnings

import torch
from torch import nn
import torchtext
from IPython import display
from torchtext.data import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
import torch.nn.functional as F
import nltk
import gensim
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
import time
import os


class ModelSimple(nn.Module):
    def __init__(self, input_size, hidden_size=128, fc_size=32, num_layers=2, dropout=0.5):
        super(ModelSimple, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.norm = nn.BatchNorm1d(fc_size)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, features, news=None, len_news=None, titles=None, len_titles=None):
        # features.shape = (BATCH_SIZE, WINDOW_SIZE, N_FEATURES)

        out, (hidden, _) = self.lstm(features)
        # out.shape = (BATCH_SIZE, WINDOW_SIZE, HIDDEN_SIZE)
        # hidden.shape = (N_LAYERS, BATCH_SIZE, HIDDEN_SIZE)

        output = self.fc1(hidden[-1])
        #         output = self.norm(output)
        output = F.relu(self.dropout(output))
        output = self.fc2(output)

        return output


class ModelLSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.5):
        super(ModelLSTM_CNN, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(hidden_size + 128, 32)
        self.norm = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(32, 1)

    def forward(self, features, news=None, len_news=None, titles=None, len_titles=None):
        # features.shape = (BATCH_SIZE, WINDOW_SIZE, N_FEATURES)

        out, (hidden, _) = self.lstm(features)
        # out.shape = (BATCH_SIZE, WINDOW_SIZE, HIDDEN_SIZE)
        # hidden.shape = (N_LAYERS, BATCH_SIZE, HIDDEN_SIZE)

        features = features.transpose(1, 2)
        # features.shape = (BATCH_SIZE, N_FEATURES, WINDOW_SIZE)
        conv_output = self.conv(features)
        # conv_output.shape = (BATCH_SIZE, 128, ...)

        conv_output = self.dropout(F.adaptive_max_pool1d(conv_output, 1))
        # conv_output.shape = (BATCH_SIZE, 128)

        output = torch.cat((hidden[-1], torch.flatten(conv_output, 1)), dim=1)

        output = self.fc1(output)
        #         output = self.norm(output)
        output = F.relu(self.dropout(output))
        output = self.fc2(output)

        return output
