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

from .prepare_data import MAX_LEN, MAX_TITLE_LEN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelSimple(nn.Module):
    def __init__(self, input_size, hidden_size=128, fc_size=32, num_layers=2, dropout=0.5, **kwargs):
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
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.5, **kwargs):
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


class TextModule(nn.Module):
    def __init__(self, embedding, embedding_size, hidden_size, dropout, max_len, **kwargs):
        super(TextModule, self).__init__()

        self.max_len = max_len

        self.embedding = embedding
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

    def forward(self, text, text_len):
        # text.shape = (BS, MAX_LEN)

        embed = self.embedding(text)
        # embed = (BS, MAX_LEN, EMBED_SIZE)

        packed_in = nn.utils.rnn.pack_padded_sequence(
            embed, text_len, batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden_state, _) = self.lstm(packed_in)
        out, lengths = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=self.max_len
        )
        # out.shape = (BS, MAX_LEN, 2 * HIDDEN_SIZE)
        # hidden_state.shape = (2, BS, HIDDEN_SIZE)

        avg_pool = torch.sum(out, dim=1) / lengths.to(device).view(-1, 1)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)

        max_pool = torch.cat([torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out, lengths)], dim=0)
        # max_pool.shape = (BS, 2 * HIDDEN_SIZE)

        output = torch.cat([hidden_state[-1], hidden_state[-2], avg_pool, max_pool], dim=1)
        # output.shape = (BS, 6 * HIDDEN_SIZE)
        return output


class ModelSimpleIndNews(nn.Module):
    def __init__(self, input_size, vocab, hidden_size=196, fc_size=64, num_layers=1,
                 dropout=0.5, embedding_size=80, hidden_news_size=32,
                 hidden_title_size=24, max_news_len=MAX_LEN, max_title_len=MAX_TITLE_LEN, **kwargs):
        super(ModelSimpleIndNews, self).__init__()

        self.max_news_len = max_news_len
        self.max_title_len = max_title_len
        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=vocab['<pad>'])

        # INDICATORS
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ind = nn.BatchNorm1d(hidden_size)

        # NEWS
        self.news_module = TextModule(self.embedding, embedding_size,
                                      hidden_news_size, dropout, max_len=max_news_len)
        self.norm_news = nn.BatchNorm1d(6 * hidden_news_size)

        # TITLES
        self.title_module = TextModule(self.embedding, embedding_size,
                                       hidden_title_size, dropout, max_len=max_title_len)
        self.norm_title = nn.BatchNorm1d(6 * hidden_title_size)

        # UNION
        self.fc1 = nn.Linear(hidden_size + 6 * hidden_news_size + 6 * hidden_title_size, fc_size)
        self.norm = nn.BatchNorm1d(fc_size)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, X, news, len_news, titles, len_titles):
        # X.shape = (BATCH_SIZE, WINDOW_SIZE, N_FEATURES)
        # news = (BS, [MAX_LEN, ...])
        last_news = torch.tensor(np.array(
            [n[-1] if len(n) > 0 else [0] * self.max_news_len for n in news]
        )).to(device)
        last_len_news = [l[-1] if len(l) > 0 else 0 for l in len_news]

        last_titles = torch.tensor(np.array(
            [t[-1] if len(t) > 0 else [0] * self.max_title_len for t in titles]
        )).to(device)
        last_len_titles = [l[-1] if len(l) > 0 else 0 for l in len_titles]

        # INDICATORS
        _, (hidden_ind, _) = self.lstm(X)
        #   out.shape = (BATCH_SIZE, WINDOW_SIZE, HIDDEN_SIZE)
        #   hidden_ind.shape = (N_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
        hidden_ind = self.norm_ind(hidden_ind[-1])
        #   hidden_ind.shape = (BATCH_SIZE, HIDDEN_SIZE)

        # NEWS
        news_out = self.news_module(last_news, last_len_news)
        news_out = self.norm_news(news_out)
        #   news_out.shape = (BATCH_SIZE, 6 * HIDDEN_NEWS_SIZE)

        # TITLES
        title_out = self.title_module(last_titles, last_len_titles)
        title_out = self.norm_title(title_out)
        #   title_out.shape = (BATCH_SIZE, 6 * HIDDEN_TITLE_SIZE)

        # UNION
        output = torch.cat([hidden_ind, news_out, title_out], dim=1)
        #   output.shape = (BATCH_SIZE, HIDDEN_SIZE + 6 * HIDDEN_NEWS_SIZE + 6 * HIDDEN_TITLE_SIZE)
        output = self.fc1(output)
        # output = self.norm(output)
        output = F.relu(self.dropout(output))
        output = self.fc2(output)

        return output
