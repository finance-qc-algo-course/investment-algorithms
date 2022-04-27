import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model1(nn.Module):
    def __init__(self, vocab, embed_dim=200):
        super(Model1, self).__init__()

        self.embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True, dropout=0.3)

        self.fc1 = nn.Linear(256, 64)
        self.norm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, text, text_len, features=None):
        embedded = self.embedding(text)

        embedded = embedded.permute(1, 0, 2)
        packed_in = nn.utils.rnn.pack_padded_sequence(embedded, text_len,
                                                      batch_first=True,
                                                      enforce_sorted=False)
        out, (hidden, cell) = self.lstm(packed_in)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc1(hidden)
        output = self.norm(output)
        output = F.relu(self.dropout(output))
        output = self.fc2(output)

        return output


class Model2(nn.Module):
    def __init__(self, vocab):
        super(Model2, self).__init__()

        self.embedding = nn.Embedding(len(vocab), 300, padding_idx=0)
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=128,
                            num_layers=3,
                            # batch_first=True,
                            bidirectional=True,
                            dropout=0.4)

        self.dropout0 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256, 128)
        # self.norm = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(32, 1)

    def forward(self, text, text_len, features=None):
        embedded = self.embedding(text)

        embedded = embedded.permute(1, 0, 2)
        packed_in = nn.utils.rnn.pack_padded_sequence(embedded, text_len,
                                                      #  batch_first=True,
                                                      enforce_sorted=False)
        out, (hidden, cell) = self.lstm(packed_in)
        # out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=MAX_LEN)

        hidden = self.dropout0(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        output = self.fc1(hidden)
        # output = self.norm(output)
        output = F.relu(self.dropout1(output))

        output = self.fc2(output)
        output = F.relu(self.dropout2(output))

        output = self.fc3(output)
        return output


class Model3(nn.Module):
    def __init__(self, vocab):
        super(Model3, self).__init__()

        self.embedding = nn.Embedding(len(vocab), 150, padding_idx=0)
        self.lstm = nn.LSTM(input_size=150,
                            hidden_size=50,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.3)

        self.dropout0 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(100, 64)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, text, text_len, features=None):
        embedded = self.embedding(text)

        embedded = embedded.permute(1, 0, 2)
        packed_in = nn.utils.rnn.pack_padded_sequence(embedded, text_len,
                                                      batch_first=True,
                                                      enforce_sorted=False)
        out, (hidden, cell) = self.lstm(packed_in)

        hidden = self.dropout0(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        output = self.fc1(hidden)
        output = F.leaky_relu(self.dropout1(output))

        output = self.fc2(output)
        return output


class Model4(nn.Module):
    def __init__(self, vocab):
        super(Model4, self).__init__()

        self.embedding = nn.Embedding(len(vocab), 100, padding_idx=0)
        self.lstm = nn.LSTM(input_size=100,
                            hidden_size=50,
                            num_layers=5,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5)

        self.dropout0 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(100, 32)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(32, 1)

    def forward(self, text, text_len, features=None):
        embedded = self.embedding(text)

        embedded = embedded.permute(1, 0, 2)
        packed_in = nn.utils.rnn.pack_padded_sequence(embedded, text_len,
                                                      batch_first=True,
                                                      enforce_sorted=False)
        out, (hidden, cell) = self.lstm(packed_in)

        hidden = self.dropout0(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        output = self.fc1(hidden)
        output = F.leaky_relu(self.dropout1(output))

        output = self.fc2(output)
        return output


class Model5(nn.Module):
    def __init__(self, vocab):
        super(Model5, self).__init__()

        self.embedding = nn.Embedding(len(vocab), 150, padding_idx=0)
        self.lstm = nn.LSTM(input_size=150,
                            hidden_size=50,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.3)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=100, out_channels=128,
                kernel_size=fs,
            ) for fs in [3, 4, 5]
        ])

        self.dropout0 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 3, 64)
        # self.norm = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, text, text_len, features=None):
        embedded = self.embedding(text)

        output, (_, _) = self.lstm(embedded)

        output = output.permute(0, 2, 1)
        conved = [F.relu(conv(output)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        pooled = self.dropout0(torch.cat(pooled, dim=1))

        output = self.fc1(pooled)
        # output = self.norm(output)
        output = F.relu(self.dropout1(output))

        output = self.fc2(output)
        return output


class Model6(nn.Module):
    def __init__(self, vocab):
        super(Model6, self).__init__()

        self.embedding = nn.Embedding(len(vocab), 150, padding_idx=0)
        self.lstm = nn.LSTM(input_size=150,
                            hidden_size=64,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.3)
        self.dropout0 = nn.Dropout(0.3)

        # -------

        self.fc1_X = nn.Linear(147, 1024)
        self.dropout1_X = nn.Dropout(0.3)
        self.fc2_X = nn.Linear(1024, 128)
        self.dropout2_X = nn.Dropout(0.3)
        self.norm = nn.BatchNorm1d(128)

        # -------

        self.fc1 = nn.Linear(64 * 2 + 128, 64)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, text, text_len, features):
        embedded = self.embedding(text)

        embedded = embedded.permute(1, 0, 2)
        packed_in = nn.utils.rnn.pack_padded_sequence(embedded, text_len,
                                                      batch_first=True,
                                                      enforce_sorted=False)
        out, (hidden, cell) = self.lstm(packed_in)
        # out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=MAX_LEN)

        hidden = self.dropout0(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # -------

        features_out = F.relu(self.dropout1_X(self.fc1_X(features)))
        features_out = self.norm(F.relu(self.dropout2_X(self.fc2_X(features_out))))

        # -------

        output = self.fc1(torch.cat([hidden, features_out], dim=1))
        # output = self.norm(output)
        output = F.leaky_relu(self.dropout1(output))

        output = self.fc2(output)
        return output


class ModelGRU1(nn.Module):
    def __init__(self, vocab, max_len, embedding_size=100, hidden_size=40):
        super(ModelGRU1, self).__init__()
        self.max_len = max_len

        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.dropout0 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8 * hidden_size, 128)
        # self.norm = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 1)

    def forward(self, text, text_len, features=None):
        # text.shape = (BS, MAX_LEN)
        # print("text.shape:", text.shape)

        embedded = self.embedding(text)
        # embedded.shape = (BS, MAX_LEN, EMBED_SIZE)
        # print("embedded.shape:", embedded.shape)

        packed_in = nn.utils.rnn.pack_padded_sequence(
            embedded,
            text_len,
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, hidden_state = self.gru(packed_in)
        out, lengths = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=self.max_len)
        # out.shape = (BS, MAX_LEN, 2 * HIDDEN_SIZE), hidden_state.shape = ()
        # print("out.shape:", out.shape)
        # print("hidden_state.shape:", hidden_state.shape)

        avg_pool = torch.sum(out, dim=1) / lengths.to(device).view(-1, 1)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('avg_pool.shape:', avg_pool.shape)

        max_pool = torch.cat([torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out, lengths)], dim=0)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('max_pool.shape:', max_pool.shape)

        # hidden_state = self.dropout0(torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1))

        pool_cat = torch.cat([hidden_state[-1], hidden_state[-2],
                              hidden_state[-3], hidden_state[-4],
                              avg_pool, max_pool], dim=1)
        # pool_cat.shape = (BS, 8 * HIDDEN_SIZE)
        # print('pool_cat.shape:', pool_cat.shape)

        output = self.fc1(self.dropout0(pool_cat))
        # output = self.norm(output)
        output = F.leaky_relu(self.dropout1(output))

        output = self.fc2(output)
        return output


class ModelGRU_IND1(nn.Module):
    def __init__(self, vocab, max_len, max_title_len, feature_input_size, embedding_size=100,
                 hidden_news_size=40, hidden_title_size=20):
        super(ModelGRU_IND1, self).__init__()
        self.max_len = max_len
        self.max_title_len = max_title_len

        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=0)
        self.gru_news = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_news_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.dropout_news0 = nn.Dropout(0.5)
        self.fc_news1 = nn.Linear(8 * hidden_news_size, 64)
        self.norm_news1 = nn.BatchNorm1d(64)
        self.dropout_news1 = nn.Dropout(0.3)

        # -------

        self.gru_titles = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_title_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )

        self.dropout_title0 = nn.Dropout(0.5)
        self.fc_title1 = nn.Linear(8 * hidden_title_size, 64)
        self.norm_title1 = nn.BatchNorm1d(64)
        self.dropout_title1 = nn.Dropout(0.3)

        # -------

        self.fc1_X = nn.Linear(feature_input_size, 256)
        self.dropout1_X = nn.Dropout(0.3)
        self.fc2_X = nn.Linear(256, 64)
        self.norm2_X = nn.BatchNorm1d(64)
        self.dropout2_X = nn.Dropout(0.3)

        # -------

        self.fc2 = nn.Linear(64 + 64 + 64, 32)
        self.norm2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, 1)

    def forward(self, text, text_len, title, title_len, features):
        # text.shape = (BS, MAX_LEN)
        # print("text.shape:", text.shape)

        embedded_news = self.embedding(text)
        embedded_titles = self.embedding(title)
        # embedded.shape = (BS, MAX_LEN, EMBED_SIZE)
        # print("embedded.shape:", embedded.shape)

        packed_news_in = nn.utils.rnn.pack_padded_sequence(
            embedded_news, text_len,
            batch_first=True, enforce_sorted=False
        )
        packed_titles_in = nn.utils.rnn.pack_padded_sequence(
            embedded_titles, title_len,
            batch_first=True, enforce_sorted=False
        )
        packed_news_out, hidden_news_state = self.gru_news(packed_news_in)
        packed_titles_out, hidden_titles_state = self.gru_titles(packed_titles_in)

        out_news, lengths_news = nn.utils.rnn.pad_packed_sequence(packed_news_out,
                                                                  batch_first=True,
                                                                  total_length=self.max_len)
        out_titles, lengths_titles = nn.utils.rnn.pad_packed_sequence(packed_titles_out,
                                                                      batch_first=True,
                                                                      total_length=self.max_title_len)
        # out.shape = (BS, MAX_LEN, 2 * HIDDEN_SIZE), hidden_state.shape = ()
        # print("out.shape:", out.shape)
        # print("hidden_state.shape:", hidden_state.shape)

        avg_pool_news = torch.sum(out_news, dim=1) / lengths_news.to(device).view(-1, 1)
        avg_pool_titles = torch.sum(out_titles, dim=1) / lengths_titles.to(device).view(-1, 1)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('avg_pool.shape:', avg_pool.shape)

        max_pool_news = torch.cat([torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out_news, lengths_news)],
                                  dim=0)
        max_pool_titles = torch.cat(
            [torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out_titles, lengths_titles)], dim=0)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('max_pool.shape:', max_pool.shape)

        # hidden_state = self.dropout0(torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1))

        pool_cat_news = torch.cat([hidden_news_state[-1], hidden_news_state[-2],
                                   hidden_news_state[-3], hidden_news_state[-4],
                                   avg_pool_news, max_pool_news], dim=1)
        pool_cat_titles = torch.cat([hidden_titles_state[-1], hidden_titles_state[-2],
                                     hidden_titles_state[-3], hidden_titles_state[-4],
                                     avg_pool_titles, max_pool_titles], dim=1)

        # pool_cat.shape = (BS, 8 * HIDDEN_SIZE)
        # print('pool_cat.shape:', pool_cat.shape)

        output_news = self.fc_news1(self.dropout_news0(pool_cat_news))
        output_titles = self.fc_title1(self.dropout_title0(pool_cat_titles))
        # output = self.norm(output)
        output_news = F.leaky_relu(self.dropout_news1(output_news))
        output_titles = F.leaky_relu(self.dropout_title1(output_titles))

        # -------

        features_out = F.relu(self.dropout1_X(self.fc1_X(features)))
        features_out = F.leaky_relu(self.dropout2_X(self.norm2_X(self.fc2_X(features_out))))

        # -------

        output = self.fc2(torch.cat([output_news, output_titles, features_out], dim=1))
        output = F.relu(self.dropout2(self.norm2(output)))
        output = self.fc3(output)
        return output


class ModelGRU_IND3(nn.Module):
    def __init__(self, vocab, max_len, max_title_len, feature_input_size, embedding_size=80,
                 hidden_news_size=30, hidden_title_size=30):
        super(ModelGRU_IND3, self).__init__()
        self.max_len = max_len
        self.max_title_len = max_title_len

        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=0)
        self.gru_news = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_news_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.dropout_news0 = nn.Dropout(0.5)
        self.fc_news1 = nn.Linear(8 * hidden_news_size, 48)
        self.norm_news1 = nn.BatchNorm1d(48)
        self.dropout_news1 = nn.Dropout(0.4)

        # -------

        self.gru_titles = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_title_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            # dropout=0.4
        )

        self.dropout_title0 = nn.Dropout(0.5)
        self.fc_title1 = nn.Linear(6 * hidden_title_size, 32)
        self.norm_title1 = nn.BatchNorm1d(32)
        self.dropout_title1 = nn.Dropout(0.3)

        # -------

        self.fc1_X = nn.Linear(feature_input_size, 128)
        self.dropout1_X = nn.Dropout(0.3)
        self.fc2_X = nn.Linear(128, 64)
        self.norm2_X = nn.BatchNorm1d(64)
        self.dropout2_X = nn.Dropout(0.4)

        # -------

        self.fc2 = nn.Linear(64 + 48 + 32, 32)
        self.norm2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, 1)

    def forward(self, text, text_len, title, title_len, features):
        # text.shape = (BS, MAX_LEN)
        # print("text.shape:", text.shape)

        embedded_news = self.embedding(text)
        embedded_titles = self.embedding(title)
        # embedded.shape = (BS, MAX_LEN, EMBED_SIZE)
        # print("embedded.shape:", embedded.shape)

        packed_news_in = nn.utils.rnn.pack_padded_sequence(
            embedded_news, text_len,
            batch_first=True, enforce_sorted=False
        )
        packed_titles_in = nn.utils.rnn.pack_padded_sequence(
            embedded_titles, title_len,
            batch_first=True, enforce_sorted=False
        )
        packed_news_out, hidden_news_state = self.gru_news(packed_news_in)
        packed_titles_out, hidden_titles_state = self.gru_titles(packed_titles_in)

        out_news, lengths_news = nn.utils.rnn.pad_packed_sequence(packed_news_out,
                                                                  batch_first=True,
                                                                  total_length=self.max_len)
        out_titles, lengths_titles = nn.utils.rnn.pad_packed_sequence(packed_titles_out,
                                                                      batch_first=True,
                                                                      total_length=self.max_title_len)
        # out.shape = (BS, MAX_LEN, 2 * HIDDEN_SIZE), hidden_state.shape = ()
        # print("out.shape:", out.shape)
        # print("hidden_state.shape:", hidden_state.shape)

        avg_pool_news = torch.sum(out_news, dim=1) / lengths_news.to(device).view(-1, 1)
        avg_pool_titles = torch.sum(out_titles, dim=1) / lengths_titles.to(device).view(-1, 1)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('avg_pool.shape:', avg_pool.shape)

        max_pool_news = torch.cat([torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out_news, lengths_news)],
                                  dim=0)
        max_pool_titles = torch.cat(
            [torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out_titles, lengths_titles)], dim=0)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('max_pool.shape:', max_pool.shape)

        # hidden_state = self.dropout0(torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1))

        pool_cat_news = torch.cat([hidden_news_state[-1], hidden_news_state[-2],
                                   hidden_news_state[-3], hidden_news_state[-4],
                                   avg_pool_news, max_pool_news], dim=1)
        pool_cat_titles = torch.cat([hidden_titles_state[-1], hidden_titles_state[-2],
                                     avg_pool_titles, max_pool_titles], dim=1)

        # pool_cat.shape = (BS, 8 * HIDDEN_SIZE)
        # print('pool_cat.shape:', pool_cat.shape)

        output_news = self.fc_news1(self.dropout_news0(pool_cat_news))
        output_titles = self.fc_title1(self.dropout_title0(pool_cat_titles))
        # output_news = self.norm_news1(output_news)
        # output_titles = self.norm_title1(output_titles)
        output_news = F.leaky_relu(self.dropout_news1(output_news))
        output_titles = F.leaky_relu(self.dropout_title1(output_titles))

        # -------

        features_out = F.relu(self.dropout1_X(self.fc1_X(features)))
        features_out = F.leaky_relu(self.dropout2_X(self.norm2_X(self.fc2_X(features_out))))

        # -------

        output = self.fc2(torch.cat([output_news, output_titles, features_out], dim=1))
        output = F.relu(self.dropout2(self.norm2(output)))
        output = self.fc3(output)
        return output


class ModelGRU_IND5(nn.Module):
    def __init__(self, vocab, max_len, max_title_len, feature_input_size, embedding_size=80,
                 hidden_news_size=40, hidden_title_size=40):
        super(ModelGRU_IND5, self).__init__()
        self.max_len = max_len
        self.max_title_len = max_title_len

        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=0)
        self.gru_news = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_news_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            # dropout=0.5
        )

        self.dropout_news0 = nn.Dropout(0.5)
        self.fc_news1 = nn.Linear(3 * hidden_news_size, 48)
        self.norm_news1 = nn.BatchNorm1d(48)
        self.dropout_news1 = nn.Dropout(0.4)

        # -------

        self.gru_titles = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_title_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            # dropout=0.4
        )

        self.dropout_title0 = nn.Dropout(0.5)
        self.fc_title1 = nn.Linear(3 * hidden_title_size, 32)
        self.norm_title1 = nn.BatchNorm1d(32)
        self.dropout_title1 = nn.Dropout(0.3)

        # -------

        self.fc1_X = nn.Linear(feature_input_size, 128)
        self.dropout1_X = nn.Dropout(0.3)
        self.fc2_X = nn.Linear(128, 64)
        self.norm2_X = nn.BatchNorm1d(64)
        self.dropout2_X = nn.Dropout(0.4)

        # -------

        self.fc2 = nn.Linear(64 + 48 + 32, 32)
        self.norm2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, 1)

    def forward(self, text, text_len, title, title_len, features):
        # text.shape = (BS, MAX_LEN)
        # print("text.shape:", text.shape)

        embedded_news = self.embedding(text)
        embedded_titles = self.embedding(title)
        # embedded.shape = (BS, MAX_LEN, EMBED_SIZE)
        # print("embedded.shape:", embedded.shape)

        packed_news_in = nn.utils.rnn.pack_padded_sequence(
            embedded_news, text_len,
            batch_first=True, enforce_sorted=False
        )
        packed_titles_in = nn.utils.rnn.pack_padded_sequence(
            embedded_titles, title_len,
            batch_first=True, enforce_sorted=False
        )
        packed_news_out, hidden_news_state = self.gru_news(packed_news_in)
        packed_titles_out, hidden_titles_state = self.gru_titles(packed_titles_in)

        out_news, lengths_news = nn.utils.rnn.pad_packed_sequence(packed_news_out,
                                                                  batch_first=True,
                                                                  total_length=self.max_len)
        out_titles, lengths_titles = nn.utils.rnn.pad_packed_sequence(packed_titles_out,
                                                                      batch_first=True,
                                                                      total_length=self.max_title_len)
        # out.shape = (BS, MAX_LEN, 2 * HIDDEN_SIZE), hidden_state.shape = ()
        # print("out.shape:", out.shape)
        # print("hidden_state.shape:", hidden_state.shape)

        avg_pool_news = torch.sum(out_news, dim=1) / lengths_news.to(device).view(-1, 1)
        avg_pool_titles = torch.sum(out_titles, dim=1) / lengths_titles.to(device).view(-1, 1)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('avg_pool.shape:', avg_pool.shape)

        max_pool_news = torch.cat([torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out_news, lengths_news)],
                                  dim=0)
        max_pool_titles = torch.cat(
            [torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out_titles, lengths_titles)], dim=0)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('max_pool.shape:', max_pool.shape)

        # hidden_state = self.dropout0(torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1))

        pool_cat_news = torch.cat([hidden_news_state[-1],
                                   avg_pool_news, max_pool_news], dim=1)
        pool_cat_titles = torch.cat([hidden_titles_state[-1],
                                     avg_pool_titles, max_pool_titles], dim=1)

        # pool_cat.shape = (BS, 8 * HIDDEN_SIZE)
        # print('pool_cat.shape:', pool_cat.shape)

        output_news = self.fc_news1(self.dropout_news0(pool_cat_news))
        output_titles = self.fc_title1(self.dropout_title0(pool_cat_titles))
        # output_news = self.norm_news1(output_news)
        # output_titles = self.norm_title1(output_titles)
        output_news = F.leaky_relu(self.dropout_news1(output_news))
        output_titles = F.leaky_relu(self.dropout_title1(output_titles))

        # -------

        features_out = F.relu(self.dropout1_X(self.fc1_X(features)))
        features_out = F.leaky_relu(self.dropout2_X(self.norm2_X(self.fc2_X(features_out))))

        # -------

        output = self.fc2(torch.cat([output_news, output_titles, features_out], dim=1))
        output = F.relu(self.dropout2(self.norm2(output)))
        output = self.fc3(output)
        return output


class ModelGRU_IND6(nn.Module):
    def __init__(self, vocab, max_len, max_title_len, feature_input_size, embedding_size=50,
                 hidden_news_size=20, hidden_title_size=20):
        super(ModelGRU_IND6, self).__init__()
        self.max_len = max_len
        self.max_title_len = max_title_len

        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=0)
        self.gru_news = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_news_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            # dropout=0.5
        )

        self.dropout_news0 = nn.Dropout(0.4)
        self.fc_news1 = nn.Linear(3 * hidden_news_size, 24)
        self.norm_news1 = nn.BatchNorm1d(24)
        self.dropout_news1 = nn.Dropout(0.3)

        # -------

        self.gru_titles = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_title_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            # dropout=0.4
        )

        self.dropout_title0 = nn.Dropout(0.5)
        self.fc_title1 = nn.Linear(3 * hidden_title_size, 16)
        self.norm_title1 = nn.BatchNorm1d(16)
        self.dropout_title1 = nn.Dropout(0.3)

        # -------

        self.fc1_X = nn.Linear(feature_input_size, 64)
        self.dropout1_X = nn.Dropout(0.3)
        self.fc2_X = nn.Linear(64, 24)
        self.norm2_X = nn.BatchNorm1d(24)
        self.dropout2_X = nn.Dropout(0.4)

        # -------

        self.fc2 = nn.Linear(24 + 24 + 16, 16)
        self.norm2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(16, 1)

    def forward(self, text, text_len, title, title_len, features):
        # text.shape = (BS, MAX_LEN)
        # print("text.shape:", text.shape)

        embedded_news = self.embedding(text)
        embedded_titles = self.embedding(title)
        # embedded.shape = (BS, MAX_LEN, EMBED_SIZE)
        # print("embedded.shape:", embedded.shape)

        packed_news_in = nn.utils.rnn.pack_padded_sequence(
            embedded_news, text_len,
            batch_first=True, enforce_sorted=False
        )
        packed_titles_in = nn.utils.rnn.pack_padded_sequence(
            embedded_titles, title_len,
            batch_first=True, enforce_sorted=False
        )
        packed_news_out, hidden_news_state = self.gru_news(packed_news_in)
        packed_titles_out, hidden_titles_state = self.gru_titles(packed_titles_in)

        out_news, lengths_news = nn.utils.rnn.pad_packed_sequence(packed_news_out,
                                                                  batch_first=True,
                                                                  total_length=self.max_len)
        out_titles, lengths_titles = nn.utils.rnn.pad_packed_sequence(packed_titles_out,
                                                                      batch_first=True,
                                                                      total_length=self.max_title_len)
        # out.shape = (BS, MAX_LEN, 2 * HIDDEN_SIZE), hidden_state.shape = ()
        # print("out.shape:", out.shape)
        # print("hidden_state.shape:", hidden_state.shape)

        avg_pool_news = torch.sum(out_news, dim=1) / lengths_news.to(device).view(-1, 1)
        avg_pool_titles = torch.sum(out_titles, dim=1) / lengths_titles.to(device).view(-1, 1)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('avg_pool.shape:', avg_pool.shape)

        max_pool_news = torch.cat([torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out_news, lengths_news)],
                                  dim=0)
        max_pool_titles = torch.cat(
            [torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out_titles, lengths_titles)], dim=0)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('max_pool.shape:', max_pool.shape)

        # hidden_state = self.dropout0(torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1))

        pool_cat_news = torch.cat([hidden_news_state[-1],
                                   avg_pool_news, max_pool_news], dim=1)
        pool_cat_titles = torch.cat([hidden_titles_state[-1],
                                     avg_pool_titles, max_pool_titles], dim=1)

        # pool_cat.shape = (BS, 8 * HIDDEN_SIZE)
        # print('pool_cat.shape:', pool_cat.shape)

        output_news = self.fc_news1(self.dropout_news0(pool_cat_news))
        output_titles = self.fc_title1(self.dropout_title0(pool_cat_titles))
        # output_news = self.norm_news1(output_news)
        # output_titles = self.norm_title1(output_titles)
        output_news = F.leaky_relu(self.dropout_news1(output_news))
        output_titles = F.leaky_relu(self.dropout_title1(output_titles))

        # -------

        features_out = F.relu(self.dropout1_X(self.fc1_X(features)))
        features_out = F.leaky_relu(self.dropout2_X(self.norm2_X(self.fc2_X(features_out))))

        # -------

        output = self.fc2(torch.cat([output_news, output_titles, features_out], dim=1))
        output = F.relu(self.dropout2(self.norm2(output)))
        output = self.fc3(output)
        return output


class ModelGRU_IND8(nn.Module):
    def __init__(self, vocab, max_len, max_title_len, feature_input_size, embedding_size=30,
                 hidden_news_size=10, hidden_title_size=10):
        super(ModelGRU_IND8, self).__init__()
        self.max_len = max_len
        self.max_title_len = max_title_len

        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=0)
        self.gru_news = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_news_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            # dropout=0.5
        )

        self.dropout_news0 = nn.Dropout(0.3)
        self.fc_news1 = nn.Linear(3 * hidden_news_size, 12)
        self.norm_news1 = nn.BatchNorm1d(12)
        self.dropout_news1 = nn.Dropout(0.2)

        # -------

        self.gru_titles = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_title_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            # dropout=0.4
        )

        self.dropout_title0 = nn.Dropout(0.5)
        self.fc_title1 = nn.Linear(3 * hidden_title_size, 8)
        self.norm_title1 = nn.BatchNorm1d(8)
        self.dropout_title1 = nn.Dropout(0.2)

        # -------

        self.fc1_X = nn.Linear(feature_input_size, 32)
        self.dropout1_X = nn.Dropout(0.3)
        self.fc2_X = nn.Linear(32, 16)
        self.norm2_X = nn.BatchNorm1d(16)
        self.dropout2_X = nn.Dropout(0.3)

        # -------

        self.fc2 = nn.Linear(16 + 8 + 12, 8)
        self.norm2 = nn.BatchNorm1d(8)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(8, 1)

    def forward(self, text, text_len, title, title_len, features):
        # text.shape = (BS, MAX_LEN)
        # print("text.shape:", text.shape)

        embedded_news = self.embedding(text)
        embedded_titles = self.embedding(title)
        # embedded.shape = (BS, MAX_LEN, EMBED_SIZE)
        # print("embedded.shape:", embedded.shape)

        packed_news_in = nn.utils.rnn.pack_padded_sequence(
            embedded_news, text_len,
            batch_first=True, enforce_sorted=False
        )
        packed_titles_in = nn.utils.rnn.pack_padded_sequence(
            embedded_titles, title_len,
            batch_first=True, enforce_sorted=False
        )
        packed_news_out, hidden_news_state = self.gru_news(packed_news_in)
        packed_titles_out, hidden_titles_state = self.gru_titles(packed_titles_in)

        out_news, lengths_news = nn.utils.rnn.pad_packed_sequence(packed_news_out,
                                                                  batch_first=True,
                                                                  total_length=self.max_len)
        out_titles, lengths_titles = nn.utils.rnn.pad_packed_sequence(packed_titles_out,
                                                                      batch_first=True,
                                                                      total_length=self.max_title_len)
        # out.shape = (BS, MAX_LEN, 2 * HIDDEN_SIZE), hidden_state.shape = ()
        # print("out.shape:", out.shape)
        # print("hidden_state.shape:", hidden_state.shape)

        avg_pool_news = torch.sum(out_news, dim=1) / lengths_news.to(device).view(-1, 1)
        avg_pool_titles = torch.sum(out_titles, dim=1) / lengths_titles.to(device).view(-1, 1)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('avg_pool.shape:', avg_pool.shape)

        max_pool_news = torch.cat([torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out_news, lengths_news)],
                                  dim=0)
        max_pool_titles = torch.cat(
            [torch.max(o[:l], dim=0)[0].view(1, -1) for o, l in zip(out_titles, lengths_titles)], dim=0)
        # avg_pool.shape = (BS, 2 * HIDDEN_SIZE)
        # print('max_pool.shape:', max_pool.shape)

        # hidden_state = self.dropout0(torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1))

        pool_cat_news = torch.cat([hidden_news_state[-1],
                                   avg_pool_news, max_pool_news], dim=1)
        pool_cat_titles = torch.cat([hidden_titles_state[-1],
                                     avg_pool_titles, max_pool_titles], dim=1)

        # pool_cat.shape = (BS, 8 * HIDDEN_SIZE)
        # print('pool_cat.shape:', pool_cat.shape)

        output_news = self.fc_news1(self.dropout_news0(pool_cat_news))
        output_titles = self.fc_title1(self.dropout_title0(pool_cat_titles))
        # output_news = self.norm_news1(output_news)
        # output_titles = self.norm_title1(output_titles)
        output_news = F.leaky_relu(self.dropout_news1(output_news))
        output_titles = F.leaky_relu(self.dropout_title1(output_titles))

        # -------

        features_out = F.relu(self.dropout1_X(self.fc1_X(features)))
        features_out = F.leaky_relu(self.dropout2_X(self.norm2_X(self.fc2_X(features_out))))

        # -------

        output = self.fc2(torch.cat([output_news, output_titles, features_out], dim=1))
        output = F.relu(self.norm2(output))
        output = self.fc3(output)
        return output
