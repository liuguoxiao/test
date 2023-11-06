import torch.nn as nn
import torch
from torch.nn import TransformerEncoderLayer
from Attention.attention import Attention

# from transformers import GPT2LMHeadModel, GPT2Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):

    def __init__(self, model, data, hidden_size, num_layers):
        super().__init__()
        self.model = model
        # batch_size, length, input_dim = data.shape[0], data.shape[1], data.shape[2]  # batch_size代表数据总长度
        batch_size, length, input_dim = data.shape[0], data.shape[1], 1  # batch_size代表数据总长度
        self.batch_size, self.length, self.input_dim = batch_size, length, input_dim
        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True).to(device)
        self.lstm_mlp = nn.Sequential(
            nn.Linear((length - 1) * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(device)
        self.lstm_linear = nn.Linear(hidden_size, 1).to(device)
        # GRU
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True).to(device)
        self.gru_linear = nn.Linear(hidden_size, 1).to(device)
        self.gru_c1 = nn.Sequential(
            nn.Linear((length - 1) * hidden_size, hidden_size),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        ).to(device)
        self.gru_c2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)
        # self.gru_c1 = nn.Linear((length - 1) * hidden_size, hidden_size)
        # self.gru_c2 = nn.ReLU()
        # self.gru_c3 = nn.Linear(hidden_size, 1)

        # BiLSTM
        self.bilstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
        self.bilstm_linear = nn.Linear(hidden_size * 2, length).to(device)
        self.bilstm_mlp = nn.Sequential(
            nn.Linear((length - 1) * hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(device)
        # seq2seq
        self.encoder = nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=True).to(device)
        self.decoder = nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=True).to(device)
        self.S2S_mlp = nn.Linear(hidden_size, 1).to(device)

        # Transformer
        self.transformer_layer = TransformerEncoderLayer(input_dim, nhead=1, dim_feedforward=256, dropout=0.1).to(
            device)
        self.transformer_linear = nn.Linear(input_dim, 1).to(device)
        self.transformer_mlp = nn.Sequential(nn.Linear((length - 1) * input_dim, input_dim),
                                             nn.ReLU(),
                                             nn.Linear(input_dim, 1)
                                             ).to(device)
        # att
        self.linear_layers = nn.ModuleList([nn.Linear(int(hidden_size / input_dim), hidden_size) for _ in range(3)]).to(
            device)
        self.attention = Attention()
        self.att_linear = nn.Linear(hidden_size * input_dim, 1).to(device)
        # MLP
        self.mlp_layer = nn.Sequential(
            nn.Linear((length - 1) * input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim),
            # nn.ReLU(inplace=True)
        ).to(device)
        self.mlp_layer_c1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, 1))
        # CNN_LSTM
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=8, kernel_size=3, padding=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1).to(device)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1).to(device)
        self.cnn_lstm = nn.LSTM(input_dim + 8, hidden_size, batch_first=True).to(device)
        self.cnn_lstm_linear = nn.Linear(hidden_size, 1).to(device)
        self.leakyrelu = nn.LeakyReLU().to(device)
        # Casual
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=2, dilation=1)
        self.conv1d_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=2, dilation=1)
        # padding =（kernel_size-1）*dilation）
        self.casual_lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True).to(device)
        self.casual_mlp = nn.Sequential(
            nn.Linear((length - 1) * hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        ).to(device)

    def forward(self, diff_before_result):
        if self.model == 'LSTM':
            train_before_data = diff_before_result[:, :-1, :]
            train_before_output, _ = self.lstm(train_before_data)
            # train_before_output = self.lstm_linear(train_before_output[:, -1, :])
            train_before_output = train_before_output.reshape(diff_before_result.shape[0], -1)
            # train_after_output = train_after_output.reshape(diff_flip_after_result.shape[0], -1)
            # train_before_output = self.lstm_mlp(train_before_output)
            # train_after_output = self.lstm_mlp(train_after_output)
            # output = (train_before_output + train_after_output) / 2
            # output = train_after_output
            output = train_before_output
            return output, diff_before_result[:, -1:, -1]
        elif self.model == 'Transformer':
            train_before_data = diff_before_result[:, :-1, :]
            bc = train_before_data.shape[0]
            # train_after_data = diff_flip_after_result[:, :-1, :]
            train_before_output = self.transformer_layer(train_before_data)
            # train_after_output = self.transformer_layer(train_after_data)
            # train_before_output = self.transformer_linear(train_before_output[:, -1, :])
            train_before_output = self.transformer_mlp(train_before_output.reshape(bc, -1))
            # train_after_output = self.transformer_linear(train_after_output[:, -1, :])
            # output = (train_before_output + train_after_output) / 2
            output = train_before_output
            # output = train_after_output
            return output, diff_before_result[:, -1:, -1]
        elif self.model == 'LSTM_att':
            train_before_data = diff_before_result[:, :-1, :]
            # train_after_data = diff_flip_after_result[:, :-1, :]
            train_before_output, _ = self.lstm(train_before_data)
            # train_after_output, _ = self.lstm(train_after_data)
            output = train_before_output
            # output = train_after_output
            # output = train_before_output + train_after_output
            B, L, H = output.shape[0], output.shape[1], output.shape[2]
            x = output.reshape(B, L, self.input_dim, int(H / self.input_dim))
            query, key, value = [l(x) for l, x in zip(self.linear_layers, (x, x, x))]
            att_x, attn = self.attention(query, key, value, dropout_value=0.1)
            att_x = att_x.reshape(B, L, self.input_dim * H)
            output = self.att_linear(att_x[:, -1, :])
            return output, diff_before_result[:, -1:, -1]
        elif self.model == 'Seq2Seq':
            train_before_data = diff_before_result[:, :-1, :]
            e_input = train_before_data
            e_output, e_h_c = self.encoder(e_input)
            d_input = train_before_data
            d_h_c = e_h_c
            d_output, d_h_c = self.decoder(d_input, d_h_c)
            train_before_output = self.S2S_mlp(d_output[:, -1, :])
            output = train_before_output
            return output, diff_before_result[:, -1:, -1]
        elif self.model == 'MLP':
            train_before_data = diff_before_result[:, :-1, :]
            bc = train_before_data.shape[0]
            hidden_1 = self.mlp_layer(train_before_data.reshape(bc, -1))
            hidden = self.normalize(hidden_1)
            train_before_output = self.mlp_layer_c1(hidden_1)
            # train_before_output = self.mlp_layer(train_before_data.reshape(bc, -1))
            output = train_before_output
            return output, diff_before_result[:, -1:, -1], hidden
        elif self.model == 'CNN_LSTM':
            train_before_data = diff_before_result[:, :-1, :]
            cnn_out1 = self.conv1(train_before_data.transpose(1, 2))
            cnn_out2 = self.conv2(cnn_out1)
            cnn_out3 = self.conv3(cnn_out2)
            M1_data = cnn_out3.transpose(1, 2)
            M1_data = torch.cat((train_before_data, M1_data), dim=2)
            train_before_output, _ = self.cnn_lstm(M1_data)
            bc = train_before_output.shape[0]
            c1 = self.gru_c1(train_before_output.reshape(bc, -1))
            c1 = self.normalize(c1)
            train_before_output = self.gru_c2(c1)
            # train_before_output = self.cnn_lstm_linear(train_before_output[:, -1, :])
            output = train_before_output
            return output, diff_before_result[:, -1:, -1], c1
        elif self.model == 'GRU':
            train_before_data = diff_before_result[:, :-1, :]
            train_before_output, _ = self.gru(train_before_data)
            bc = train_before_output.shape[0]
            c1 = self.gru_c1(train_before_output.reshape(bc, -1))
            c = self.normalize(c1)
            train_before_output = self.gru_c2(c)
            # train_before_output = self.gru_mlp(train_before_output.reshape(bc, -1))
            # train_before_output = self.gru_linear(train_before_output[:, -1, :])
            output = train_before_output
            return output, diff_before_result[:, -1:, -1], c
        elif self.model == 'BiLSTM':
            train_before_data = diff_before_result[:, :-1, :]
            cnn_out = self.conv1d(train_before_data.transpose(1, 2))
            cnn_out = self.conv1d_1(cnn_out)
            cnn_out = cnn_out[:, :, :-self.conv1d_1.padding[0]]
            cnn_out = cnn_out[:, :, :-self.conv1d.padding[0]].transpose(1, 2)
            train_before_output, _ = self.bilstm(train_before_data)
            # train_before_output = self.bilstm_linear(train_before_output[:, -1, :])
            bc = train_before_data.shape[0]
            train_before_output = self.bilstm_mlp(train_before_output.reshape(bc, -1))
            output = train_before_output
            return output, diff_before_result[:, -1:, -1]
        elif self.model == 'CausalConv1d_LSTM':
            train_before_data = diff_before_result[:, :-1, :]
            cnn_out = self.conv1d(train_before_data.transpose(1, 2))
            cnn_out = self.conv1d_1(cnn_out)
            cnn_out = cnn_out[:, :, :-self.conv1d_1.padding[0]]
            cnn_out = cnn_out[:, :, :-self.conv1d.padding[0]].transpose(1, 2)
            train_before_output, _ = self.casual_lstm(cnn_out)
            bc = train_before_output.shape[0]
            train_before_output = self.casual_mlp(train_before_output.reshape(bc, -1))
            output = train_before_output
            return output, diff_before_result[:, -1:, -1]

    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output
