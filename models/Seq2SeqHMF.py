import argparse
import torch
import torch.nn as nn
from models.dctnet import dct_channel_block
from models.tgcn import TGCNCell

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class STPECell(nn.Module):
    def __init__(self, hidden_dim, i):
        super(STPECell, self).__init__()
        self.i = i
        self._hidden_dim = hidden_dim
        self.BiLSTM = nn.LSTM(hidden_dim if i == 0 else hidden_dim * 3, hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, conv_inputs, encoder_h_prev):
        batch_size, num_nodes, _ = conv_inputs.shape

        encoder_h_each_time = []
        for k in range(batch_size):
            xx = conv_inputs[k, :, :].reshape(1, num_nodes, -1)
            if self.i > 0 and encoder_h_prev is not None:
                prev_hidden = encoder_h_prev[k, :, :, :]
                next_ = torch.cat((xx, prev_hidden), dim=2)
            else:
                next_ = xx
            w_out, w_hidden = self.BiLSTM(next_)  # (1, num_nodes, hidden_dim*2)
            encoder_h_each_time.append(w_out)

        hidden_tensor = torch.stack(encoder_h_each_time)  # (batch_size, 1, num_nodes, hidden_dim*2)
        output_temporal = hidden_tensor[:, 0, :, :]  # (batch_size, num_nodes, hidden_dim*2)

        return output_temporal, hidden_tensor


class Encoder(nn.Module):
    def __init__(self, adj, hidden_dim, seq_len):
        super(Encoder, self).__init__()
        self.adj = adj
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.tgcn_cell = TGCNCell(adj, self._input_dim, self._hidden_dim)
        self.Encoders = nn.ModuleList(
            [STPECell(self._hidden_dim, i) for i in range(seq_len)]
        )
        self.conv1 = nn.Conv1d(5, out_channels=hidden_dim, kernel_size=1)

        self.linear = nn.Linear(5, 1)
        self.hpool = nn.AvgPool1d(kernel_size=1, stride=2, padding=0)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, _ = inputs.shape
        assert self._input_dim == num_nodes

        split_tensors = torch.split(inputs, [1, 5], dim=3)
        combine = split_tensors[1]

        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)

        output = None
        encoder_h = []  # 存储每个时间步的BiLSTM输出

        for i in range(seq_len):
            x = self.linear(combine[:, i, :, :]).reshape(batch_size, -1)  # (batch, num_nodes)
            for j in range(2):  # 临近节点步长
                output, hidden_state = self.tgcn_cell(x, hidden_state)

            x_conv = combine[:, i, :, :].transpose(1, 2)  # (batch, 5, num_nodes)
            encoder_h_each_time = []
            for k in range(batch_size):
                xx = x_conv[k, :, :].reshape(1, 5, num_nodes)  # (1, 5, num_nodes)
                xx = self.conv1(xx).transpose(1, 2)  # (1, num_nodes, hidden_dim)
                if i > 0:
                    each_node_last_hidden = encoder_h[i - 1][k, :, :, :]
                    next_ = torch.cat((xx, each_node_last_hidden), dim=2)
                else:
                    next_ = xx
                w_out, w_hidden = self.Encoders[i].BiLSTM(next_)
                encoder_h_each_time.append(w_out)

            hidden_tensor_each_time = torch.stack(encoder_h_each_time)
            encoder_h.append(hidden_tensor_each_time)

        final_hidden = encoder_h[seq_len - 1]
        final_hidden = final_hidden[:, 0, :, :]  # (batch, num_nodes, hidden_dim*2)
        w_out = self.hpool(final_hidden)  # (batch, num_nodes, hidden_dim)

        output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        w_out = w_out + output

        return w_out


class Decoder(nn.Module):

    def __init__(self, adj, hidden_dim, pre_len):
        super(Decoder, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.pre_len = pre_len
        self.Decoders = nn.ModuleList(
            [nn.LSTM(hidden_dim * 1 if i == 0 else hidden_dim * 2 + 1, hidden_dim, num_layers=1, bidirectional=True)
             for i in range(pre_len)]
        )
        self.Predictors = (
            nn.ModuleList(
                [nn.Linear(hidden_dim * 2, 1) for i in range(pre_len)]
            )
        )
        # DCT通道注意力
        self.fecam = dct_channel_block(self._input_dim)

    def forward(self, inputs):
        w_out = inputs  # (batch, num_nodes, hidden_dim)
        batch_size, num_nodes, _ = w_out.shape
        device = w_out.device

        y = torch.zeros(batch_size, num_nodes, 1, device=device)
        y_list = torch.zeros(batch_size, num_nodes, 1, device=device)

        for i, predictor in enumerate(self.Predictors):
            dh = w_out if i == 0 else torch.cat((do, y), dim=2)
            do, dhidden = self.Decoders[i](dh)
            y = predictor(do)
            y_list = y if i == 0 else torch.cat((y_list, y), dim=2)

        return self.fecam(y_list)


class Seq2SeqHMF(nn.Module):
    def __init__(self, adj, hidden_dim, pre_len, seq_len: int, **kwargs):
        super(Seq2SeqHMF, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self._adj = adj
        self.pre_len = pre_len
        self.Encoder = Encoder(self._adj, hidden_dim, seq_len)
        self.Decoder = Decoder(self._adj, hidden_dim, pre_len)

    def forward(self, inputs):
        final_hidden = self.Encoder(inputs)
        outputs = self.Decoder(final_hidden)
        return torch.tanh(outputs)

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=128)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
