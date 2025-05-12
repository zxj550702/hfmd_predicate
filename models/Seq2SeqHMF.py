import argparse
import torch
import torch.nn as nn
from models.dctnet import dct_channel_block
from models.tgcn import TGCNCell

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class STPECell(nn.Module):
    def __init__(self, adj, hidden_dim, i):
        super(STPECell, self).__init__()
        self.i = i
        self.adj = adj
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.TGCN = TGCNCell(adj, self._input_dim, self._hidden_dim)
        self.BiLSTM = nn.LSTM(hidden_dim if i == 0 else hidden_dim * 3, hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, inputs, hidden_state, input_spa_state_vector, input_tem_state_vector):
        batch_size, num_nodes, _ = inputs.shape
        xx = inputs

        for j in range(2):  # 临近节点步长
            output, hidden_state = self.TGCN(input_spa_state_vector, hidden_state)
        output_spatial = output.reshape(batch_size, num_nodes, self._hidden_dim)

        encoder_h_each_time = []
        for k in range(self._input_dim):
            xxx = xx[:, k, :].reshape(batch_size, 1, -1)
            if self.i > 0:
                each_node_last_hidden = input_tem_state_vector[:, k, :].reshape(batch_size, 1, -1)
                next_ = torch.cat((xxx, each_node_last_hidden), dim=2)
            else:
                next_ = xxx
            tem_out, tem_hidden = self.BiLSTM(next_)  # bilstm
            encoder_h_each_time.append(tem_out)
        output_temporal = torch.stack(encoder_h_each_time, dim=2).reshape(batch_size, num_nodes, -1)
        return output_spatial, output_temporal


class Encoder(nn.Module):
    def __init__(self, adj, hidden_dim, seq_len):
        super(Encoder, self).__init__()
        self.adj = adj
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.Encoders = nn.ModuleList(
            [STPECell(self.adj, self._hidden_dim, i) for i in range(seq_len)]
        )
        self.conv1 = nn.Conv1d(7, out_channels=hidden_dim, kernel_size=1)
        self.linear = nn.Linear(self._hidden_dim, 1)
        self.STpool = nn.AvgPool1d(kernel_size=1, stride=2, padding=0)
        self.convert_mlp = nn.Sequential(
            nn.Linear(seq_len - 1, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, _ = inputs.shape
        combine = inputs
        assert self._input_dim == num_nodes
        output_s = torch.zeros(batch_size, num_nodes * self._hidden_dim).cuda()
        output_t = torch.zeros(batch_size, num_nodes, self._hidden_dim).cuda()
        hibridStateVector = []
        for i in range(seq_len):
            x = self.conv1(combine[:, i, :, :].transpose(1, 2)).transpose(1, 2)
            if i > 0:
                vt = torch.concat((x, output_s), dim=2)
                vt = self.STpool(vt)
            else:
                vt = x
            vt = self.linear(vt).reshape(batch_size, -1)
            output_s, output_t = self.Encoders[i](x, output_s.reshape(batch_size, -1), vt, output_t)
            hibridStateVector.append(output_s + self.STpool(output_t))
        encoder_h_each_node = torch.stack(hibridStateVector)
        last_encoder_h_each_node = encoder_h_each_node[seq_len - 1, :, :, :]  # 取出权重最大的最后一步t
        pre_times_out = []
        for i in range(seq_len - 1):
            pre_times_out.append(encoder_h_each_node[i, :, :, :])
        pre_times_out = torch.stack(pre_times_out).transpose(0, 3)
        weight_state_vector = self.convert_mlp(pre_times_out).transpose(0, 3).reshape(batch_size, num_nodes, -1)  # 前n-1步学习一个权重矩阵
        final_state = last_encoder_h_each_node * weight_state_vector
        return final_state


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
        self.fecam = dct_channel_block(self._input_dim)

    def forward(self, inputs):
        decoder_last_hidden = inputs
        batch_size, num_nodes, _ = decoder_last_hidden.shape
        y_list = None
        y = None

        for i, predictor in enumerate(self.Predictors):
            decoder_h_each_time = []
            for k in range(num_nodes):  # 按节点进行解码与预测
                hh = decoder_last_hidden[:, k, :].reshape(batch_size, 1, -1)
                if i > 0:
                    next_ = torch.cat((hh, y[:, k, :].reshape(batch_size, 1, -1)), dim=2)  # 多步预测，隐藏状态拼接上一次的预测值
                else:
                    next_ = hh
                do, dh = self.Decoders[i](next_.transpose(0, 1))  # 解码
                decoder_h_each_time.append(do[0, :, :])
            # 得到当次解码的总隐藏状态
            decoder_tensor_each_time = torch.stack(decoder_h_each_time)
            y = predictor(decoder_tensor_each_time).transpose(0, 1)  # 预测
            y_list = y if i == 0 else torch.cat((y_list, y), dim=2)  # 记录预测值
            decoder_last_hidden = decoder_tensor_each_time.transpose(0, 1)  # 更新上一次的hidden
        return self.fecam(y_list)

# class STPECell(nn.Module):
#     def __init__(self, adj, hidden_dim, i):
#         super(STPECell, self).__init__()
#         self.i = i
#         self.adj = adj
#         self._input_dim = adj.shape[0]
#         self._hidden_dim = hidden_dim
#         self.TGCN = TGCNCell(adj, self._input_dim, self._hidden_dim)
#         self.BiLSTM = nn.LSTM(hidden_dim if i == 0 else hidden_dim * 3, hidden_dim, num_layers=1, bidirectional=True)
#
#     def forward(self, inputs, hidden_state, input_spa_state_vector, input_tem_state_vector):
#         batch_size, num_nodes, _ = inputs.shape
#         xx = inputs
#
#         for j in range(2):  # 临近节点步长
#             output, hidden_state = self.TGCN(input_spa_state_vector, hidden_state)
#         output_spatial = output.reshape(batch_size, num_nodes, self._hidden_dim)
#
#         if self.i > 0:
#             each_node_last_hidden = input_tem_state_vector
#             next_ = torch.cat((xx, each_node_last_hidden), dim=2)
#         else:
#             next_ = xx
#         tem_out, tem_hidden = self.BiLSTM(next_)  # bilstm
#         output_temporal = tem_out
#         return output_spatial, output_temporal
#
#
# class Encoder(nn.Module):
#     def __init__(self, adj, hidden_dim, seq_len):
#         super(Encoder, self).__init__()
#         self.adj = adj
#         self._input_dim = adj.shape[0]
#         self._hidden_dim = hidden_dim
#         self.Encoders = nn.ModuleList(
#             [STPECell(self.adj, self._hidden_dim, i) for i in range(seq_len)]
#         )
#         self.conv1 = nn.Conv1d(7, out_channels=hidden_dim, kernel_size=1)
#         self.linear = nn.Linear(self._hidden_dim, 1)
#         self.STpool = nn.AvgPool1d(kernel_size=1, stride=2, padding=0)
#         self.convert_mlp = nn.Sequential(
#             nn.Linear(seq_len - 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, inputs):
#         batch_size, seq_len, num_nodes, _ = inputs.shape
#         combine = inputs
#         assert self._input_dim == num_nodes
#         output_s = torch.zeros(batch_size, num_nodes * self._hidden_dim).cuda()
#         output_t = torch.zeros(batch_size, num_nodes , self._hidden_dim).cuda()
#         hibridStateVector = []
#         for i in range(seq_len):
#             x = self.conv1(combine[:, i, :, :].transpose(1, 2)).transpose(1, 2)
#             if i > 0:
#                 vt = torch.concat((x, output_s), dim=2)
#                 vt = self.STpool(vt)
#             else:
#                 vt = x
#             vt = self.linear(vt).reshape(batch_size, -1)
#             output_s, output_t = self.Encoders[i](x, output_s.reshape(batch_size, -1), vt, output_t)
#             hibridStateVector.append(output_s + self.STpool(output_t))
#         encoder_h_each_node = torch.stack(hibridStateVector)
#         last_encoder_h_each_node = encoder_h_each_node[seq_len - 1, :, :, :]  # 取出权重最大的最后一步t
#         pre_times_out = []
#         for i in range(seq_len - 1):
#             pre_times_out.append(encoder_h_each_node[i, :, :, :])
#         pre_times_out = torch.stack(pre_times_out).transpose(0, 3)
#         weight_state_vector = self.convert_mlp(pre_times_out).transpose(0, 3).reshape(batch_size, num_nodes, -1)  # 前n-1步学习一个权重矩阵
#         final_state = last_encoder_h_each_node * weight_state_vector
#         return final_state
#
#
# class Decoder(nn.Module):
#     def __init__(self, adj, hidden_dim, pre_len):
#         super(Decoder, self).__init__()
#         self._input_dim = adj.shape[0]
#         self._hidden_dim = hidden_dim
#         self.pre_len = pre_len
#         self.Decoders = nn.ModuleList(
#             [nn.LSTM(hidden_dim * 1 if i == 0 else hidden_dim * 2 + 1, hidden_dim, num_layers=1, bidirectional=True)
#              for i in range(pre_len)]
#         )
#         self.Predictors = (
#             nn.ModuleList(
#                 [nn.Linear(hidden_dim * 2, 1) for i in range(pre_len)]
#             )
#         )
#         self.fecam = dct_channel_block(self._input_dim)
#
#     def forward(self, inputs):
#         decoder_last_hidden = inputs
#         batch_size, num_nodes, _ = decoder_last_hidden.shape
#         y_list = None
#         y = None
#         for i, predictor in enumerate(self.Predictors):
#             hh = decoder_last_hidden
#             if i > 0:
#                 next_ = torch.cat((hh, y), dim=2)  # 多步预测，隐藏状态拼接上一次的预测值
#             else:
#                 next_ = hh
#             do, dh = self.Decoders[i](next_)  # 解码
#             # 得到当次解码的总隐藏状态
#             y = predictor(do)  # 预测
#             y_list = y if i == 0 else torch.cat((y_list, y), dim=2)  # 记录预测值
#             decoder_last_hidden = do # 更新上一次的hidden
#         return self.fecam(y_list)


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


