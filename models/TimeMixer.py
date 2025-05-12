import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Autoformer_EncDec import series_decomp
from models.Embed import DataEmbedding_wo_pos
from models.StandardNorm import Normalize
import argparse
import math


class TimeMixer(nn.Module):

    def __init__(self, configs):
        super(TimeMixer, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence

        self.layer = configs.e_layers

        self.mix_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                5 * (2 ** (configs.down_sampling_layers - i)),
                configs.d_model,
            )
            for i in range(configs.down_sampling_layers + 1)
        ])

    def __multi_scale_process_inputs(self, x_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1)) # 第一行不做处理

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        x_enc = x_enc_sampling_list
        return x_enc

    def forecast(self, x_enc):
        x_enc = self.__multi_scale_process_inputs(x_enc)

        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc, ):
            x_mix = self.mix_layers[i](x.reshape(x.shape[0], -1))
            x_list.append(x_mix)

        dec_out = torch.stack(x_list, dim=-1).sum(-1)

        return dec_out

    def forward(self, x_enc):
        out = []
        for feature_idx in range(x_enc.shape[2]):  # 按每个城市来进行预测
            dec_out = self.forecast(x_enc[:, :, feature_idx, :])
            out.append(dec_out)
        out = torch.stack(out)
        out = out.transpose(1, 0)
        return torch.sigmoid(out)

    @staticmethod
    def timemixer_arg_set(seq_len, pre_len):

        args = argparse.Namespace()

        # basic config
        args.task_name = 'short_term_forecast'
        args.is_training = 1
        args.model_id = 'test'
        args.model = 'Autoformer'

        # data loader
        args.features = 'MS'
        args.target = 'OT'
        args.freq = 'h'

        # forecasting task
        args.seq_len = seq_len
        args.label_len = seq_len
        args.pred_len = pre_len
        args.seasonal_patterns = 'Monthly'
        args.inverse = False

        # model define
        args.top_k = 5
        args.num_kernels = 6
        args.enc_in = 6
        args.dec_in = 6
        args.c_out = 1
        args.d_model = 128
        args.n_heads = 4
        args.e_layers = 1
        args.d_layers = 1
        args.d_ff = 32
        args.moving_avg = 25
        args.factor = 1
        args.distil = False
        args.dropout = 0.1
        args.embed = 'timeF'
        args.activation = 'gelu'
        args.output_attention = True
        args.channel_independence = 0  ### 0：多变量通道混合，1：多变量通道独立
        args.decomp_method = 'moving_avg'
        args.use_norm = 1
        args.down_sampling_window = 2
        args.down_sampling_layers = int(math.log2(seq_len))
        args.down_sampling_method = 'avg'
        args.use_future_temporal_feature = 0

        # imputation task
        args.mask_rate = 0.25

        # anomaly detection task
        args.anomaly_ratio = 0.25

        # de-stationary projector params
        args.p_hidden_dims = [128, 128]
        args.p_hidden_layers = 1

        return args
