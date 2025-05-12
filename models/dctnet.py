from distutils.command.config import config
import torch.nn as nn
import math
import numpy as np
import torch
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim = (-d))
        r = torch.stack((t.real, t.imag), -1)
        return r

    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:,:,0], x[:,:,1]), dim = (-d))
        return t.real


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = rfft(v, 1)  # 快速傅里叶变换

    # k 是一个张量，其包含了DCT变换中各个频率分量的角频率。
    # 它通过将一个从0到N-1的等差数列乘以-π/(2*N)来计算得到，
    # 这样可以确保生成的k值覆盖了DCT变换所需的整个频率范围。
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)

    # W_r 是余弦权重张量，用于DCT变换的计算。
    W_r = torch.cos(k)

    # W_i 是正弦权重张量，尽管在标准的DCT变换中不直接使用，
    # 但在这里它被计算出来，可能是为了后续的复数操作，
    # 或者是为了与DFT（离散傅里叶变换）的计算步骤兼容。
    # 它通过计算k的正弦值得到，每个元素对应一个频率分量的正弦权重
    W_i = torch.sin(k)

    # 通过以下操作，我们将FFT的实部乘以余弦权重，并将FFT的虚部乘以正弦权重，然后相减，
    # 这样就得到了实际的DCT变换结果。
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    V = 2 * V.view(*x_shape)

    return V


class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(channel, channel*2, bias=False),
                nn.Dropout(p=0.1),
                nn.ReLU(inplace=True),
                nn.Linear(channel*2, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        o = x
        b, c, l = x.size()
        list = []
        for i in range(c):
            y = x[:, i, :]
            freq = dct(y)
            list.append(freq)
        stack_dct = torch.stack(list, dim=1)
        stack_dct = stack_dct.clone().detach()
        lr_weight = self.fc(stack_dct.transpose(1, 2))

        return o * lr_weight.transpose(1, 2) #result



