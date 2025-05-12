# 参考网址:https://blog.csdn.net/weixin_44791964/article/details/121371986

import torch
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

class LuongAttention(nn.Module):
    def __init__(self, dense_dim, seq_len, pre_len):
        super(LuongAttention, self).__init__()
        self.pre_len = pre_len
        self.seq_len = seq_len
        self.linear = nn.Linear(dense_dim, 1) # 全连接层

    def forward(self, encoder_hindens, decoder_hidens):
        out_list = []
        temp_score_list =[] # 当前批次的score
        temp_vt_list = [] # 当前批次的vt23
        batch_size, city_num, dim = encoder_hindens[0].shape
        for de_hiden in decoder_hidens: #解码器隐藏状态
            a_list = [] # 当前批次的a
            for en_hidne in encoder_hindens:   #获取编码器隐藏状态，并计算所有score
                score = torch.matmul(de_hiden, en_hidne.transpose(-1, -2)) # 点积 h_de 和 h_en d，计算score
                temp_score_list.append(score)
            for score in temp_score_list:  # 计算所有a值,公式: a = score_i/sum(scores)
                a = score / (sum(temp_score_list)) # 计算 a 的值
                a_list.append(a)
            # 公式: vt = t - 1 * sum( a * h_en )，bug与隐患:经过计算后的vt一定是大于1，是否需要重新归一化或者其他操作
            # 将 a 和 h_en 对应位置相乘
            multipied_list =[]
            for i in range(0, self.seq_len):
                a = a_list[i] #############################
                h_en = encoder_hindens[i] # 取得隐藏状态
                a = torch.matmul(a, h_en)
                multipied = h_en * a
                multipied_list.append(multipied)
            # 得到vt
            tensor_look_back = self.seq_len * torch.ones(batch_size, city_num, dim).cuda()
            # debug = sum(multipied_list)
            vt = tensor_look_back - 1 * sum(multipied_list) ## bug与隐患，vt必大于1

            ## 归一化
            #normalized_tensor = vt / self.seq_len
            # 对每个特征进行归一化
            min_values = vt.min(dim=-1, keepdim=True)[0]
            max_values = vt.max(dim=-1, keepdim=True)[0]

            normalized_tensor = (vt - min_values) / (max_values - min_values)
            out_list.append(normalized_tensor) # 当前批次的vt全部进入列表
        out_list = torch.stack(out_list, dim=0)
        return out_list

#自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.02, alpha=0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha  # 学习因子
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # 建立都是0的矩阵，大小为（输入维度，输出维度）
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  # 见下图
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(self.W, input)
        # print(h.shape)  torch.Size([2708, 8]) 8是label的个数
        N = h.size()[0]
        # print(N)  2708 nodes的个数
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)  # 见下图
        a_input = a_input.view(N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # 即论文里的eij
        # squeeze除去维数为1的维度

        zero_vec = -9e15 * torch.ones_like(e).transpose(1, 0)

        #attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.mm(e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # 对应论文公式3，attention就是公式里的αij
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)
        #if self.concat:
            #return F.elu(h_prime)
        #else:
            #return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# 1、SE-通道注意力
'''
    SENet是通道注意力机制的典型实现。
    2017年提出的SENet是最后一届ImageNet竞赛的冠军，其实现示意图如下所示，对于输入进来的特征层，我们关注其每一个通道的权重，
    对于SENet而言，其重点是获得输入进来的特征层，每一个通道的权值。利用SENet，我们可以让网络关注它最需要关注的通道。
    其具体实现方式就是：
    1、对输入进来的特征层进行全局平均池化。
    2、然后进行两次全连接，第一次全连接神经元个数较少，第二次全连接神经元个数和输入特征层相同。
    3、在完成两次全连接后，我们再取一次Sigmoid将值固定到0-1之间，此时我们获得了输入特征层每一个通道的权值（0-1之间）。
    4、在获得这个权值后，我们将这个权值乘上原输入特征层即可。
'''
class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 2、CBAM-通道注意力机制和空间注意力机制
'''
    CBAM将通道注意力机制和空间注意力机制进行一个结合，相比于SENet只关注通道的注意力机制可以取得更好的效果。
    
    1.图像的上半部分为通道注意力机制，通道注意力机制的实现可以分为两个部分，我们会对输入进来的单个特征层，分别进行全局平均池化和全局最大池化。
    之后对平均池化和最大池化的结果，利用共享的全连接层进行处理，我们会对处理后的两个结果进行相加，然后取一个sigmoid，
    此时我们获得了输入特征层每一个通道的权值（0-1之间）。在获得这个权值后，我们将这个权值乘上原输入特征层即可。
    图像的下半部分为空间注意力机制，我们会对输入进来的特征层，在每一个特征点的通道上取最大值和平均值。之后将这两个结果进行一个堆叠，
    利用一次通道数为1的卷积调整通道数，然后取一个sigmoid，此时我们获得了输入特征层每一个特征点的权值（0-1之间）。
    在获得这个权值后，我们将这个权值乘上原输入特征层即可。
'''

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


# 3、ECA注意力
'''
ECANet是也是通道注意力机制的一种实现形式。ECANet可以看作是SENet的改进版。ECANet的作者认为SENet对通道注意力机制的预测带来了副作用，
捕获所有通道的依赖关系是低效并且是不必要的。在ECANet的论文中，作者认为卷积具有良好的跨通道信息获取能力。
ECA模块的思想是非常简单的，它去除了原来SE模块中的全连接层，直接在全局平均池化之后的特征上通过一个1D卷积进行学习。
既然使用到了1D卷积，那么1D卷积的卷积核大小的选择就变得非常重要了，了解过卷积原理的同学很快就可以明白，1D卷积的卷积核大小会影响注意力机制
每个权重的计算要考虑的通道数量。用更专业的名词就是跨通道交互的覆盖率。
'''


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


