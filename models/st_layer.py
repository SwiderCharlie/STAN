import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.temporal_block import TemporalBlock
from models.spatial_block import SpatialBlock


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list

        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


# Gated Fusion操作
# z = sigmoid(H_S x W_S + H_T x W_T + b)
# H = z * H_S + (1 - z) * H_T
class gatedFusion(nn.Module):
    '''
    gated fusion
    HS: [batch_size, num_step, num_vertex, D]
    HT: [batch_size, num_step, num_vertex, D]
    D: output dims
    return: [batch_size, num_step, num_vertex, D]
    '''
    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


# ST-Layer：时间块和空间块的输出通过Gated Fusion自适应融合
class STLayer(nn.Module):
    def __init__(self, d_model, num_nodes, d_hidden_mt, num_heads, dropout, alpha, no_meta_temporal, no_meta_spatial):
        super(STLayer, self).__init__()
        self.temporal_block = TemporalBlock(d_model, d_hidden_mt, num_heads, dropout, no_meta_temporal)
        self.spatial_block = SpatialBlock(d_model, num_nodes, d_hidden_mt, num_heads, dropout, alpha, no_meta_spatial)
        self.gateFusion = gatedFusion(d_model, 0.1)

    def forward(self, X, adj, in_degree, out_degree):
        HT = self.temporal_block(X)
        HS = self.spatial_block(X, adj, in_degree, out_degree)
        output = self.gateFusion(HS, HT)
        del HS, HT
        return output
