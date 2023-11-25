import torch
import torch.nn as nn
import torch.nn.functional as F
from models.noshared_attn import SpatialAttention


# GAT
class GATLayer(nn.Module):
    def __init__(self, d_model, dropout, alpha):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.d_model = d_model

        self.W = nn.Parameter(torch.empty(size=(d_model, d_model)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * d_model, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inputs, adj):
        Wh = torch.mm(inputs, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.d_model, :])
        Wh2 = torch.matmul(Wh, self.a[self.d_model:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


# 空间块
class SpatialBlock(nn.Module):
    def __init__(self, d_model, num_nodes, d_hidden_mt, num_heads, dropout, alpha, noMeta):
        super(SpatialBlock, self).__init__()
        self.in_degree_emb = nn.Embedding(num_nodes, d_model, padding_idx=0)
        self.out_degree_emb = nn.Embedding(num_nodes, d_model, padding_idx=0)
        self.gat = GATLayer(d_model, dropout, alpha)
        self.spatial_attn = SpatialAttention(d_model, num_nodes, d_hidden_mt, num_heads, dropout, noMeta)
        self.pred_linear = nn.Linear(2 * d_model, d_model)

    def forward(self, inputs, adj, in_degree, out_degree):
        # 关于节点入度和出度的中心编码
        spatial_emb = self.in_degree_emb(in_degree) + self.out_degree_emb(out_degree)
        batch_size, seq_len, num_nodes, d_model = inputs.shape
        out1 = inputs.reshape(batch_size * seq_len, num_nodes, d_model)
        out1_list = [self.gat(sample, adj) for sample in out1]
        out1 = torch.stack(out1_list).reshape(batch_size, seq_len, num_nodes, d_model)
        del out1_list
        out2 = self.spatial_attn(inputs + spatial_emb)
        # 线性层输出
        output = self.pred_linear(torch.cat([out1, out2], dim=-1))

        return output
