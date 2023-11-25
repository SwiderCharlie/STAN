import torch
import torch.nn as nn
import math
from models.noshared_attn import TemporalAttention


# GRU用来建模短期时间相关性
class GRULayer(nn.Module):
    def __init__(self, d_model, dropout):
        super(GRULayer, self).__init__()
        self.gru_cell = nn.GRUCell(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, d_model = inputs.shape
        inputs = inputs.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, d_model)
        hx = torch.zeros_like(inputs[:, 0, :]).to(inputs.device)
        output = []
        for i in range(inputs.shape[1]):
            hx = self.gru_cell(inputs[:, i, :], hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        return output


# Transformer模型中的位置编码操作
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand(1, x.size(1), x.size(2), x.size(3)).to(x.device)


# 时间块
class TemporalBlock(nn.Module):
    def __init__(self, d_model, d_hidden_mt, num_heads, dropout, no_meta):
        super(TemporalBlock, self).__init__()
        self.pos_embedding = PositionalEmbedding(d_model)
        self.temporal_attn = TemporalAttention(d_model, d_hidden_mt, num_heads, no_meta)
        self.gru = GRULayer(d_model, dropout)
        self.pred_linear = nn.Linear(2 * d_model, d_model)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, d_model = inputs.shape
        out1 = self.gru(inputs).reshape(batch_size, num_nodes, seq_len, d_model).transpose(1, 2)
        out2 = self.temporal_attn(inputs + self.pos_embedding(inputs))
        output = self.pred_linear(torch.cat([out1, out2], dim=-1))

        return output
