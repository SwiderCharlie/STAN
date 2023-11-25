import torch
import torch.nn as nn
import math


# 使用MLP学习Q、K、V的参数
class MetaLearner(nn.Module):
    def __init__(self, d_model, d_hidden_mt, num_heads):
        super(MetaLearner, self).__init__()
        self.num_heads = num_heads
        self.attn_size = d_model // num_heads
        self.linear1 = nn.Linear(d_model, d_hidden_mt)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden_mt, 3 * num_heads * self.attn_size * d_model)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, d_model = inputs.shape
        out = self.relu(self.linear1(inputs))
        out = self.linear2(out)
        out = out.reshape((batch_size, seq_len, num_nodes, 3, self.num_heads, self.attn_size, d_model))  # (B, T, N, 3, H, d_k, d_model)
        out = out.permute((3, 0, 1, 2, 4, 5, 6))  # (3, B, T, N, H, d_k, d_model)
        return out


# 使用Meta-Learning学习到的W_Q、W_K、W_V计算Q、K、V
def multihead_linear_transform(W, inputs):
    batch_size, seq_len, num_nodes, num_heads, attn_size, d_model = W.shape
    inputs = inputs.reshape((batch_size, seq_len, num_nodes, 1, d_model, 1))
    # (B, T, N, H, d_k, d_model) x (B, T, N, 1, d_model, 1) -> (B, T, N, H, d_k)
    out = torch.matmul(W, inputs).squeeze(-1)
    return out


# 时间no-shaerd注意力机制
class TemporalAttention(nn.Module):
    def __init__(self, d_model, d_hidden_mt, num_heads, noMeta):
        super(TemporalAttention, self).__init__()
        self.noMeta = noMeta
        self.num_heads = num_heads
        if self.noMeta:
            self.linear_q = nn.Linear(d_model, d_model, bias=False)
            self.linear_k = nn.Linear(d_model, d_model, bias=False)
            self.linear_v = nn.Linear(d_model, d_model, bias=False)
        else:
            self.meta_learner = MetaLearner(d_model, d_hidden_mt, num_heads)

        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, num_nodes, d_model)
        batch_size, seq_len, num_nodes, d_model = inputs.shape
        attn_size = d_model // self.num_heads

        if self.noMeta:
            Q = self.linear_q(inputs).reshape((batch_size, seq_len, num_nodes, self.num_heads, attn_size))
            K = self.linear_k(inputs).reshape((batch_size, seq_len, num_nodes, self.num_heads, attn_size))
            V = self.linear_v(inputs).reshape((batch_size, seq_len, num_nodes, self.num_heads, attn_size))
        else:
            W_q, W_k, W_v = self.meta_learner(inputs)
            Q = multihead_linear_transform(W_q, inputs)
            K = multihead_linear_transform(W_k, inputs)
            V = multihead_linear_transform(W_v, inputs)

        Q = Q.permute((0, 2, 3, 1, 4))  # (B, N, H, T, d_k)
        K = K.permute((0, 2, 3, 4, 1))
        V = V.permute((0, 2, 3, 1, 4))

        alpha = torch.matmul(Q, K) / math.sqrt(attn_size)
        alpha = torch.softmax(alpha, dim=-1)
        # np.save('TA_marix.npy', alpha.detach().cpu().numpy())
        out = torch.matmul(alpha, V)
        out = out.permute((0, 3, 1, 2, 4))  # (B, T, N, H, d_k)
        out = out.reshape((batch_size, seq_len, num_nodes, d_model))

        out = self.linear(out)  # (B, T, N, d_model)
        out = self.layer_norm(out + inputs)

        return out


# 空间no-shared注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, d_model, num_nodes, d_hidden_mt, num_heads, dropout, noMeta):
        super(SpatialAttention, self).__init__()
        self.noMeta = noMeta
        self.num_heads = num_heads
        if self.noMeta:
            self.linear_q = nn.Linear(d_model, d_model, bias=False)
            self.linear_k = nn.Linear(d_model, d_model, bias=False)
            self.linear_v = nn.Linear(d_model, d_model, bias=False)

        else:
            self.meta_learner = MetaLearner(d_model, d_hidden_mt, num_heads)

        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

        self.spatial_pos_encoder = nn.Linear(1, num_heads)
        self.edge_encoder = nn.Linear(1, num_heads)
        self.edge_dis_encoder = nn.Embedding(num_nodes * num_heads * num_heads, 1)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, num_nodes, d_model)
        batch_size, seq_len, num_nodes, d_model = inputs.shape
        attn_size = d_model // self.num_heads

        if self.noMeta:
            Q = self.linear_q(inputs).reshape((batch_size, seq_len, num_nodes, self.num_heads, attn_size))
            K = self.linear_k(inputs).reshape((batch_size, seq_len, num_nodes, self.num_heads, attn_size))
            V = self.linear_v(inputs).reshape((batch_size, seq_len, num_nodes, self.num_heads, attn_size))
        else:
            W_q, W_k, W_v = self.meta_learner(inputs)
            Q = multihead_linear_transform(W_q, inputs)
            K = multihead_linear_transform(W_k, inputs)
            V = multihead_linear_transform(W_v, inputs)

        Q = Q.permute((0, 1, 3, 2, 4))  # (B, T, H, N, d_k)
        K = K.permute((0, 1, 3, 2, 4)).transpose(-1, -2)
        V = V.permute((0, 1, 3, 2, 4))

        alpha = torch.matmul(Q, K) / math.sqrt(attn_size)
        alpha = torch.softmax(alpha, dim=-1)
        # np.save('SA_marix.npy', alpha.detach().cpu().numpy())
        out = torch.matmul(alpha, V)

        out = out.permute((0, 1, 3, 2, 4))  # (B, T, N, H, d_k)
        out = out.reshape((batch_size, seq_len, num_nodes, d_model))

        out = self.linear(out)  # (B, T, N, d_model)
        out = self.dropout(out)
        out = self.layer_norm(out + inputs)

        return out
