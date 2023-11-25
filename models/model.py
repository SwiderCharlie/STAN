import torch
import torch.nn as nn
import torch.nn.functional as F
from models.st_layer import STLayer


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.inputLinear = nn.Linear(args.input_dim, args.hidden_dim)
        self.layerList = nn.ModuleList([STLayer(args.hidden_dim, args.num_nodes, args.hidden_dim, args.num_heads,
                                                args.dropout, args.alpha, args.no_meta_temporal, args.no_meta_spatial)
                                        for _ in range(args.num_blocks)])
        self.norm = nn.GroupNorm(args.num_groups, args.hidden_dim)
        self.outLinear1 = nn.Linear(args.hidden_dim, args.feedforward_dim)
        self.outLinear2 = nn.Linear(args.feedforward_dim, 1)

    def forward(self, X, adj, in_degree, out_degree):
        """
        :param X: (batch_size, seq_len, num_nodes, c_in)
        :param adj: (num_nodes, num_nodes)
        :param in_degree: (num_nodes)
        :param out_degree: (num_nodes)
        :return:
        """
        # 输入层
        output = self.inputLinear(X)
        # 堆叠的多个ST-Layer
        for layer in self.layerList:
            output = output + layer(output, adj, in_degree, out_degree)
            output = self.norm(output.transpose(1, 3))
            output = output.transpose(1, 3)
        # 输出层
        output = self.outLinear2(F.relu(self.outLinear1(output)))
        return output

