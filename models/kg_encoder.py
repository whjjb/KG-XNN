import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class KGEncoderGAT(nn.Module):
    def __init__(self, in_dim=300, hid=256, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid, heads=heads, concat=False)
        self.gat2 = GATConv(hid, hid, heads=heads, concat=False)

    def forward(self, x, edge_index):
        h = F.elu(self.gat1(x, edge_index))
        h = self.gat2(h, edge_index)
        return h  # [N_nodes, hid]
