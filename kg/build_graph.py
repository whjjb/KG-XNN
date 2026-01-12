import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd

def make_edge_index(edges_df: pd.DataFrame, num_nodes: int):
    # 仅使用有向边 (src, dst)
    src = edges_df["src"].to_numpy()
    dst = edges_df["dst"].to_numpy()
    data = np.ones_like(src, dtype=np.float32)
    adj = coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)  # [2, E], [E]
    return edge_index  # torch.LongTensor(2, E)
