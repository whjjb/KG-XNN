import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict, deque
import numpy as np


def _ancestors_bfs(node_id: int, parents, max_depth: int = 3):
    """
    从 node_id 出发，沿 is_a 关系向上做 BFS，返回“祖先节点集合”（不含自己）
    parents: dict[node_id] -> list[parent_id]
    """
    vis, q = set(), deque([(node_id, 0)])
    while q:
        u, dep = q.popleft()
        for p in parents.get(u, []):
            if p not in vis:
                vis.add(p)
                if dep + 1 < max_depth:
                    q.append((p, dep + 1))
    return vis


def compute_pos_weight(attr_targets: torch.Tensor, max_weight: float = 20.0) -> torch.Tensor:
    """
    依据类层面的出现率统计每个属性的稀有度，计算 BCE 用的 pos_weight。
    attr_targets: [num_classes, num_attrs] 的 0/1 张量
    """
    freq = attr_targets.float().mean(dim=0)            # 每列出现率
    eps = 1e-6
    pos_weight = ((1.0 - freq) / (freq + eps)).clamp_(min=0.1, max=max_weight)
    return pos_weight


def build_attr_matrix(kg_dir: str, classes: list, max_up_depth: int = 3, device: str = "cpu"):
    """
    从 kg_dir 中读取 nodes.csv / edges.csv，直接构造“祖先广播后的”属性监督矩阵。

    参数
    ----
    kg_dir       : 包含 nodes.csv / edges.csv 的目录
    classes      : CIFAR-100 的类别名列表（顺序需与数据集一致）
    max_up_depth : 祖先广播的向上深度（默认 3：父、祖父、曾祖父）
    device       : 返回的张量所在设备

    返回
    ----
    attr_matrix: torch.FloatTensor [num_classes, num_attrs]，0/1，多标签（含祖先广播）
    attr_names : list[str]，长度为 num_attrs，与 attr_matrix 的列顺序一一对应
    """
    kg_dir = Path(kg_dir)

    # 1) 读取 KG
    nodes_df = pd.read_csv(kg_dir / "nodes.csv")
    edges_df = pd.read_csv(kg_dir / "edges.csv")

    # 2) 规范化 & 基本索引
    nodes_df["id"] = nodes_df["id"].astype(int)
    nodes_df["name"] = nodes_df["name"].astype(str)
    nodes_df["type"] = nodes_df["type"].astype(str).str.strip().str.lower()

    edges_df["src"] = edges_df["src"].astype(int)
    edges_df["dst"] = edges_df["dst"].astype(int)
    edges_df["rel"] = edges_df["rel"].astype(str).str.strip().str.lower()

    id2name = dict(zip(nodes_df["id"], nodes_df["name"]))
    name2id = {v: k for k, v in id2name.items()}

    # 3) 确定“属性空间”（列空间）= KG 中所有 attribute 节点
    attr_ids = nodes_df.loc[nodes_df["type"] == "attribute", "id"].tolist()
    attr_names = [id2name[i] for i in attr_ids]
    attr_id2col = {aid: j for j, aid in enumerate(attr_ids)}
    num_attrs = len(attr_ids)

    # 4) 建图：is_a 的父指针 + has_attribute 出边
    parents = defaultdict(list)   # child -> [parent...]
    has_attr = defaultdict(list)  # node  -> [attribute_id...]

    for _, r in edges_df.iterrows():
        s, rel, d = int(r["src"]), r["rel"], int(r["dst"])
        if rel == "is_a":
            parents[s].append(d)
        elif rel == "has_attribute":
            # 只接受指向“attribute 节点”的边（避免脏数据）
            if d in attr_id2col:
                has_attr[s].append(d)

    # 5) 构造 [num_classes, num_attrs]，同时做“祖先属性广播”
    mat = np.zeros((len(classes), num_attrs), dtype=np.float32)
    for ci, cname in enumerate(classes):
        if cname not in name2id:
            # CIFAR 类名要和 KG class 节点 name 一致（下划线/大小写/复数都要一致）
            continue
        cid = name2id[cname]
        rel_nodes = {cid} | _ancestors_bfs(cid, parents, max_up_depth)  # 本类 + 祖先
        pos_attrs = set()
        for n in rel_nodes:
            for a in has_attr.get(n, []):
                pos_attrs.add(a)
        for a in pos_attrs:
            col = attr_id2col.get(a)
            if col is not None:
                mat[ci, col] = 1.0

    attr_matrix = torch.from_numpy(mat).to(device)
    return attr_matrix, attr_names
