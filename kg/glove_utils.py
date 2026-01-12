# kg/glove_utils.py
import os
import re
import numpy as np
import torch

__all__ = ["load_glove_as_dict", "build_node_embedding_from_glove", "simple_tokenize"]

def simple_tokenize(name: str):
    """
    将节点名转为 token 列表：
    - 下划线/连字符/斜杠 -> 空格
    - 只保留字母数字空格
    - 小写
    """
    s = name.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"[^a-zA-Z0-9 ]+", " ", s)
    toks = [t for t in s.lower().split() if t]
    return toks

def load_glove_as_dict(glove_txt_path: str, dim: int):
    """
    读取 GloVe 文本文件，返回 {token: np.ndarray(dim,)} 字典。
    """
    if not os.path.isfile(glove_txt_path):
        raise FileNotFoundError(f"GloVe file not found: {glove_txt_path}")
    w2v = {}
    with open(glove_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < dim + 1:
                continue
            token = parts[0]
            try:
                vec = np.asarray(parts[1:1+dim], dtype=np.float32)
            except ValueError:
                continue
            if vec.shape[0] == dim:
                w2v[token] = vec
    return w2v

def build_node_embedding_from_glove(node_names, w2v: dict, dim: int, rng=None):
    """
    给定节点名列表和 GloVe 词典，返回 [N_nodes, dim] 的初始化矩阵（np.ndarray）。
    策略：每个节点名 -> tokenize -> 对命中的 token 词向量求均值；若全 OOV，则随机初始化。
    """
    if rng is None:
        rng = np.random.default_rng(123)
    N = len(node_names)
    mat = np.zeros((N, dim), dtype=np.float32)
    hit, total = 0, N
    for i, name in enumerate(node_names):
        toks = simple_tokenize(str(name))
        vecs = [w2v[t] for t in toks if t in w2v]
        if len(vecs) == 0:
            # OOV: 随机小幅噪声
            mat[i] = rng.normal(0.0, 0.02, size=(dim,)).astype(np.float32)
        else:
            hit += 1
            mat[i] = np.mean(np.stack(vecs, axis=0), axis=0)
    coverage = hit / max(1, total)
    return mat, coverage
