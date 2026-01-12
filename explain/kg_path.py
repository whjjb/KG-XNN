# kgxnn/explain/kg_path.py
import pandas as pd
from collections import defaultdict, deque

class KGPathExtractor:
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, max_up_depth: int = 3):
        self.nodes = nodes.copy()
        self.edges = edges.copy()
        self.max_up_depth = max_up_depth

        self.id2name = dict(zip(self.nodes["id"], self.nodes["name"]))
        self.name2id = {v: k for k, v in self.id2name.items()}

        # 邻接索引
        self.parents = defaultdict(list)        # is_a 上行
        self.children = defaultdict(list)       # is_a 下行（目前用不到，但留着）
        self.attr_of = defaultdict(list)        # class/hypernym -> attribute ids
        for _, r in self.edges.iterrows():
            s, rel, d = int(r["src"]), r["rel"], int(r["dst"])
            if rel == "is_a":
                self.parents[s].append(d)
                self.children[d].append(s)
            elif rel == "has_attribute":
                self.attr_of[s].append(d)

    def upward_chain_ids(self, node_id, max_len=4):
        """返回从自身到祖先的一条链（用于显示层级）。"""
        path = [node_id]
        cur = node_id
        while len(path) < max_len and self.parents.get(cur):
            cur = self.parents[cur][0]  # 简单取第一个父类
            if cur == path[-1]:
                break
            path.append(cur)
        return path

    def ancestors(self, node_id, max_depth=None):
        """BFS 取多层祖先集合。"""
        if max_depth is None:
            max_depth = self.max_up_depth
        vis, q = set(), deque([(node_id, 0)])
        while q:
            u, d = q.popleft()
            if d == 0:
                pass  # 不把自己加入 ancestors（后面会单独 union）
            for p in self.parents.get(u, []):
                if p not in vis:
                    vis.add(p)
                    if d + 1 < max_depth:
                        q.append((p, d + 1))
        return vis  # 不含自己

    def upward_chain(self, node_id, max_len=4):
        """把 upward_chain_ids 转成名字列表用于展示。"""
        return [self.id2name[i] for i in self.upward_chain_ids(node_id, max_len=max_len)]

    def describe(self, pred_class_name: str, topk_node_ids, fallback_attr_topm: int = 5):
        # 类不在图中：直接提示
        if pred_class_name not in self.name2id:
            return f"{pred_class_name}（该类别未在知识图谱中注册，仅显示视觉解释）"

        cls_id = self.name2id[pred_class_name]

        # 1) 语义层级链（展示）
        chain = self.upward_chain(cls_id, max_len=4)
        segs = []
        if len(chain) > 1:
            segs.append("语义层级: " + " → ".join(chain))

        # 2) 构造"相关节点集合" = 本类 + 祖先 + 它们的属性
        anc = self.ancestors(cls_id, max_depth=self.max_up_depth)
        related = set([cls_id]) | set(anc)

        # 把本类与祖先的属性加进来
        related_attrs = set()
        for nid in list(related):
            for a in self.attr_of.get(nid, []):
                related_attrs.add(a)
        related_all = related | related_attrs

        # 新增：展示相关域节点（本组+祖先的属性名）
        related_attr_names = [self.id2name[a] for a in related_attrs]
        segs.append("相关域节点: " + (", ".join(related_attr_names) if related_attr_names else "无"))

        # 3) 从传入的 topk_node_ids 里筛出与之相关的节点（交集）
        topk_node_ids = [int(n) for n in topk_node_ids]
        explained_ids = [n for n in topk_node_ids if n in related_all]

        # 4) 展示注意力命中节点（topk与相关域的交集）
        if explained_ids:
            explained_names = [self.id2name[n] for n in explained_ids]
            segs.append("注意力命中节点: " + ", ".join(explained_names))
        else:
            segs.append("注意力命中节点: 无")

        # 使用换行连接各部分
        return "\n".join(segs)