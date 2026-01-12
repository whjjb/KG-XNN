import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFuse(nn.Module):
    def __init__(self, v_dim=2048, k_dim=256, out_dim=256):
        super().__init__()
        self.q_proj = nn.Linear(v_dim,  k_dim)
        self.k_proj = nn.Linear(k_dim,  k_dim)
        self.v_proj = nn.Linear(k_dim,  out_dim)

    def forward(self, v_feats, kg_feats):
        # v_feats: [B, v_dim]  (视觉主干输出)
        # kg_feats: [N, k_dim] (KG 编码器输出)
        Q = self.q_proj(v_feats)              # [B, k]
        K = self.k_proj(kg_feats)             # [N, k]
        V = self.v_proj(kg_feats)             # [N, out]
        attn = torch.softmax(Q @ K.T / (K.size(1) ** 0.5), dim=-1)  # [B, N]
        z = attn @ V                          # [B, out]
        return z, attn