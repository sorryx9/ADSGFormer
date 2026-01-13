import torch
import torch.nn as nn
from einops import rearrange
from .utils import LearnableGraphConv, DropPath

class LearnableGraphConvV2(LearnableGraphConv):
    def __init__(self, in_features, out_features, adj, bias=True):
        super().__init__(in_features, out_features, adj, bias)
        self.adj_2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj_2, 1e-6)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = self.adj.to(input.device)
        adj_2 = self.adj_2.to(input.device)

        # Combine first and second order adjacency
        adj_combined = (adj.T + adj) / 2 + (adj_2.T + adj_2) / 2
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        output = torch.matmul(adj_combined * E, self.M * h0) + torch.matmul(adj_combined * (1 - E), self.M * h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

class KPA(nn.Module):
    """Kinematics Prior Attention (Original)"""
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(KPA, self).__init__()
        self.gconv = LearnableGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # x shape: [B, J, C]
        x = self.gconv(x)  # [B, J, C]
        
        # Reshape for BatchNorm1d: [B, J, C] -> [B*J, C]
        B, J, C = x.shape
        x = x.reshape(B * J, C)
        x = self.bn(x)  # [B*J, C]
        x = x.reshape(B, J, C)  # [B, J, C]
        
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        else:
            x = self.relu(x)
        return x

class KPAV2(KPA):
    """Kinematics Prior Attention V2"""
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super().__init__(adj, input_dim, output_dim, p_dropout)
        self.gconv = LearnableGraphConvV2(input_dim, output_dim, adj)

class KPAttention(nn.Module):
    """Kinematics Prior Attention Module - Replacing Spatial Self-Attention"""
    def __init__(self, adj, dim, num_heads=8, drop_path=0, drop_rate=0, norm_layer=nn.LayerNorm, 
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kpa = KPA(adj, dim, dim, p_dropout=None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 17, dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, vis=False):
        x = self.kpa(x)
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        res = x.clone()
        x = self.norm1(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.comb == True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb == False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.comb == True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb == False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = res + self.drop_path(x)
        return x

class KPAttentionV2(KPAttention):
    def __init__(self, adj, dim, num_heads=8, drop_path=0, drop_rate=0, norm_layer=nn.LayerNorm,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__(adj, dim, num_heads, drop_path, drop_rate, norm_layer, qkv_bias, qk_scale, attn_drop, proj_drop, comb, vis)
        self.kpa = KPAV2(adj, dim, dim, p_dropout=None)
