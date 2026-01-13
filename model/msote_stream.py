import torch
import torch.nn as nn
from einops import rearrange
from .utils import LearnableGraphConv, DropPath, adj_mx_from_skeleton_temporal

class RelativeBias(nn.Module):
    def __init__(self, max_len, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        size = 2 * max_len - 1
        self.bias = nn.Parameter(torch.zeros(num_heads, size))

    def forward(self, seqlen):
        device = self.bias.device
        idx = torch.arange(seqlen, device=device)
        rel = idx[None, :] - idx[:, None]  # [T,T]
        rel = rel + (self.max_len - 1)
        rel = rel.clamp(0, 2 * self.max_len - 2)
        return self.bias[:, rel]  # [H,T,T]

class TPA(nn.Module):
    """Trajectory Prior Attention (Original)"""
    def __init__(self, adj_temporal, input_dim, output_dim, p_dropout=None):
        super(TPA, self).__init__()
        self.gconv = LearnableGraphConv(input_dim, output_dim, adj_temporal)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # x shape: [B, T, C]
        x = self.gconv(x)  # [B, T, C]
        
        # Reshape for BatchNorm1d: [B, T, C] -> [B*T, C]
        B, T, C = x.shape
        x = x.reshape(B * T, C)
        x = self.bn(x)  # [B*T, C]
        x = x.reshape(B, T, C)  # [B, T, C]
        
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        else:
            x = self.relu(x)
        return x

class StackedTPA(nn.Module):
    """Stacked TPA module"""
    def __init__(self, adj_temporal, input_dim, output_dim, hid_dim, p_dropout):
        super(StackedTPA, self).__init__()
        self.gconv1 = TPA(adj_temporal, input_dim, hid_dim, p_dropout)
        self.gconv2 = TPA(adj_temporal, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out

class TPAttention(nn.Module):
    """Trajectory Prior Attention Module - For Dual-Path Temporal Processing"""
    def __init__(self, adj_temporal, num_frame, dim, num_heads=8, drop_path=0, drop_rate=0, 
                 norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                 comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.tpa = StackedTPA(adj_temporal, dim, dim, dim, p_dropout=None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, vis=False):
        # Check if we need to create a new TPA for different temporal dimensions
        B, T, C = x.shape
        if T != self.Temporal_pos_embed.shape[1]:
            # Create new temporal adjacency matrix for current sequence length
            adj_temp = adj_mx_from_skeleton_temporal(T)
            # Ensure it's on the right device
            temp_tpa = StackedTPA(adj_temp, C, C, C, p_dropout=None)
            temp_tpa = temp_tpa.to(x.device)
            x = temp_tpa(x)
            
            # Use truncated or padded positional embedding
            if T <= self.Temporal_pos_embed.shape[1]:
                pos_embed = self.Temporal_pos_embed[:, :T, :]
            else:
                # Pad with zeros if needed
                pad_size = T - self.Temporal_pos_embed.shape[1]
                pos_embed = torch.cat([
                    self.Temporal_pos_embed,
                    torch.zeros(1, pad_size, C, device=x.device)
                ], dim=1)
            x = x + pos_embed
        else:
            x = self.tpa(x)
            x = x + self.Temporal_pos_embed
        
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

class TPAttentionPlus(TPAttention):
    def __init__(self, adj_temporal, num_frame, dim, num_heads=8, drop_path=0, drop_rate=0,
                 norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 comb=False, vis=False, relative_pos=False):
        super().__init__(adj_temporal, num_frame, dim, num_heads, drop_path, drop_rate,
                         norm_layer, qkv_bias, qk_scale, attn_drop, proj_drop, comb, vis)
        self.relative_pos = relative_pos
        if relative_pos:
            self.rel_bias = RelativeBias(num_frame, num_heads)

    def forward(self, x, vis=False):
        B, T, C = x.shape
        # Handle variable length via temporary TPA and positional embedding logic (same as parent)
        if T != self.Temporal_pos_embed.shape[1]:
            # Similar logic as parent but duplicated because super().forward is tricky to intercept
            adj_temp = adj_mx_from_skeleton_temporal(T)
            temp_tpa = type(self.tpa)(adj_temp, C, C, C, p_dropout=None).to(x.device)
            x = temp_tpa(x)
            if T <= self.Temporal_pos_embed.shape[1]:
                pos_embed = self.Temporal_pos_embed[:, :T, :]
            else:
                pad_size = T - self.Temporal_pos_embed.shape[1]
                pos_embed = torch.cat([
                    self.Temporal_pos_embed,
                    torch.zeros(1, pad_size, C, device=x.device)
                ], dim=1)
            x = x + pos_embed
        else:
            x = self.tpa(x)
            x = x + self.Temporal_pos_embed

        x = self.pos_drop(x)
        res = x.clone()
        x = self.norm1(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.comb is True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos:
            bias = self.rel_bias(N)  # [H, N, N]
            attn = attn + bias.unsqueeze(0)  # [B, H, N, N]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.comb is True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = res + self.drop_path(x)
        return x

class Attention(nn.Module):
    """Standard Attention Module - Path A for Temporal Processing"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., st_mode='vanilla'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.st_mode = st_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seqlen=1):
        if self.st_mode == 'stage_st':
            return self.forward_spatial(x)
        elif self.st_mode == 'stage_ts':
            return self.forward_temporal(x, seqlen)
        else:
            return self.forward_spatial(x)

    def forward_spatial(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_temporal(self, x, seqlen=8):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        qt = q.reshape(-1, seqlen, self.num_heads, N, C // self.num_heads).permute(0, 2, 3, 1, 4)
        kt = k.reshape(-1, seqlen, self.num_heads, N, C // self.num_heads).permute(0, 2, 3, 1, 4)
        vt = v.reshape(-1, seqlen, self.num_heads, N, C // self.num_heads).permute(0, 2, 3, 1, 4)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
