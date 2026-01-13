import torch
import torch.nn as nn
from collections import OrderedDict

from .utils import (
    MLP, DropPath, 
    adj_mx_from_skeleton, 
    adj_mx_from_skeleton_temporal, 
    h36m_skeleton_parents,
    trunc_normal_
)
from .sogc_stream import KPAttentionV2
from .msote_stream import Attention, TPAttentionPlus
from .fusion import GatedFusion, ConfidenceGate

class ADSGBlock(nn.Module):
    """
    ADSG Block - Adaptive Dual-Stream Gated Block
    Combines Spatial (SOGC) and Temporal (MSOTE) streams with Gated Fusion.
    """
    def __init__(self, adj, adj_temporal, num_frame, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, st_mode='stage_st',
                 fusion='gate', relative_pos=False):
        super().__init__()
        self.st_mode = st_mode
        self.fusion_type = fusion
        
        # Norm layers
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_b = norm_layer(dim)
        self.norm2 = norm_layer(dim) # For MLP
        
        if st_mode == 'stage_st':
            # Spatial Stream: SOGC (KPAttentionV2)
            self.kp_attn = KPAttentionV2(adj, dim, num_heads, drop_path, drop, norm_layer, qkv_bias, qk_scale,
                                         attn_drop, proj_drop=drop, comb=False, vis=False)
        elif st_mode == 'stage_ts':
            # Temporal Stream: MSOTE (Dual-path)
            if fusion == 'gate':
                self.fusion = GatedFusion(dim)
            else:
                self.fusion = nn.Linear(dim * 2, dim)
                nn.init.xavier_uniform_(self.fusion.weight)
                nn.init.zeros_(self.fusion.bias)

            # Path A: Standard Attention
            self.temp_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, proj_drop=drop, st_mode='stage_ts')
            
            # Path B: TPA Plus
            self.tp_attn = TPAttentionPlus(adj_temporal, num_frame, dim, num_heads, drop_path,
                                           drop, norm_layer, qkv_bias, qk_scale, attn_drop, proj_drop=drop,
                                           comb=False, vis=False, relative_pos=relative_pos)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seqlen=1):
        if self.st_mode == 'stage_st':
            # Spatial processing
            x = x + self.drop_path(self.kp_attn(self.norm1(x)))
        elif self.st_mode == 'stage_ts':
            # Temporal dual-path processing
            # Path A
            x_a = self.temp_attn(self.norm1_a(x), seqlen)
            
            # Path B: Reshape for TPA
            BF, J, C = x.shape
            B = BF // seqlen
            F = seqlen
            
            x_reshaped = x.reshape(B, F, J, C).permute(0, 2, 1, 3).reshape(B * J, F, C)
            x_b_reshaped = self.tp_attn(self.norm1_b(x_reshaped))
            x_b = x_b_reshaped.reshape(B, J, F, C).permute(0, 2, 1, 3).reshape(B * F, J, C)
            
            # Fusion
            if self.fusion_type == 'gate':
                x = x + self.drop_path(self.fusion(x_a, x_b))
            else:
                x_fused = torch.cat([x_a, x_b], dim=-1)
                x = x + self.drop_path(self.fusion(x_fused))
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ADSGFormer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=4,
                 num_joints=17, maxlen=243,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, att_fuse=True,
                 use_ktp=True, ts_fusion='gate', relative_pos=False, ms_temporal=True, use_conf_gate=True,
                 dataset_name="h36m", spatial_optimization=False): # spatial_optimization included for compatibility
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.num_joints = num_joints
        self.maxlen = maxlen
        self.use_conf_gate = use_conf_gate
        self.att_fuse = att_fuse

        # Helper/Pre-processing
        self.conf_gate = ConfidenceGate() if use_conf_gate else nn.Identity()

        # Graphs
        self.adj = adj_mx_from_skeleton(h36m_skeleton_parents)
        
        if ms_temporal:
             # Build multi-scale temporal graph
             import numpy as np
             import scipy.sparse as sp
             strides = (1, 2, 4)
             num_frame = maxlen
             edges = []
             for s in strides:
                 for i in range(num_frame - s):
                     edges.append([i, i + s])
             edges = np.array(edges, dtype=np.int32)
             data = np.ones(edges.shape[0])
             i = edges[:,0]; j = edges[:,1]
             adj = sp.coo_matrix((data, (i, j)), shape=(num_frame, num_frame), dtype=np.float32)
             
             # Normalized logic
             adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
             rowsum = np.array(adj.sum(1))
             r_inv = np.power(rowsum, -1).flatten()
             r_inv[np.isinf(r_inv)] = 0.
             r_mat_inv = sp.diags(r_inv)
             adj = r_mat_inv.dot(adj)
             adj = torch.tensor(adj.todense(), dtype=torch.float)
             adj = adj * (1 - torch.eye(adj.shape[0])) + torch.eye(adj.shape[0])
             self.adj_t = adj
        else:
             self.adj_t = adj_mx_from_skeleton_temporal(maxlen)

        # Embeddings
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        # Stochastic Depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Building Blocks
        self.blocks_st = nn.ModuleList([
            ADSGBlock(self.adj, self.adj_t, maxlen, dim=dim_feat, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                      norm_layer=norm_layer, st_mode="stage_st", fusion=ts_fusion,
                      relative_pos=relative_pos)
            for i in range(depth)])
            
        self.blocks_ts = nn.ModuleList([
            ADSGBlock(self.adj, self.adj_t, maxlen, dim=dim_feat, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                      norm_layer=norm_layer, st_mode="stage_ts", fusion=ts_fusion,
                      relative_pos=relative_pos)
            for i in range(depth)])

        self.norm = norm_layer(dim_feat)

        # Attention Fusion for the two streams (Inter-Block Fusion)
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat*2, 2) for i in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)

        # Heads
        if dim_rep:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim_feat, dim_rep)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
            
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_rep=False):
        # Confidence Gating
        if self.use_conf_gate:
            x = self.conf_gate(x)

        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        BF = x.shape[0]
        
        x = self.joints_embed(x)
        x = x + self.pos_embed
        _, J, C = x.shape
        
        x = x.reshape(-1, F, J, C) + self.temp_embed[:,:F,:,:]
        x = x.reshape(BF, J, C)
        x = self.pos_drop(x)

        alphas = []
        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, F)
            x_ts = blk_ts(x, F)
            
            if self.att_fuse:
                att = self.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                alpha = att(alpha)
                alpha = alpha.softmax(dim=-1)
                alphas.append(alpha)
                x = x_st * alpha[:,:,0:1] + x_ts * alpha[:,:,1:2]
            else:
                x = (x_st + x_ts) * 0.5

        x = self.norm(x)
        x = x.reshape(B, F, J, -1)
        x = self.pre_logits(x)

        if return_rep:
            return x

        x = self.head(x)
        return x
