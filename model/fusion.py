import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_a, x_b):
        cat = torch.cat([x_a, x_b], dim=-1)
        g = self.gate(cat)
        fused = g * x_b + (1 - g) * x_a
        return self.proj(torch.cat([fused, x_a], dim=-1))

class ConfidenceGate(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Optional confidence gating: if input has confidence channel, modulate the 2D coords
        if x.shape[-1] >= 3:
            conf = x[..., 2:3]
            x = x.clone()
            x[..., :2] = x[..., :2] * conf
        return x
