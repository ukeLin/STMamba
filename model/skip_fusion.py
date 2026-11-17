"""Local difference-gated residual interaction after ordinary upsampling."""
import torch
from torch import nn
import torch.nn.functional as F


class LocalDifferenceFusion(nn.Module):
    def __init__(self, channels, residual_init=1e-3):
        super().__init__()
        width = max(channels // 8, 16)

        def project():
            return nn.Sequential(nn.Conv2d(channels, width, 1, bias=False),
                                 nn.GroupNorm(1, width), nn.SiLU())

        self.deep_context = project()
        self.skip_context = project()
        self.gate = nn.Sequential(
            nn.Conv2d(4 * width, 4 * width, 3, padding=1, groups=4 * width),
            nn.SiLU(), nn.Conv2d(4 * width, 2, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.refine = nn.Sequential(nn.Conv2d(3 * width, channels, 1, bias=False),
                                    nn.GroupNorm(1, channels))
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), residual_init))

    def forward(self, deep, skip, base, return_details=False):
        d, s = self.deep_context(deep), self.skip_context(skip)
        detail = s - F.avg_pool2d(s, 3, stride=1, padding=1, count_include_pad=False)
        evidence = torch.cat((d, s, (d - s).abs(), detail.abs()), dim=1)
        weights = self.gate(evidence).float().softmax(1).to(d.dtype)
        wd, ws = weights[:, :1], weights[:, 1:]
        residual = self.refine(torch.cat((wd * d, ws * s, ws * detail), dim=1))
        result = base + self.residual_scale.to(base.dtype) * residual
        return (result, weights) if return_details else result
