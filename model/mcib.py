"""Multi-Scale Context Interaction Bridge: pre-mix, one H/16 SS2D, redistribute.

Inspired by SegMAN's multi-scale context modeling. This implementation aligns
native encoder stages and injects shared context back into all three stages.
"""
import torch
from torch import nn
import torch.nn.functional as F
from model.vmamba.vmamba import LayerNorm2d, SS2D


class MultiScaleContextInteractionBridge(nn.Module):
    def __init__(self, encoder_dims=(96, 192, 384, 768), dim=96, residual_init=1e-3):
        super().__init__()
        if len(encoder_dims) != 4 or dim < 1:
            raise ValueError('MCIB expects four encoder stages and a positive context width')
        self.encoder_dims = tuple(encoder_dims)
        self.align_f2 = nn.Conv2d(encoder_dims[1] * 4, dim, 1)
        self.align_f3 = nn.Conv2d(encoder_dims[2], dim, 1)
        self.align_f4 = nn.Conv2d(encoder_dims[3], dim, 1)
        self.cross_scale_mix = nn.Conv2d(3 * dim, dim, 1)
        self.norm = LayerNorm2d(dim)
        self.context = SS2D(d_model=dim, d_state=1, ssm_ratio=1.0, d_conv=3,
                            conv_bias=False, forward_type='v05_noz', channel_first=True)
        self.inject = nn.ModuleList(nn.Conv2d(dim, channels, 1) for channels in encoder_dims[1:])
        self.residual_scale = nn.Parameter(torch.full((3,), float(residual_init)))

    def forward(self, features):
        if len(features) != 4:
            raise ValueError('MCIB expects [F1,F2,F3,F4] in shallow-to-deep order')
        f1, f2, f3, f4 = features
        # Standard 224/256 inputs need no padding. Support odd feature grids too.
        padded_f2 = F.pad(f2, (0, f2.shape[-1] % 2, 0, f2.shape[-2] % 2), mode='replicate')
        z2 = self.align_f2(F.pixel_unshuffle(padded_f2, 2))
        if z2.shape[-2:] != f3.shape[-2:]:
            z2 = F.interpolate(z2, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        z3 = self.align_f3(f3)
        z4 = self.align_f4(F.interpolate(f4, size=f3.shape[-2:], mode='bilinear', align_corners=False))
        mixed = self.cross_scale_mix(torch.cat((z2, z3, z4), dim=1))
        context = mixed + self.context(self.norm(mixed))
        aligned = (F.interpolate(context, size=f2.shape[-2:], mode='bilinear', align_corners=False),
                   context, F.adaptive_avg_pool2d(context, output_size=f4.shape[-2:]))
        enhanced = [feature + self.residual_scale[i] * projection(value)
                    for i, (feature, projection, value) in enumerate(zip((f2, f3, f4), self.inject, aligned))]
        return [f1, *enhanced]
