"""Progressive SGS/SC-CTF decoder with ordinary bilinear + convolution upsampling."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vmamba.vmamba import LayerNorm2d, Linear2d
from model.vmamba.SCVSSBlock import VSSBlockGroup
from typing import Sequence, Type, Optional

class MSVSS(nn.Module):
    def __init__(self, dim, depth, drop_path=0., fusion_options=None, **kwargs):
        super().__init__()
        if depth < 1:
            raise ValueError("A decoder stage must contain at least one block")
        rates = list(drop_path) if isinstance(drop_path, Sequence) else [drop_path] * depth
        if len(rates) != depth:
            raise ValueError("Provide one drop-path rate per block")
        self.blocks = nn.ModuleList(VSSBlockGroup(
            hidden_dim=dim, drop_path=rates[i], fusion_options=fusion_options, **kwargs)
            for i in range(depth))

    def forward(self, x, prediction_logits):
        for block in self.blocks:
            x = block(x, prediction_logits)
        return x

class BilinearUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, size=None):
        x = F.interpolate(x, size=size, scale_factor=2 if size is None else None,
                          mode='bilinear', align_corners=False)
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_path: Sequence[float] | float,
        fusion_options=None,
    ) -> None:
        super(UpBlock, self).__init__()
        self.up = BilinearUpsample(in_channels, out_channels)
        self.concat_layer = Linear2d(2 * out_channels, out_channels)
        self.vss_layer = MSVSS(dim=out_channels, depth=depth, drop_path=drop_path, fusion_options=fusion_options)

    def forward(self, input: torch.Tensor, skip: torch.Tensor, prediction_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.up(input, size=skip.shape[-2:])
        out = torch.cat(tensors=(out, skip), dim=1)
        out = self.concat_layer(out)
        out = self.vss_layer(out, prediction_logits)
        return out

class Decoder(nn.Module):
    """Four progressive SC-CTF decoder scales with S0--S4 supervision."""
    def __init__(self, dims, num_classes, depths=(2, 2, 2, 2), drop_path_rate=0.2, fusion_options=None):
        super().__init__()
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("Expected four encoder scales and four decoder depths")
        rates = torch.linspace(drop_path_rate, 0, sum(depths)).tolist()
        self.initial_head = nn.Conv2d(dims[0], num_classes, 1)
        self.deep_stage = MSVSS(dims[0], depths[0], rates[:depths[0]], fusion_options=fusion_options)
        self.up_layers = nn.ModuleList()
        offset = depths[0]
        for i in range(1, 4):
            self.up_layers.append(UpBlock(dims[i-1], dims[i], depths[i], rates[offset:offset+depths[i]], fusion_options=fusion_options))
            offset += depths[i]
        self.stage_heads = nn.ModuleList(nn.Conv2d(d, num_classes, 1) for d in dims[:3])
        self.final_expand = nn.Sequential(BilinearUpsample(dims[-1], dims[-1] // 2),
                                          BilinearUpsample(dims[-1] // 2, dims[-1] // 4))
        self.final_head = nn.Conv2d(dims[-1] // 4, num_classes, 1)

    @staticmethod
    def foreground(logits):
        if logits.shape[1] == 1:
            return logits.sigmoid()
        return logits.softmax(dim=1)[:, 1:].amax(dim=1, keepdim=True)

    def forward(self, features, return_aux=False, output_size=None):
        out = features[0]
        predictions = [self.initial_head(out)]
        prior = predictions[-1]  # full logits: SGS foreground plus multiclass entropy
        out = self.deep_stage(out, prior)
        predictions.append(self.stage_heads[0](out))
        for i, up in enumerate(self.up_layers):
            prior = predictions[-1]  # full logits: SGS foreground plus multiclass entropy
            out = up(out, features[i+1], prior)
            if i < 2:
                predictions.append(self.stage_heads[i+1](out))
        final = self.final_head(self.final_expand(out))
        if output_size is not None and final.shape[-2:] != tuple(output_size):
            final = F.interpolate(final, size=output_size, mode='bilinear', align_corners=False)
        predictions.append(final)
        return tuple(predictions) if return_aux else predictions[-1]
