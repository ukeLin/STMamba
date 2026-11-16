"""Progressive decoder with four SC-CTF stages and two final 2x AE layers."""
from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange
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

class LKPE(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Type[nn.Module] = nn.LayerNorm):
        super(LKPE, self).__init__()
        self.dim = dim
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=True)
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return x

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
        self.up = LKPE(in_channels)
        self.concat_layer = Linear2d(2 * out_channels, out_channels)
        self.vss_layer = MSVSS(dim=out_channels, depth=depth, drop_path=drop_path, fusion_options=fusion_options)

    def forward(self, input: torch.Tensor, skip: torch.Tensor, prediction_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.up(input)
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
        # Two applications of Eq. (7)--(8) recover /4 -> /2 -> /1.
        self.final_expand = nn.Sequential(LKPE(dims[-1]), LKPE(dims[-1] // 2))
        self.final_head = nn.Conv2d(dims[-1] // 4, num_classes, 1)

    @staticmethod
    def foreground(logits):
        if logits.shape[1] == 1:
            return logits.sigmoid()
        return logits.softmax(dim=1)[:, 1:].amax(dim=1, keepdim=True)

    def forward(self, features, return_aux=False):
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
        predictions.append(self.final_head(self.final_expand(out)))
        return tuple(predictions) if return_aux else predictions[-1]
