from __future__ import annotations
import torch
from torch import Tensor
from torch import nn
from typing import Any
from model.encoder import Encoder
from model.STDecoder import Decoder


class STMamba(nn.Module):
    def __init__(
        self,
        encoder_name: str = "tiny_0230s",
        in_channels: int = 3,
        num_classes: int = 1,
        decoder_depths: tuple = (2, 2, 2, 2),
        drop_path_rate: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super(STMamba, self).__init__()
        self.encoder = Encoder(name=encoder_name, in_channels=in_channels, **kwargs)
        self.decoder = Decoder(
            dims=self.encoder.dims[::-1],  # Reverse the dims for decoder
            num_classes=num_classes,
            depths=decoder_depths,
            drop_path_rate=drop_path_rate,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = features[::-1]  
        return self.decoder(features)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        """冻结编码器参数"""
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        """解冻编码器参数"""
        self.encoder.unfreeze_params()