from __future__ import annotations
import torch
import torch.nn as nn
from model.encoder import Encoder
from model.STDecoder import Decoder
from typing import Any

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
        features = features[::-1]  # Reverse features for decoder
        return self.decoder(features)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        """冻结编码器参数"""
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        """解冻编码器参数"""
        self.encoder.unfreeze_params()

        

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = STMamba(encoder_name="tiny_0230s", in_channels=3, num_classes=1)
    model = model.to(device)
    
    # Print model structure
    print(f"Model structure:\n{model}")
    
    # Generate a random input tensor
    batch_size = 2
    input_channels = 3
    input_height = 224
    input_width = 224
    x = torch.randn(batch_size, input_channels, input_height, input_width).to(device)
    
    # Print input shape
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        # Get encoder features
        encoder_features = model.encoder(x)
        print("\nEncoder feature shapes:")
        for i, feat in enumerate(encoder_features):
            print(f"  Feature {i}: {feat.shape}")
        
        # Reverse features for decoder
        decoder_input = encoder_features[::-1]
        
        # Get output
        output = model.decoder(decoder_input)
        print(f"\nOutput shape: {output.shape}") 