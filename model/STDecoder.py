from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn as nn
from einops import rearrange
from model.vmamba.vmamba import LayerNorm2d, Linear2d
from model.vmamba.SCVSSBlock import VSSBlockGroup
from typing import Sequence, Type, Optional

class MSConv(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)) -> None:
        super(MSConv, self).__init__()
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
            for kernel_size in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + sum([conv(x) for conv in self.dw_convs])

class MS_MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        channels_first: bool = False,
        kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super(MS_MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.multiscale_conv = MSConv(hidden_features, kernel_sizes=kernel_sizes)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.multiscale_conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MSVSS(nn.Module):
    def __init__(
        self,
        dim: int,  
        depth: int,
        drop_path: Sequence[float] | float = 0.0,
        use_checkpoint: bool = False,
        norm_layer: Type[nn.Module] = LayerNorm2d,
        channel_first: bool = True,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer: Type[nn.Module] = nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v05_noz",
        mlp_ratio: float = 4.0,
        mlp_act_layer: Type[nn.Module] = nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
    ) -> None:
        super().__init__()  
        block1 = VSSBlockGroup(
            hidden_dim=dim,  # 保持特征维度不变
            drop_path=drop_path[0] if isinstance(drop_path, Sequence) else drop_path,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=ssm_d_state,  # 状态维度
            ssm_ratio=ssm_ratio,      # SSM隐藏维度比例
            ssm_dt_rank=ssm_dt_rank,  # SSM时间步长rank
            ssm_act_layer=ssm_act_layer,  # SSM激活函数
            ssm_conv=ssm_conv,        # SSM卷积核大小
            ssm_conv_bias=ssm_conv_bias,  # SSM卷积偏置
            ssm_drop_rate=ssm_drop_rate,  # SSM dropout率
            ssm_init=ssm_init,        # SSM初始化方式
            forward_type=forward_type, # 前向传播类型
            mlp_ratio=mlp_ratio,      # MLP隐藏层比例
            mlp_act_layer=mlp_act_layer,  # MLP激活函数
            mlp_drop_rate=mlp_drop_rate,  # MLP dropout率
            gmlp=gmlp,                # 是否使用gMLP
            use_checkpoint=use_checkpoint,  # 是否使用梯度检查点
            customized_mlp=MS_MLP     # 使用多尺度MLP
        )
        
        # 第二个VSSBlock（如果depth > 1）
        block2 = VSSBlockGroup(
            hidden_dim=dim,  # 特征维度保持不变
            drop_path=drop_path[1] if isinstance(drop_path, Sequence) else drop_path,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            gmlp=gmlp,
            use_checkpoint=use_checkpoint,
            customized_mlp=MS_MLP
        ) if depth > 1 else None
        
        # 第三个VSSBlock（如果depth > 2）
        block3 = VSSBlockGroup(
            hidden_dim=dim,  # 特征维度保持不变
            drop_path=drop_path[2] if isinstance(drop_path, Sequence) else drop_path,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            gmlp=gmlp,
            use_checkpoint=use_checkpoint,
            customized_mlp=MS_MLP
        ) if depth > 2 else None
        
        block4 = VSSBlockGroup(
            hidden_dim=dim,  # 特征维度保持不变
            drop_path=drop_path[3] if isinstance(drop_path, Sequence) else drop_path,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            gmlp=gmlp,
            use_checkpoint=use_checkpoint,
            customized_mlp=MS_MLP
        ) if depth > 3 else None
        
        # 构建block序列，只包含非None的block
        blocks = [block1]
        if block2 is not None: blocks.append(block2)
        if block3 is not None: blocks.append(block3)
        if block4 is not None: blocks.append(block4)
        
        self.blocks = nn.ModuleList(blocks)  # 在super().__init__()之后设置属性
        
    def forward(self, x: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, saliency_mask)
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

class FLKPE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dim_scale: int = 4,
        norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super(FLKPE, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 16, dim * 16, kernel_size=3, padding=1, groups=dim * 16, bias=True)
        )

        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
        self.out = nn.Conv2d(self.output_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return self.out(x)

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = self.norm(x)
        return x

class SaliencyPredictor(nn.Module):
    def __init__(
        self,
        in_channels=[96, 192, 384, 768],  # 从浅到深的通道数
        num_classes=1,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        self.proj = nn.Conv2d(in_channels[0], 64, kernel_size=1)
        
        # 只保留第一阶段的预测器
        self.predictor = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        # 只使用最深层特征 (features[0])
        x = self.proj(features[0])
        mask = self.predictor(x)
        
        return mask

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_path: Sequence[float] | float,
    ) -> None:
        super(UpBlock, self).__init__()
        self.up = LKPE(in_channels)
        self.concat_layer = Linear2d(2 * out_channels, out_channels)
        self.vss_layer = MSVSS(dim=out_channels, depth=depth, drop_path=drop_path)

    def forward(self, input: torch.Tensor, skip: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.up(input)
        out = torch.cat(tensors=(out, skip), dim=1)
        out = self.concat_layer(out)
        out = self.vss_layer(out, saliency_mask)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int], 
        num_classes: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        drop_path_rate: float = 0.2,
    ) -> None:
        super(Decoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (len(dims) - 1) * 2)]
        
        # 添加显著性预测模块
        self.saliency_predictor = SaliencyPredictor(
            in_channels=dims,
            num_classes=num_classes,
            norm_layer=nn.LayerNorm
        )
        
        # 展开循环，创建上采样层
        # 第一层: 768 -> 384
        self.up_layer1 = UpBlock(
            in_channels=dims[0],  # 768 (7x7)
            out_channels=dims[1],  # 384 (14x14)
            depth=depths[1],
            drop_path=dpr[sum(depths[: 0]): sum(depths[: 1])]
        )
        
        # 第二层: 384 -> 192
        self.up_layer2 = UpBlock(
            in_channels=dims[1],  # 384 (14x14)
            out_channels=dims[2],  # 192 (28x28)
            depth=depths[2],
            drop_path=dpr[sum(depths[: 1]): sum(depths[: 2])]
        )
        
        # 第三层: 192 -> 96
        self.up_layer3 = UpBlock(
            in_channels=dims[2],  # 192 (28x28)
            out_channels=dims[3],  # 96 (56x56)
            depth=depths[3],
            drop_path=dpr[sum(depths[: 2]): sum(depths[: 3])]
        )
        
        # 最后的输出层
        self.out_layers = nn.Sequential(FLKPE(dims[-1], num_classes))
    
    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        saliency_mask = self.saliency_predictor(features)
        out = features[0]

        out1 = self.up_layer1(out, features[1], saliency_mask)  

        out2 = self.up_layer2(out1, features[2], out1)  

        out3 = self.up_layer3(out2, features[3], out2) 
        
        out = self.out_layers[0](out3) 
        
        return out
