import math
from functools import partial
from typing import Any, List, Type, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_

from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
from .csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F
from .csms6s import CrossScan, CrossMerge
from .csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction
from .csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex

from .csms6s import CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4
from .csms6s import CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4

from .saliency_scan import (
    CrossScan_Regular, CrossMerge_Regular,
    CrossScan_Saliency1, CrossMerge_Saliency1,
    CrossScan_Saliency2, CrossMerge_Saliency2,
    CrossScan_Saliency3, CrossMerge_Saliency3
)

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        channels_first: bool = False,
        **kwargs: Any,
    ):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x

class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError

class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


class SS2Dv2:
    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_cnorm:
            self.out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================

        self.forward_core = partial(
            self.forward_corev2, 
            force_fp32=False, 
            SelectiveScan=SelectiveScanOflex, 
            no_einsum=True,
            CrossScan=CrossScan_Regular,
            CrossMerge=CrossMerge_Regular
        )
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        # 修改这里，考虑到输入已经被分成4组
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (1, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (1, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (1, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=1, merge=True)  # (D, N)
            self.Ds = self.D_init(d_inner, copies=1, merge=True)  # (D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((d_inner)))
            self.A_logs = nn.Parameter(torch.randn((d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((1, d_inner, dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((1, d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((1, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((1, d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor = None,
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=SelectiveScanOflex,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
        # ==============================
        cascade2d=False,
        **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        L = H * W
        R = dt_projs_weight.shape[-1]  # dt_rank

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                x: torch.Tensor,
                proj_weight: torch.Tensor,
                proj_bias: torch.Tensor,
                dt_weight: torch.Tensor,
                dt_bias: torch.Tensor,  # (2*c)
                _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                _Ds: torch.Tensor,
                width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1), bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)  # (B, D, H, W)
            xs = xs.view(B, D, L)  # (B, D, L)

            if no_einsum:
                x_dbl = F.conv1d(xs, x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None))
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=1)
                dts = F.conv1d(dts.contiguous(), dt_projs_weight.view(-1, R, 1))
                # 为B和C添加group维度
                Bs = Bs.view(B, 1, N, L)  # (B, 1, N, L)
                Cs = Cs.view(B, 1, N, L)  # (B, 1, N, L)
            else:
                x_dbl = torch.einsum("b d l, c d -> b c l", xs, x_proj_weight.squeeze(0))
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=1)
                dts = torch.einsum("b r l, d r -> b d l", dts, dt_projs_weight.squeeze(0))
                # 为B和C添加group维度
                Bs = Bs.view(B, 1, N, L)  # (B, 1, N, L)
                Cs = Cs.view(B, 1, N, L)  # (B, 1, N, L)

            # 调整所有张量的形状以匹配selective_scan的要求
            # u: (batch_size, dim, seqlen)
            # delta: (batch_size, dim, seqlen)
            # A: (dim, dstate)
            # B: (batch_size, n_groups, dstate, seqlen)
            # C: (batch_size, n_groups, dstate, seqlen)
            # D: (dim,)
            # delta_bias: (dim,)
            dts = dts.contiguous()  # (B, D, L)
            As = -torch.exp(A_logs.to(torch.float))  # (D, N)
            Bs = Bs.contiguous()  # (B, 1, N, L)
            Cs = Cs.contiguous()  # (B, 1, N, L)
            Ds = Ds.to(torch.float)  # (D,)
            # 确保delta_bias的形状正确
            delta_bias = dt_projs_bias.squeeze(0).to(torch.float)  # (D,)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            # 确保ys的形状正确
            ys = ys.view(B, 1, D, L)  # (B, 1, D, L)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SS2D(nn.Module, mamba_init, SS2Dv2):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
  
        self.__initv2__(**kwargs)


class TreeFusionModule(nn.Module):
    """树型特征融合模块，实现渐进式特征聚合
    流程：直接特征融合 -> 应用注意力 -> 输出
    """
    def __init__(self, dim, channel_first=True):
        super().__init__()
        self.dim = dim
        self.channel_first = channel_first
        
        # 特征融合层 - 使用简单高效的融合模块
        if channel_first:
            # 第一次融合：特征1和特征2
            self.fusion1 = nn.Sequential(
                nn.Conv2d(dim*2, dim*2, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim*2),
                nn.SiLU()
            )
            
            # 第二次融合：(特征1+特征2)和特征3
            self.fusion2 = nn.Sequential(
                nn.Conv2d(dim*3, dim*3, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim*3),
                nn.SiLU()
            )
            
            # 第三次融合：(特征1+特征2+特征3)和特征4
            self.fusion3 = nn.Sequential(
                nn.Conv2d(dim*4, dim*4, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim*4),
                nn.SiLU()
            )
            

        else:
            # 第一次融合：特征1和特征2
            self.fusion1 = nn.Sequential(
                nn.Linear(dim*2, dim*2),
                nn.LayerNorm(dim*2),
                nn.SiLU()
            )
            
            # 第二次融合：(特征1+特征2)和特征3
            self.fusion2 = nn.Sequential(
                nn.Linear(dim*3, dim*3),
                nn.LayerNorm(dim*3),
                nn.SiLU()
            )
            
            # 第三次融合：(特征1+特征2+特征3)和特征4
            self.fusion3 = nn.Sequential(
                nn.Linear(dim*4, dim*4),
                nn.LayerNorm(dim*4),
                nn.SiLU()
            )

        
    def forward(self, x1, x2, x3, x4):
        """
        树型特征融合
        Args:
            x1, x2, x3, x4: 四个分支的特征，可以是(B, C, H, W)或(B, H, W, C)格式
        """
        if self.channel_first:
            # channel_first格式处理
            # 第一次融合：特征1和特征2
            f12 = torch.cat([x1, x2], dim=1)
            f12 = self.fusion1(f12)
            
            # 第二次融合：(特征1+特征2)和特征3
            f123 = torch.cat([f12, x3], dim=1)
            f123 = self.fusion2(f123)
        
            # 第三次融合：(特征1+特征2+特征3)和特征4
            f1234 = torch.cat([f123, x4], dim=1)
            f1234 = self.fusion3(f1234)
            
            # 应用通道注意力
            # attention = self.channel_attention(f1234)
            out = f1234
            
            # 最终归一化
            # out = self.norm(out)
            
        else:
            # channel_last格式处理 (B, H, W, C)
            # 第一次融合：特征1和特征2
            f12 = torch.cat([x1, x2], dim=-1)
            f12 = self.fusion1(f12)
        
            # 第二次融合：(特征1+特征2)和特征3
            f123 = torch.cat([f12, x3], dim=-1)
            f123 = self.fusion2(f123)
        
            # 第三次融合：(特征1+特征2+特征3)和特征4
            f1234 = torch.cat([f123, x4], dim=-1)
            f1234 = self.fusion3(f1234)
            
            # 应用通道注意力 - 对于channel_last格式
            # 先计算空间平均池化
            B, H, W, C = f1234.shape
            f1234_pool = f1234.mean(dim=[1, 2])  # (B, C)
            # attention = self.channel_attention(f1234_pool).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, C)
            out = f1234
        

        
        return out

class VSSBlockGroup(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        use_tree_fusion: bool = True,  # 添加树型融合开关
        **kwargs,
    ):
        super().__init__()
        self.channel_first = channel_first
        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.use_tree_fusion = use_tree_fusion  # 是否使用树型融合
        
        # Layer norm
        self.norm = norm_layer(hidden_dim)
        
        # Four SS2D groups
        group_dim = hidden_dim // 4
        self.mamba_g1 = SS2D(
            d_model=group_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.mamba_g2 = SS2D(
            d_model=group_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.mamba_g3 = SS2D(
            d_model=group_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        self.mamba_g4 = SS2D(
            d_model=group_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        
        # 添加树型融合模块
        if use_tree_fusion:
            self.tree_fusion = TreeFusionModule(group_dim, channel_first=True)

        self.drop_path = DropPath(drop_path)
        
        if mlp_ratio > 0:
            _MLP = Mlp if not gmlp else gMlp
            _MLP = kwargs.get("customized_mlp", None) or _MLP
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=channel_first,
            )
        else:
            self.mlp = None

    def _forward(self, x: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            
        identity = x
        x = self.norm(x)

        # Split channels into 4 groups
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=-1 if not self.channel_first else 1)

        # Apply different scan patterns to each group
        x_mamba1 = self.mamba_g1(x1, CrossScan=CrossScan_Regular, CrossMerge=CrossMerge_Regular, saliency_mask=saliency_mask)
        x_mamba2 = self.mamba_g2(x2, CrossScan=CrossScan_Saliency1, CrossMerge=CrossMerge_Saliency1, saliency_mask=saliency_mask)
        x_mamba3 = self.mamba_g3(x3, CrossScan=CrossScan_Saliency2, CrossMerge=CrossMerge_Saliency2, saliency_mask=saliency_mask)
        x_mamba4 = self.mamba_g4(x4, CrossScan=CrossScan_Saliency3, CrossMerge=CrossMerge_Saliency3, saliency_mask=saliency_mask)

        # 使用树型融合或简单拼接
        if self.use_tree_fusion:
            # 确保输入到tree_fusion的特征都是channel_first格式
            if not self.channel_first:
                x_mamba1 = x_mamba1.permute(0, 3, 1, 2)
                x_mamba2 = x_mamba2.permute(0, 3, 1, 2)
                x_mamba3 = x_mamba3.permute(0, 3, 1, 2)
                x_mamba4 = x_mamba4.permute(0, 3, 1, 2)
            
            x_mamba = self.tree_fusion(x_mamba1, x_mamba2, x_mamba3, x_mamba4)
            
            # 如果原始输入是channel_last格式，则转换回来
            if not self.channel_first:
                x_mamba = x_mamba.permute(0, 2, 3, 1)
        else:
            # 简单拼接
            x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=-1 if not self.channel_first else 1)

        # Skip connection
        x_mamba = x_mamba + identity
        x_mamba = self.drop_path(x_mamba)

        # MLP if enabled
        if self.mlp is not None:
            if self.post_norm:
                x_mamba = x_mamba + self.drop_path(self.norm2(self.mlp(x_mamba)))
            else:
                x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba)))

        return x_mamba

    def forward(self, x: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, saliency_mask)
        else:
            return self._forward(x, saliency_mask)

