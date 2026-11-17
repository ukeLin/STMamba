"""Shape-conditioned complementary tree fusion (SC-CTF).

All path features are in their original spatial order before entering this tree.
Entropy is prediction uncertainty, not a calibrated probability of correctness.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def _foreground(probabilities, mode):
    if mode not in ('nonbackground', 'max'):
        raise ValueError('Unsupported foreground mode')
    if probabilities.shape[1] == 1:
        return probabilities
    if mode == 'max':
        return probabilities[:, 1:].amax(1, keepdim=True)
    return 1 - probabilities[:, :1]


def foreground_probability(logits, mode='nonbackground'):
    probabilities = logits.float().sigmoid() if logits.shape[1] == 1 else logits.float().softmax(1)
    return _foreground(probabilities, mode)


def prediction_condition(logits, size, detach=False, foreground_mode='nonbackground'):
    """Q=[foreground, full-class entropy]; multiclass foreground is 1-P(background)."""
    if logits is None or logits.ndim != 4:
        raise ValueError('SC-CTF requires full previous-stage prediction logits (B,K,H,W)')
    logits = logits.detach() if detach else logits
    probabilities = logits.float().sigmoid() if logits.shape[1] == 1 else logits.float().softmax(1)
    probabilities = F.interpolate(probabilities, size=size, mode='bilinear', align_corners=False)
    foreground = _foreground(probabilities, foreground_mode)
    if logits.shape[1] == 1:
        probabilities = torch.cat((1 - probabilities, probabilities), 1)
    entropy = -(probabilities * probabilities.clamp_min(1e-7).log()).sum(1, keepdim=True)
    entropy = (entropy / math.log(probabilities.shape[1])).clamp(0, 1)
    return foreground, torch.cat((foreground, entropy), 1)


class PairGate(nn.Module):
    """Two spatial weights per pixel, normalized over the two branches."""
    def __init__(self, channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3 * channels + 2, max(channels // 2, 8), 1),
            nn.SiLU(),
            nn.Conv2d(max(channels // 2, 8), 2, 1),
        )

    def weights(self, left, right, condition):
        # Keep the continuous condition in the autograd graph, including under AMP.
        condition = condition.to(dtype=left.dtype)
        evidence = torch.cat((left, right, (left - right).abs(), condition), 1)
        return self.network(evidence).float().softmax(1).to(left.dtype)

    def forward(self, left, right, condition):
        weights = self.weights(left, right, condition)
        return weights[:, :1] * left + weights[:, 1:] * right, weights


class ShapeConditionedCTF(nn.Module):
    PAIRINGS = {'14_23': ((0, 3), (1, 2)), '12_34': ((0, 1), (2, 3)),
                '13_24': ((0, 2), (1, 3))}

    def __init__(self, in_channels, path_channels, pairing='14_23',
                 condition_mode='real', mode='scctf', residual_enabled=True):
        super().__init__()
        if pairing not in self.PAIRINGS or condition_mode not in ('real', 'none', 'shuffle'):
            raise ValueError('Unsupported pairing or condition mode')
        if mode not in ('scctf', 'balanced', 'direct'):
            raise ValueError('Unsupported fusion mode')
        self.pairing, self.condition_mode, self.mode = pairing, condition_mode, mode
        self.residual_enabled = residual_enabled
        if mode != 'direct':
            self.pair_gate_14 = PairGate(path_channels) if mode == 'scctf' else None
            self.pair_gate_23 = PairGate(path_channels) if mode == 'scctf' else None
            self.cross_pair_gate = PairGate(path_channels) if mode == 'scctf' else None
        self.out_proj = nn.Conv2d((4 if mode == 'direct' else 2) * path_channels, in_channels, 1)
        self.residual = nn.Identity()
        if residual_enabled:
            self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, g1, g2, g3, g4, condition, return_details=False):
        groups = (g1, g2, g3, g4)
        if self.condition_mode == 'none':
            condition = torch.zeros_like(condition)
        elif self.condition_mode == 'shuffle':
            # A deterministic mismatch that never touches SGS permutations.
            condition = condition.roll(1, 0) if x.shape[0] > 1 else condition.flatten(2).roll(
                max(condition.shape[-1] * condition.shape[-2] // 2, 1), -1).reshape_as(condition)
        details = {}
        if self.mode == 'direct':
            fused = self.out_proj(torch.cat(groups, 1))
        else:
            (a, b), (c, d) = self.PAIRINGS[self.pairing]
            if self.mode == 'scctf':
                f14, w14 = self.pair_gate_14(groups[a], groups[b], condition)
                f23, w23 = self.pair_gate_23(groups[c], groups[d], condition)
                beta = self.cross_pair_gate.weights(f14, f23, condition)
                details = {'pair_14': w14, 'pair_23': w23, 'cross_pair': beta}
            else:
                f14, f23 = (groups[a] + groups[b]) / 2, (groups[c] + groups[d]) / 2
                beta = x.new_full((x.shape[0], 2, *x.shape[-2:]), .5)
            fused = self.out_proj(torch.cat((beta[:, :1] * f14, beta[:, 1:] * f23), 1))
        output = fused + self.res_scale * self.residual(x) if self.residual_enabled else fused
        return (output, details) if return_details else output
