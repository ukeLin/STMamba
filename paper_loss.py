"""Final-output-led deep supervision of S0--S4 on raw logits."""
import torch
from torch import nn
import torch.nn.functional as F


class DeepSupervisionLoss(nn.Module):
    def __init__(self, aux_weight=0.2):
        super().__init__()
        if aux_weight < 0:
            raise ValueError('aux_weight must be nonnegative')
        self.aux_weight = float(aux_weight)

    def single(self, logits, target):
        if target.ndim == 3:
            target = target.unsqueeze(1)
        # Upsample predictions to preserve small structures in the target.
        logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
        if logits.shape[1] == 1:
            target = target.float()
            ce = F.binary_cross_entropy_with_logits(logits, target)
            probability = logits.sigmoid()
        else:
            labels = target[:, 0].long()
            ce = F.cross_entropy(logits, labels)
            target = F.one_hot(labels, logits.shape[1]).permute(0, 3, 1, 2).float()
            probability = logits.softmax(1)
        axes = (0, 2, 3)
        dice = 1 - ((2 * (probability * target).sum(axes) + 1e-5) /
                    (probability.square().sum(axes) + target.square().sum(axes) + 1e-5)).mean()
        return 0.4 * ce + 0.6 * dice

    def forward(self, predictions, target):
        if isinstance(predictions, (tuple, list)):
            if len(predictions) != 5:
                raise ValueError('Deep supervision requires S0--S4')
            losses = torch.stack([self.single(p, target) for p in predictions])
            return (losses[-1] + self.aux_weight * losses[:-1].sum()) / (1 + 4 * self.aux_weight)
        return self.single(predictions, target)
