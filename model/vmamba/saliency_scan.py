"""Algorithm 1 and Eq. (4): bijective shape-guided token permutations."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def shape_guided_paths(prior, size=None):
    if prior.ndim == 3:
        prior = prior.unsqueeze(1)
    if prior.shape[1] != 1:
        raise ValueError("SGS expects a single foreground probability channel")
    if size is not None and prior.shape[-2:] != tuple(size):
        prior = F.interpolate(prior.float(), size=size, mode="bilinear", align_corners=False)
    masks = (prior[:, 0] >= 0.5).cpu().numpy()
    paths = []
    for mask in masks:
        height, width = mask.shape
        foreground, previous = [], None
        for row in range(height):
            columns = np.flatnonzero(mask[row])
            if not len(columns):
                continue
            if previous is not None and abs(int(columns[0]) - previous) > abs(int(columns[-1]) - previous):
                columns = columns[::-1]
            foreground.extend((row * width + columns).tolist())
            previous = int(columns[-1])
        fg = np.asarray(foreground, dtype=np.int64)
        bg = np.flatnonzero(~mask.reshape(-1))
        p1, p2 = np.concatenate((fg, bg)), np.concatenate((bg, fg))
        paths.append(np.stack((p1, p2, p2[::-1], p1[::-1])))
    return torch.as_tensor(np.stack(paths), dtype=torch.long, device=prior.device)


def _indices(prior, batch, height, width, device, path):
    if prior is None:
        return torch.arange(height * width, device=device).expand(batch, -1)
    if prior.dtype == torch.long and prior.ndim == 3 and prior.shape[1:] == (4, height * width):
        return prior[:, path]
    return shape_guided_paths(prior, (height, width))[:, path]


def scan(x, prior, path):
    b, c, h, w = x.shape
    indices = _indices(prior, b, h, w, x.device, path)
    return x.flatten(2).gather(2, indices[:, None].expand(-1, c, -1)).unsqueeze(1)


def merge(ys, prior, path):
    if ys.ndim == 5:
        b, _, c, h, w = ys.shape
        ys = ys.flatten(3)
    else:
        b, _, c, length = ys.shape
        if prior is None or prior.shape[-2] * prior.shape[-1] != length:
            raise ValueError("Pass a spatial prior or a 5-D sequence to recover H and W")
        h, w = prior.shape[-2:]
    indices = _indices(prior, b, h, w, ys.device, path)
    # scatter is the inverse permutation; gather with the same order is not.
    return torch.zeros_like(ys[:, 0]).scatter(2, indices[:, None].expand(-1, c, -1), ys[:, 0]).reshape(b, c, h, w)


class CrossScan_Regular:
    apply = staticmethod(lambda x, prior=None: scan(x, prior, 0))


class CrossScan_Saliency1:
    apply = staticmethod(lambda x, prior=None: scan(x, prior, 1))


class CrossScan_Saliency2:
    apply = staticmethod(lambda x, prior=None: scan(x, prior, 2))


class CrossScan_Saliency3:
    apply = staticmethod(lambda x, prior=None: scan(x, prior, 3))


class CrossMerge_Regular:
    apply = staticmethod(lambda x, prior=None: merge(x, prior, 0))


class CrossMerge_Saliency1:
    apply = staticmethod(lambda x, prior=None: merge(x, prior, 1))


class CrossMerge_Saliency2:
    apply = staticmethod(lambda x, prior=None: merge(x, prior, 2))


class CrossMerge_Saliency3:
    apply = staticmethod(lambda x, prior=None: merge(x, prior, 3))
