"""Binary per-image and foreground multiclass per-volume segmentation metrics."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def binary_metrics(prediction, target):
    p, t = np.asarray(prediction, bool), np.asarray(target, bool)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()
    return {'dice':float((2*tp)/(2*tp+fp+fn)) if 2*tp+fp+fn else 1.,
            'iou':float(tp/(tp+fp+fn)) if tp+fp+fn else 1.,
            'accuracy':float((tp+tn)/p.size),
            'sensitivity':float(tp/(tp+fn)) if tp+fn else 1.,
            'specificity':float(tn/(tn+fp)) if tn+fp else 1.}


def volume_metrics(prediction, target):
    p, t = np.asarray(prediction, bool), np.asarray(target, bool)
    dice = binary_metrics(p, t)['dice']
    # HD95 is undefined when exactly one set is empty; report it explicitly.
    if not p.any() or not t.any():
        return {'dice':dice, 'hd95':0. if not p.any() and not t.any() else None}
    ps, ts = p ^ binary_erosion(p), t ^ binary_erosion(t)
    distances = np.concatenate((distance_transform_edt(~ts)[ps], distance_transform_edt(~ps)[ts]))
    return {'dice':dice, 'hd95':float(np.percentile(distances, 95))}
