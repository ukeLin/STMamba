import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from paper_loss import DeepSupervisionLoss
from training.data import remap_synapse, prepare_batch
from training.metrics import binary_metrics, volume_metrics
from model.complementary_fusion import prediction_condition, ShapeConditionedCTF
from model.vmamba.saliency_scan import shape_guided_paths, scan, merge


def test_synapse_eight_organs():
    assert remap_synapse(np.arange(14)).tolist() == [0,7,4,3,2,0,5,8,1,0,0,6,0,0]
    assert remap_synapse(np.arange(9), 'indexed9').tolist() == list(range(9))
    with pytest.raises(ValueError):
        remap_synapse(np.array([13]), 'indexed9')


def test_weighted_deep_supervision():
    torch.manual_seed(5)
    heads = [torch.randn(2, 1, size, size, requires_grad=True) for size in (2, 2, 4, 8, 16)]
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    loss = DeepSupervisionLoss()
    individual = [loss.single(head, target) for head in heads]
    torch.testing.assert_close(loss(heads, target), (individual[-1]+.2*sum(individual[:-1]))/1.8)
    torch.testing.assert_close(DeepSupervisionLoss(0)(heads, target), individual[-1])
    loss(heads, target).backward()
    assert all(head.grad is not None and torch.isfinite(head.grad).all() for head in heads)


def test_exact_path_cache_and_gradient():
    prior = torch.tensor([[[[0.,0.,1.,1.,0.],[0.,0.,0.,0.,0.],[1.,1.,0.,1.,0.]]]])
    paths = shape_guided_paths(prior)
    assert torch.equal(paths[:, 3], paths[:, 0].flip(-1))
    assert torch.equal(paths[:, 2], paths[:, 1].flip(-1))
    for path in range(4):
        x = torch.randn(1, 3, 3, 5, dtype=torch.double, requires_grad=True)
        original, cached = scan(x, prior, path), scan(x, paths, path)
        torch.testing.assert_close(original, cached, rtol=0, atol=0)
        recovered = merge(cached.reshape(1,1,3,3,5), paths, path)
        torch.testing.assert_close(recovered, x, rtol=0, atol=0)
        recovered.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


def test_full_class_entropy_and_fusion():
    logits = torch.zeros(2, 4, 3, 5, requires_grad=True)
    foreground, condition = prediction_condition(logits, (3,5))
    torch.testing.assert_close(foreground, torch.full_like(foreground, .25))
    torch.testing.assert_close(condition[:, 1], torch.ones_like(condition[:, 1]))
    fusion = ShapeConditionedCTF(16, 4)
    groups = [torch.randn(2,4,3,5) for _ in range(4)]
    output, weights = fusion(torch.randn(2,16,3,5), *groups, condition, return_details=True)
    for value in weights.values():
        torch.testing.assert_close(value.sum(1), torch.ones_like(value[:,0]))
    output.square().mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_metrics_penalize_false_positives():
    assert binary_metrics([1,1], [0,0])['dice'] == 0
    assert binary_metrics([0,0], [0,0])['dice'] == 1
    assert volume_metrics(np.ones((2,2)), np.zeros((2,2)))['hd95'] is None


def test_augmentation_preserves_label_alignment():
    masks = torch.randint(0,2,(8,1,16,16),dtype=torch.uint8)
    images = masks.expand(-1,3,-1,-1).contiguous()*255
    x, y = prepare_batch(images, masks, 'binary', augment=True)
    restored = x*torch.tensor([.229,.224,.225])[None,:,None,None]+torch.tensor([.485,.456,.406])[None,:,None,None]
    torch.testing.assert_close(restored, y.expand_as(restored), rtol=0, atol=1e-6)
