import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.skip_fusion import LocalDifferenceFusion
from model.complementary_fusion import foreground_probability, prediction_condition
from model.STDecoder import Decoder


def test_split_class_evidence_remains_foreground():
    probabilities = torch.tensor([.1, .4, .3, .2]).view(1, 4, 1, 1)
    logits = probabilities.log().requires_grad_()
    fg, condition = prediction_condition(logits, (3, 5))
    torch.testing.assert_close(fg, torch.full_like(fg, .9))
    assert (foreground_probability(logits, mode='max') < .5).all()
    torch.testing.assert_close(Decoder.foreground(logits), foreground_probability(logits))
    fg.sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    torch.testing.assert_close(condition[:, :1], fg)
    binary = torch.tensor([[[[-2., 0., 2.]]]])
    torch.testing.assert_close(foreground_probability(binary), binary.sigmoid())


@pytest.mark.parametrize('channels,size', [(96, (13, 17)), (192, (14, 14)), (384, (7, 9))])
def test_skip_fusion_preserves_base_and_trains(channels, size):
    module = LocalDifferenceFusion(channels)
    deep, skip, base = [torch.randn(2, channels, *size, requires_grad=True) for _ in range(3)]
    out, weights = module(deep, skip, base, return_details=True)
    assert out.shape == base.shape
    torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
    torch.testing.assert_close(weights, torch.full_like(weights, .5))
    out.square().mean().backward()
    for x in (deep, skip, base):
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert module.gate[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        module.residual_scale.zero_()
    torch.testing.assert_close(module(deep, skip, base), base, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='AMP check requires CUDA')
def test_skip_fusion_bf16():
    module = LocalDifferenceFusion(96).cuda()
    x = torch.randn(2, 96, 14, 14, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = module(x, x.flip(-1), x)
        loss = out.float().square().mean()
    loss.backward()
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()
