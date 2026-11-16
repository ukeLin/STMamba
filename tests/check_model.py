"""Run from repository root: python tests/check_model.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import torch
from model.vmamba.saliency_scan import shape_guided_paths, scan, merge
from model import STMamba
from model.STMamba import STMamba as PublicModel
from paper_loss import DeepSupervisionLoss

torch.set_num_threads(4)
torch.manual_seed(42)
assert STMamba is PublicModel

# Oracle exercises a blank row and nearest-endpoint direction selection.
mask = torch.tensor([[[[0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 0, 0, 1]]]], dtype=torch.float)
fg = [3, 4, 14, 11, 10]
bg = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13]
expected = [fg + bg, bg + fg, list(reversed(bg + fg)), list(reversed(fg + bg))]
assert shape_guided_paths(mask)[0].tolist() == expected
for prior in (mask, torch.zeros_like(mask), torch.ones_like(mask)):
    paths = shape_guided_paths(prior)
    assert torch.equal(paths.sort(-1).values, torch.arange(15).expand_as(paths))
    for path in range(4):
        x = torch.randn(1, 2, 3, 5, dtype=torch.double, requires_grad=True)
        result = merge(scan(x, prior, path).reshape(1, 1, 2, 3, 5), prior, path)
        torch.testing.assert_close(result, x, rtol=0, atol=0)
        weights = torch.randn_like(x)
        (result * weights).sum().backward()
        torch.testing.assert_close(x.grad, weights, rtol=0, atol=0)
        assert torch.autograd.gradcheck(lambda z: merge(scan(z, prior, path).reshape(1, 1, 2, 3, 5), prior, path), (x,))
print('PASS: Algorithm 1 oracle, full/empty masks, rectangular permutations and gradcheck', flush=True)

results = []
for classes, size in ((1, 256), (4, 224), (9, 224)):
    model = STMamba(num_classes=classes).cuda().train()
    x = torch.randn(1, 3, size, size, device='cuda')
    labels = torch.randint(0, 2 if classes == 1 else classes, (1, 1, size, size), device='cuda')
    outputs = model(x, return_aux=True)
    assert len(outputs) == 5 and outputs[-1].shape == (1, classes, size, size)
    for output in outputs:
        output.retain_grad()
    objective = DeepSupervisionLoss()
    loss = objective(outputs, labels)
    torch.testing.assert_close(loss, (objective.single(outputs[-1], labels) + .2*torch.stack([objective.single(o, labels) for o in outputs[:-1]]).sum())/1.8)
    loss.backward()
    assert all(o.grad is not None and torch.isfinite(o.grad).all() and o.grad.abs().sum() > 0 for o in outputs)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    model.freeze_encoder()
    assert not any(p.requires_grad for p in model.encoder.parameters())
    model.unfreeze_encoder()
    assert all(p.requires_grad for p in model.encoder.parameters())
    model.eval()
    with torch.no_grad():
        output = model(x)
        assert output.shape == outputs[-1].shape
        # Fix features, change only the guidance, then inspect actual SSM output.
        group = model.decoder.deep_stage.blocks[0].pair_vss_14
        from model.vmamba.saliency_scan import CrossScan_Regular, CrossMerge_Regular
        feature = torch.randn(1, 192, 7, 7, device='cuda')
        p0 = torch.zeros(1, 1, 7, 7, device='cuda')
        p1 = p0.clone(); p1[:, :, 1:6, 1:6] = 1
        a = group(feature, CrossScan=CrossScan_Regular, CrossMerge=CrossMerge_Regular, saliency_mask=p0)
        b = group(feature, CrossScan=CrossScan_Regular, CrossMerge=CrossMerge_Regular, saliency_mask=p1)
        delta = (a-b).abs().max().item()
        assert delta > 1e-7, delta
    record = dict(classes=classes, size=size, shapes=[list(o.shape) for o in outputs], loss=loss.item(), prior_effect_max=delta)
    results.append(record)
    print('PASS:', record, flush=True)
    del model, outputs, output, loss, x
    torch.cuda.empty_cache()
Path('runs/checks').mkdir(parents=True, exist_ok=True)
Path('runs/checks/model_checks.json').write_text(json.dumps(results, indent=2))
