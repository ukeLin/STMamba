"""GPU checks for v2: single joint context scan, residual injection and plain upsampling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import torch
from model import STMamba
from model.mcib import MultiScaleContextInteractionBridge
from model.STDecoder import BilinearUpsample

torch.manual_seed(42)
torch.set_num_threads(4)
results=[]
for size in (224,256):
    features=[torch.randn(1,c,size//scale,size//scale,device='cuda',requires_grad=True)
              for c,scale in zip((96,192,384,768),(4,8,16,32))]
    bridge=MultiScaleContextInteractionBridge().cuda().eval()
    scans=[]
    hook=bridge.context.register_forward_pre_hook(lambda module,args: scans.append(list(args[0].shape)))
    output=bridge(features)
    hook.remove()
    assert scans==[[1,96,size//16,size//16]],scans
    assert bridge.context.x_proj_weight.shape[0]==4  # regular joint SS2D, one module call
    assert output[0] is features[0]
    assert all(a.shape==b.shape for a,b in zip(features,output))
    sum(value.square().mean() for value in output).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in bridge.parameters())
    with torch.no_grad():
        altered=[value.detach().clone() for value in features]
        altered[-1]=altered[-1]+.5
        assert (bridge(altered)[1]-output[1]).abs().max()>1e-7
        bridge.residual_scale.zero_()
        identity=bridge(features)
        for a,b in zip(identity,features):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    results.append({'input_size':size,'joint_scan_shape':scans[0],'single_joint_scan':True,
                    'cross_scale_effect':True,'zero_scale_exact_identity':True,'bridge_gradients':True})

bridge=MultiScaleContextInteractionBridge(dim=128).cuda().eval()
odd=[torch.randn(1,c,h,w,device='cuda') for c,h,w in
     ((96,58,62),(192,29,31),(384,15,16),(768,8,8))]
assert all(a.shape==b.shape for a,b in zip(bridge(odd),odd))

model=STMamba(num_classes=9).cuda().train()
assert sum(isinstance(m,BilinearUpsample) for m in model.decoder.modules())==5
assert not any(type(m).__name__ in ('LKPE','FLKPE','PatchExpand') for m in model.decoder.modules())
x=torch.randn(2,3,64,96,device='cuda')
y=torch.randint(0,9,(2,64,96),device='cuda')
optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-3)
before=model.bridge.cross_scale_mix.weight.detach().clone()
from training.losses import DeepSupervisionLoss
loss=DeepSupervisionLoss(0,.5,.5)(model(x),y)
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
optimizer.step()
assert not torch.equal(before,model.bridge.cross_scale_mix.weight)
model.eval()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    output=model(torch.randn(1,1,64,96,device='cuda'))
assert output.shape==(1,9,64,96) and torch.isfinite(output).all()
model.load_state_dict(model.state_dict(),strict=True)
report={'status':'PASS','mcib':results,'odd_grid_dim128':True,'ordinary_upsampling_layers':5,
        'final_only_fp32_optimizer_step':True,'rectangular_grayscale_bf16':True,
        'bridge_parameters':sum(p.numel() for p in model.bridge.parameters())}
Path('runs/checks').mkdir(parents=True,exist_ok=True)
Path('runs/checks/mcib_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
