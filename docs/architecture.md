# Architecture and ablations

SGS forms its foreground path using nearest row endpoints and its background path in raster order. Every path is a complete permutation. Gather applies the path; scatter restores spatial positions. Scan/restore operations inside one block reuse the same exact paths.

SC-CTF receives full previous-stage logits. Its condition Q combines maximum non-background probability and normalized full-class entropy, including background. Binary tasks use binary entropy. Entropy measures predictive uncertainty; it is not calibrated correctness.

Two 1x1 projections yield XA and XB, each C/4 channels. P1/P4 share XA and one VSS block; P2/P3 share XB and another. Each path performs one directional selective scan. All features return to the spatial grid before fusion.

Each tree node uses a two-layer 1x1 gate on `[left,right,abs(left-right),Q]`, then Softmax over the two spatial branch weights. Level one fuses reverse pairs. Level two weights the two pairs, concatenates them and projects 2c channels to C. A learnable scalar, initialized to one, scales an identity spatial residual.

```python
from model import STMamba
model = STMamba(num_classes=9)  # full SC-CTF
model = STMamba(num_classes=9, fusion_options={"detach_condition": True})
model = STMamba(num_classes=9, fusion_options={"condition_mode": "none"})
model = STMamba(num_classes=9, fusion_options={"condition_mode": "shuffle"})
model = STMamba(num_classes=9, fusion_options={"pairing": "12_34"})
model = STMamba(num_classes=9, fusion_options={"pairing": "13_24"})
model = STMamba(num_classes=9, fusion_options={"mode": "balanced"})
model = STMamba(num_classes=9, fusion_options={"mode": "direct"})
model = STMamba(num_classes=9, fusion_options={"residual_enabled": False})
```

The default pairing is `14_23`. `none` zeros fusion conditions, preserving gate parameter count. `shuffle` cyclically shifts Q across samples, or flattened spatial positions for batch size one, without changing true SGS masks. Identical/constant conditions are invariant to shifting. Balanced/direct modes have different parameter counts and must be profiled separately.

`ShapeConditionedCTF.forward(..., return_details=True)` returns the three nodes' weights. Conditions remain differentiable by default; discrete SGS sorting is non-differentiable. For 224x224 inputs, S0--S4 sizes are 7x7, 7x7, 14x14, 28x28 and 224x224. Auxiliary logits are resized to label resolution for supervision. Normal inference returns S4; `return_aux=True` returns all five maps.
