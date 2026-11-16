# STMamba v2

**Shape-Guided Token Reordering and Complementary Tree Fusion for Medical Image Segmentation**

STMamba v2 combines an unchanged VMamba-Tiny V2 encoder, a Multi-Scale Context Interaction Bridge (MCIB), and a progressive SGS/SC-CTF decoder with ordinary bilinear upsampling and convolution. SGS uses intermediate predictions to reorder tokens; SC-CTF combines reversed path pairs using feature differences, foreground probabilities, and prediction entropy.

This is the MCIB v2 implementation. Historical v1 results and decoder checkpoints should not be attributed to this revision. Existing training jobs keep their saved source snapshots and continue using their original architecture.

## Architecture

- VMamba-Tiny V2 encoder: depths `[2,2,8,2]`, dimensions `[96,192,384,768]`.
- MCIB: F2 uses 2x Pixel Unshuffle, F3 is projected directly, and F4 is bilinearly upsampled to the F3 grid (H/16). Three 96-channel branches are pre-mixed with a 1x1 convolution before one joint SS2D. Separate residual projections inject context back into F2/F3/F4; F1 is retained unchanged. The three injection scales start at 0.001.
- Four SGS paths, paired as `(P1,reverse(P1))` and `(P2,reverse(P2))`, with shared inputs and VSS weights within each pair.
- A two-level conditional fusion tree with spatial Softmax gates and an input residual.
- Decoder scales `/32,/16,/8,/4`, followed by two ordinary 2x bilinear + 3x3 Conv/BN/ReLU layers. The three skip-fusion upsampling layers use the same ordinary operation. AE, LKPE and channel-to-space expansion are removed.
- Five prediction heads support progressive guidance. ISIC uses final loss weight 1 and auxiliary weights 0.2, normalized by 1.8, with `0.4 CE + 0.6 Dice`. Synapse directly supervises only the final output with `0.5 CE + 0.5 Dice`.



## Installation

Linux, an NVIDIA GPU and a CUDA toolkit are required. Install a PyTorch build compatible with your CUDA environment before installing the remaining dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation ./kernels/selective_scan
python tools/download_pretrained.py
```

The compiler standard defaults to C++20 for PyTorch >=2.10 and C++17 otherwise. Override with `SELECTIVE_SCAN_CXX_STANDARD` if needed. This revision is validated on Python 3.12, PyTorch 2.14.0+cu130 and NVIDIA H20; other combinations have not been validated.

The download script fetches the ImageNet checkpoint from the [official VMamba release](https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth). Data and model weights are excluded from Git.

## Data

```text
data/
├── Synapse/
│   ├── train_npz/        # NPZ with image and label
│   └── test_vol_h5/      # H5 with image and label
├── ISIC2017/
│   ├── train/{images,masks}/
│   ├── val/{images,masks}/
│   └── test/{images,masks}/
└── ISIC2018/
    ├── train/{images,masks}/
    ├── val/{images,masks}/
    └── test/{images,masks}/
```

Use [official ISIC segmentation splits](https://challenge.isic-archive.com/data/): ISIC2017 **2000/150/600**, ISIC2018 **2594/100/1000** for train/validation/test. Masks use the image stem or `_segmentation.png`. Counts and split disjointness are checked.

Synapse uses the supplied **18 training / 12 test cases**, with **2211 listed training slices**. Targets are **eight organs plus background**, not thirteen organs. Configure `original13` source IDs or already converted `indexed9` labels in this exact target order:

| Target | Organ | Original source ID |
| --- | --- | --- |
| 0 | Background and excluded organs | 0,5,9,10,12,13 |
| 1 | Aorta | 8 |
| 2 | Gallbladder | 4 |
| 3 | Left kidney | 3 |
| 4 | Right kidney | 2 |
| 5 | Liver | 6 |
| 6 | Pancreas | 11 |
| 7 | Spleen | 1 |
| 8 | Stomach | 7 |

The same mapping is applied to training and test labels. A different eight-organ class order must be converted explicitly.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/synapse.json \
  --data-root data/Synapse --output runs/synapse
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/isic2017.json \
  --data-root data/ISIC2017 --output runs/isic2017
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/isic2018.json \
  --data-root data/ISIC2018 --output runs/isic2018
```

Defaults: 300 epochs, batch size 32, AdamW, learning rate `5e-4`, weight decay `1e-3`, and cosine decay to `1e-6`. The pretrained encoder and decoder are jointly optimized from the first epoch. Synapse uses FP32 with gradient norm clipping at 1.0; ISIC defaults to bf16. Precision is selected from the dataset configuration unless overridden by `--precision`.

ISIC caches deterministic decoding/resizing and loads the tensors onto the GPU once. Synapse caches raw decoded slices in CPU-accessible memory maps, applies its original random augmentation at native resolution, and then resizes to 224x224. Random augmentation remains online. Run configuration, epoch history, checkpoints and test results are saved in `--output`.

```bash
# Resume optimizer, scheduler and RNG state.
python train.py --config configs/isic2017.json --data-root data/ISIC2017 \
  --output runs/isic2017 --resume runs/isic2017/last.pt

# Evaluate a fixed checkpoint without training.
python train.py --config configs/isic2017.json --data-root data/ISIC2017 \
  --output runs/isic2017 --test-only --checkpoint runs/isic2017/last.pt
```

ISIC augmentation uses horizontal/vertical flips and random 90-degree rotations. Synapse uses the original SomeOf(0--4) augmentation recipe: vertical flipping, Gaussian noise, blur, contrast, scaling, rotation, shear, piecewise affine deformation, and translation. Both datasets use three-channel ImageNet normalization; Synapse additionally clamps augmented/resized CT intensities to [0,1] before normalization. Masks use nearest-neighbor resizing. The loss implementation is in `training/losses.py`.

The last checkpoint is tested after training. ISIC validation additionally saves `best.pt`; test data never selects checkpoints. Synapse has no extra validation partition in the supplied protocol and tests only after training. Binary metrics are per-image means at the configured input resolution, not native-resolution challenge scores. Synapse predictions are restored to the original grid for per-volume Dice and HD95 in voxel units; undefined HD95 values and their count are recorded explicitly.

## Tests

```bash
pytest tests/test_core.py
CUDA_VISIBLE_DEVICES=0 python tests/check_model.py
CUDA_VISIBLE_DEVICES=0 python tests/check_mcib.py
```

## Layout

```text
configs/       Dataset training presets
kernels/       Selective scan CUDA sources
model/         Encoder, SGS, SC-CTF and decoder
tests/         Numerical and model regression checks
tools/         Weight download and multi-GPU launcher
training/      Losses, caching, splits, metrics and runner
train.py       Training and checkpoint evaluation
```

## Acknowledgements

The encoder and selective scan build on [VMamba](https://github.com/MzeroMiko/VMamba) and [Mamba](https://github.com/state-spaces/mamba). MCIB's multi-scale context modeling is inspired by [SegMAN (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_SegMAN_Omni-scale_Context_Modeling_with_State_Space_Models_and_Local_CVPR_2025_paper.html); it uses native encoder-stage alignment at H/16, cross-scale pre-mixing and stage-specific residual redistribution. Third-party license texts are preserved in [LICENSES](LICENSES). An additional license grant for original STMamba contributions has not been specified.
