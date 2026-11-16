# STMamba

**Shape-Guided Token Reordering and Complementary Tree Fusion for Medical Image Segmentation**

STMamba combines a VMamba-Tiny V2 encoder with a progressive decoder. Shape-guided scanning (SGS) uses intermediate predictions to reorder tokens. Shape-conditioned complementary tree fusion (SC-CTF) combines reversed path pairs using feature differences, foreground probabilities, and prediction entropy.

This is the revised SC-CTF implementation. Its decoder differs from the earlier serial TGF/BTBB model. Historical results and decoder checkpoints should not be attributed to this revision.

## Architecture

- VMamba-Tiny V2 encoder: depths `[2,2,8,2]`, dimensions `[96,192,384,768]`.
- Four SGS paths, paired as `(P1,reverse(P1))` and `(P2,reverse(P2))`, with shared inputs and VSS weights within each pair.
- A two-level conditional fusion tree with spatial Softmax gates and an input residual.
- Decoder scales `/32,/16,/8,/4`, followed by two 2x adaptive expansion layers.
- Five supervised outputs: final weight 1, each auxiliary weight 0.2, normalized by 1.8. Each head uses `0.4 CE + 0.6 Dice`.

See [architecture and ablations](docs/architecture.md) and [training protocol](docs/training.md).

## Installation

Linux, an NVIDIA GPU and a CUDA toolkit are required. Install a PyTorch build compatible with your CUDA environment before installing the remaining dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
SELECTIVE_SCAN_MODES=oflex pip install --no-build-isolation ./kernels/selective_scan
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

Synapse uses the supplied **18 training / 12 test cases**, with **2211 listed training slices**. Targets are **eight organs plus background**, not thirteen organs. Configure `original13` source IDs or already converted `indexed9` labels using the [explicit mapping](docs/training.md#synapse-labels).

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/synapse.json \
  --data-root data/Synapse --output runs/synapse
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/isic2017.json \
  --data-root data/ISIC2017 --output runs/isic2017
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/isic2018.json \
  --data-root data/ISIC2018 --output runs/isic2018
```

Defaults: 300 epochs, batch size 32, AdamW, learning rate `5e-4`, weight decay `1e-3`, and cosine decay to `1e-6`. The pretrained encoder and decoder are jointly optimized from the first epoch. bf16 is enabled by default; use `--precision fp32` to disable it.

Deterministic decoding/resizing is cached on disk and loaded onto the GPU once. Random augmentation remains online. Use `--cache-device cpu` when GPU memory is limited. Run configuration, epoch history, checkpoints and test results are saved in `--output`.

```bash
# Resume optimizer, scheduler and RNG state.
python train.py --config configs/isic2017.json --data-root data/ISIC2017 \
  --output runs/isic2017 --resume runs/isic2017/last.pt

# Evaluate a fixed checkpoint without training.
python train.py --config configs/isic2017.json --data-root data/ISIC2017 \
  --output runs/isic2017 --test-only --checkpoint runs/isic2017/last.pt
```

The last checkpoint is tested after training. ISIC validation additionally saves `best.pt`; test data never selects checkpoints. Synapse has no extra validation partition in the supplied protocol and tests only after training.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/test_core.py
CUDA_VISIBLE_DEVICES=0 python tests/check_model.py
```

## Layout

```text
configs/       Dataset training presets
docs/          Architecture and protocol
kernels/       Selective scan CUDA sources
lists/         Synapse case/slice lists
model/         Encoder, SGS, SC-CTF and decoder
tests/         Numerical and model regression checks
tools/         Weight download and multi-GPU launcher
training/      Caching, augmentation, metrics and runner
train.py       Training and checkpoint evaluation
paper_loss.py  Weighted CE/Dice deep supervision
```

## Acknowledgements

The encoder and selective scan build on [VMamba](https://github.com/MzeroMiko/VMamba) and [Mamba](https://github.com/state-spaces/mamba). Their license texts are preserved in [LICENSES](LICENSES). An additional license grant for original STMamba contributions has not been specified.
