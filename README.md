# STMamba v2

PyTorch implementation of STMamba for medical image segmentation, with a VMamba encoder, Multi-Scale Context Interaction Bridge (MCIB), and complementary tree fusion decoder.

## Installation

Linux with an NVIDIA GPU and CUDA toolkit.

```bash
pip install -r requirements.txt
pip install --no-build-isolation ./kernels/selective_scan
python tools/download_pretrained.py
```

## Datasets

```text
data/
├── Synapse/
│   ├── train_npz/
│   └── test_vol_h5/
├── ISIC2017/
│   ├── train/{images,masks}/
│   ├── val/{images,masks}/
│   └── test/{images,masks}/
├── ISIC2018/
│   ├── train/{images,masks}/
│   ├── val/{images,masks}/
│   └── test/{images,masks}/
└── ACDC/
    ├── train/
    ├── valid/
    └── test/
```

Synapse expects original 13-organ labels and trains on eight organs. ISIC uses the official train/validation/test splits; ACDC uses the supplied 70/10/20 subject split.

## Training

```bash
python train.py --config configs/synapse.json --data-root data/Synapse --output runs/synapse
python train.py --config configs/isic2017.json --data-root data/ISIC2017 --output runs/isic2017
python train.py --config configs/isic2018.json --data-root data/ISIC2018 --output runs/isic2018
python train.py --config configs/acdc.json --data-root data/ACDC --output runs/acdc
```

## Acknowledgements

[VMamba](https://github.com/MzeroMiko/VMamba) · [Mamba](https://github.com/state-spaces/mamba) · [SegMAN](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_SegMAN_Omni-scale_Context_Modeling_with_State_Space_Models_and_Local_CVPR_2025_paper.html) · [MONAI](https://github.com/Project-MONAI/MONAI) · [MSVM-UNet](https://github.com/gndlwch2w/msvm-unet)

Third-party licenses: [LICENSES](LICENSES).
