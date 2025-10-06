<div align="center">
<h1>STMamba: Shape-consistent Modeling and Tree-guided Fusion for Medical Image Segmentation</h1>

</div>

## Abstract

Medical image segmentation requires models that preserve lesion spatial continuity while maintaining computational efficiency. Recently, vision Mamba (ViM) models have gained attention for their ability to capture long-range dependencies with linear complexity. However, conventional scanning strategies often disrupt lesion continuity by mixing lesions and background within scanning sequences. To address this, we propose STMamba, which introduces a shape-consistent scanning strategy guided by shape priors to preserve lesion continuity. Furthermore, to enrich feature diversity and control computational cost, we adopt a group-based design that applies variant shape-consistent scanning paths in parallel. In addition, we design a tree-guided fusion strategy that hierarchically aggregates multi-group representations, ensuring more coherent and consistent feature integration. An adaptive expansion layer is further employed to enhance upsampling by jointly modeling spatial and channel information. Extensive experiments on five public datasets, including ISIC2017, ISIC2018, Synapse, BUSI, and ACDC, demonstrate that STMamba consistently outperforms state-of-the-art methods. Our code is available at https://github.com/ukeLin/STMamba.

## Installation

We recommend the following platforms: 

```
Python 3.8 / Pytorch 2.0.0 / NVIDIA GeForce RTX 3090 / CUDA 11.8.0 / Ubuntu
```

In addition, you need to install the necessary packages using the following instructions:

```bash
pip install -r requirements.txt
```

And install a runtime environment that supports Mamba:

```bash
cd ./kernels/selective_scan
pip install -e .
```

## Prepare data & Pretrained model

### Dataset:

#### ISIC datasets
- The ISIC17 divided into a 7:3 ratio, can be found here {[GoogleDrive](https://drive.google.com/file/d/1ZTOVI5Vp3KTQFDt5moJThJ_xYp2pKBAK/view?usp=sharing)}.
- The ISIC18 divided into a 7:3 ratio, can be found here {[GoogleDrive](https://drive.google.com/file/d/1AOpPgSEAfgUS2w4rCGaJBbNYbRh3Z_FQ/view?usp=sharing)}.
- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

#### Synapse datasets

- For the Synapse dataset, can be found here {[GoogleDrive](https://drive.google.com/file/d/1-eDXzTgXrTTo7hcrWZnh_wVEtB92PBNz/view?usp=sharing)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz

#### ACDC Dataset
- Download the preprocessed ACDC dataset from [Zenodo](https://zenodo.org/records/15038913) and move into `dataset/acdc/` folder.

#### ImageNet pretrained model:

You should download the pretrained VMamba-Tiny V2 model (vssm1_tiny_0230s_ckpt_epoch_264.pth) from [VMamba](https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth), and then put it in the `model/pretrain/` folder for initialization.

#### Pretrained STMamba weights:

We provide pretrained STMamba weights for five datasets:

- ISIC2017: [Download]()
- ISIC2018: [Download]()
- ACDC: [Download]()
- Synapse: [Download]()
- BUSI: [Download]()


## Training

Using the following command to train & evaluate STMamba:

```python
# Synapse Multi-Organ Dataset
python train_synapse.py
# ACDC Dataset
python train_acdc.py
```

## Citation

```
[Citation for STMamba will go here]
```

## Acknowledgements

We thank the authors of [TransUNet](https://github.com/Beckschen/TransUNet), [Mamba](https://github.com/state-spaces/mamba), [VMamba](https://github.com/MzeroMiko/VMamba), [VM-UNet](https://github.com/JCruan519/VM-UNet), and [CCViM](https://github.com/zymissy/CCViM/tree/master) for making their valuable code & data publicly available.
