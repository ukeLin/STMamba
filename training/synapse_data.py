"""Original Synapse augmentation protocol, with raw decoding cached before augmentation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import imgaug as ia
import imgaug.augmenters as iaa
from training.data import remap_synapse

# imgaug 0.4 uses the NumPy alias removed in NumPy 1.24.
np.bool = bool


def normalize_ct(images):
    """Clamp resized/augmented [0,1] CT, replicate channels, apply ImageNet statistics."""
    images = images.float().clamp(0, 1).expand(-1, 3, -1, -1)
    mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = images.new_tensor([.229, .224, .225])[None, :, None, None]
    return (images - mean) / std


def original_augmentation():
    return iaa.SomeOf((0, 4), [
        iaa.Flipud(0.5, name='Flipud'),
        iaa.AdditiveGaussianNoise(scale=0.005),
        iaa.GaussianBlur(sigma=1.0),
        iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
        iaa.Affine(scale={'x': (0.5, 2), 'y': (0.5, 2)}),
        iaa.Affine(rotate=(-40, 40)),
        iaa.Affine(shear=(-16, 16)),
        iaa.PiecewiseAffine(scale=(0.008, 0.03)),
        iaa.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}),
    ], random_order=True)


def augment_and_resize(image, label, augmenter, size):
    semantic = np.stack([label == k for k in range(9)], axis=-1).astype(np.int32)
    deterministic = augmenter.to_deterministic()
    image = deterministic.augment_image(image)
    segmentation = ia.SegmentationMapsOnImage(semantic, shape=image.shape)
    label = np.argmax(deterministic.augment_segmentation_maps(segmentation).get_arr(), axis=-1).astype(np.float32)
    if image.shape != (size, size):
        factors = (size/image.shape[0], size/image.shape[1])
        image = zoom(image, factors, order=3)
        label = zoom(label, factors, order=0)
    return np.asarray(image, np.float32), np.asarray(label, np.float32)


def seed_worker(worker_id):
    ia.seed(torch.initial_seed() % (2**32))


class CachedSynapseDataset(Dataset):
    def __init__(self, config, root, cache_root, workers=6):
        self.size = config['image_size']
        names = (Path(config['list_dir'])/'train.txt').read_text().splitlines()
        files = [Path(root)/'train_npz'/f'{name}.npz' for name in names]
        label_format = config.get('label_format', 'original13')
        signature = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in files]
        key = hashlib.sha256(json.dumps(['raw_float32_v1', label_format, signature]).encode()).hexdigest()[:16]
        folder = Path(cache_root)/'Synapse'/f'raw_{key}'
        folder.mkdir(parents=True, exist_ok=True)
        ip, mp, done = folder/'images.npy', folder/'labels.npy', folder/'complete.json'
        if not done.is_file():
            with np.load(files[0]) as first:
                shape = first['image'].shape
            images = np.lib.format.open_memmap(str(ip)+'.part', mode='w+', dtype=np.float32, shape=(len(files), *shape))
            labels = np.lib.format.open_memmap(str(mp)+'.part', mode='w+', dtype=np.uint8, shape=(len(files), *shape))

            def decode(path):
                with np.load(path) as item:
                    return np.asarray(item['image'], np.float32), remap_synapse(item['label'], label_format)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for index, (image, label) in enumerate(pool.map(decode, files)):
                    images[index], labels[index] = image, label
                    if index % 250 == 0:
                        print(f'raw cache Synapse: {index+1}/{len(files)}', flush=True)
            images.flush(); labels.flush()
            del images, labels
            os.replace(str(ip)+'.part', ip); os.replace(str(mp)+'.part', mp)
            done.write_text(json.dumps({'count':len(files),'raw_shape':shape,'stage':'before augmentation and resizing'}))
        self.images = np.load(ip, mmap_mode='r')
        self.labels = np.load(mp, mmap_mode='r')
        self.augmenter = original_augmentation()
        print(f'loaded raw Synapse cache: {len(self.images)}; online original augmentation before resizing', flush=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = augment_and_resize(np.array(self.images[index], copy=True),
            np.array(self.labels[index], copy=True), self.augmenter, self.size)
        return normalize_ct(torch.from_numpy(image)[None, None])[0], torch.from_numpy(label)[None]
