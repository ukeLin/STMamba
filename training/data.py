"""Cache deterministic decoding/resizing; keep stochastic augmentation online."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import torch


def remap_synapse(labels, label_format='original13'):
    labels = np.asarray(labels)
    if not np.equal(labels, np.rint(labels)).all():
        raise ValueError('Synapse labels must be integer class IDs')
    if label_format == 'indexed9':
        if labels.min() < 0 or labels.max() > 8:
            raise ValueError('indexed9 expects labels in 0..8')
        return labels.astype(np.uint8)
    if label_format != 'original13' or labels.min() < 0 or labels.max() > 13:
        raise ValueError('original13 expects labels in 0..13')
    # Target order: aorta, gallbladder, left kidney, right kidney, liver,
    # pancreas, spleen, stomach. Other original organs become background.
    result = np.zeros(labels.shape, dtype=np.uint8)
    for source, target in ((8,1), (4,2), (3,3), (2,4), (6,5), (11,6), (1,7), (7,8)):
        result[labels == source] = target
    return result


def pairs(root, split):
    root = Path(root)
    images = sorted(p for p in (root/split/'images').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    result = []
    for image in images:
        candidates = [root/split/'masks'/f'{image.stem}_segmentation.png', root/split/'masks'/f'{image.stem}.png']
        mask = next((p for p in candidates if p.is_file()), None)
        if mask is None:
            raise FileNotFoundError(f'No mask for {image.name}')
        result.append((image, mask))
    if not result:
        raise ValueError(f'Empty dataset: {root / split}')
    return result


def validate_splits(config, root):
    if config['dataset'] == 'ACDC':
        from training.monai_data import acdc_split_counts
        return acdc_split_counts(config, root)
    if config['task'] == 'binary':
        rows = {split:pairs(root, split) for split in ('train', 'val', 'test')}
        ids = {split:{p.stem for p, _ in values} for split, values in rows.items()}
        counts = {split:len(values) for split, values in rows.items()}
        if counts != config['split_counts']:
            raise ValueError(f'Dataset split mismatch: {counts}, expected {config["split_counts"]}')
    else:
        folder = Path(config.get('list_dir', 'training/splits/synapse'))
        train = (folder/'train.txt').read_text().splitlines()
        test = (folder/'test_vol.txt').read_text().splitlines()
        ids = {'train':{re.match(r'case(\d+)', x)[1] for x in train},
               'test':{re.match(r'case(\d+)', x)[1] for x in test}}
        counts = {'train_slices':len(train), 'train_cases':len(ids['train']), 'test_cases':len(test)}
        if counts['train_cases'] != 18 or counts['test_cases'] != 12:
            raise ValueError(f'Synapse case split mismatch: {counts}')
    names = list(ids)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            if ids[name] & ids[other]:
                raise ValueError(f'Split overlap: {name}/{other}')
    return counts


def cached_split(config, root, split, cache_root, device, workers=8):
    size = config['image_size']
    if config['task'] == 'binary':
        records = pairs(root, split)
    else:
        names = (Path(config.get('list_dir', 'training/splits/synapse'))/'train.txt').read_text().splitlines()
        records = [(Path(root)/'train_npz'/f'{name}.npz', None) for name in names]
    signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                 for record in records for p in record if p is not None]
    revision = 3 if config['task'] == 'binary' else [4, config.get('label_format', 'original13')]
    key = hashlib.sha256(json.dumps([revision, size, signature]).encode()).hexdigest()[:16]
    folder = Path(cache_root)/config['dataset']/f'{split}_{size}_{key}'
    folder.mkdir(parents=True, exist_ok=True)
    ip, mp, done = folder/'images.npy', folder/'masks.npy', folder/'complete.json'

    def decode(record):
        image, mask = record
        if mask is not None:
            with Image.open(image) as source:
                x = np.asarray(source.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)).copy()
            with Image.open(mask) as source:
                y = np.asarray(source.convert('L').resize((size, size), Image.Resampling.NEAREST)) > 127
            return x.transpose(2, 0, 1), y[None].astype(np.uint8)
        with np.load(image) as source:
            x, y = source['image'], source['label']
        y = remap_synapse(y, config.get('label_format', 'original13'))
        factor = (size/x.shape[0], size/x.shape[1])
        x = zoom(x, factor, order=3).astype(np.float32)
        y = zoom(y, factor, order=0).astype(np.uint8)
        if not np.isfinite(x).all() or y.max() >= config['num_classes']:
            raise ValueError(f'Invalid image or label: {image}')
        return x[None], y[None]

    if not done.is_file():
        image_dtype = np.uint8 if config['task'] == 'binary' else np.float32
        channels = 3 if config['task'] == 'binary' else 1
        images = np.lib.format.open_memmap(str(ip)+'.part', mode='w+', dtype=image_dtype, shape=(len(records), channels, size, size))
        masks = np.lib.format.open_memmap(str(mp)+'.part', mode='w+', dtype=np.uint8, shape=(len(records), 1, size, size))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (x, y) in enumerate(pool.map(decode, records)):
                images[index], masks[index] = x, y
                if index % 250 == 0:
                    print(f'cache {config["dataset"]}/{split}: {index+1}/{len(records)}', flush=True)
        images.flush(); masks.flush()
        del images, masks
        os.replace(str(ip)+'.part', ip); os.replace(str(mp)+'.part', mp)
        done.write_text(json.dumps({'count':len(records), 'size':size, 'cache_version':3}))
    images = torch.from_numpy(np.load(ip)).to(device)
    masks = torch.from_numpy(np.load(mp)).to(device)
    print(f'loaded cache {config["dataset"]}/{split}: {len(images)} on {device}', flush=True)
    return images, masks


def prepare_batch(images, masks, task, augment=False):
    images, masks = images.float(), masks.float()
    if task == 'binary':
        images = images / 255
    if augment:
        for dimension in (-1, -2):
            flip = torch.rand(len(images), 1, 1, 1, device=images.device) < .5
            images = torch.where(flip, images.flip(dimension), images)
            masks = torch.where(flip, masks.flip(dimension), masks)
        turns = torch.randint(0, 4, (len(images),), device=images.device)
        for turn in (1, 2, 3):
            selected = turns == turn
            images[selected] = images[selected].rot90(turn, (-2, -1))
            masks[selected] = masks[selected].rot90(turn, (-2, -1))
    if task == 'binary':
        mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
        std = images.new_tensor([.229, .224, .225])[None, :, None, None]
        images = (images - mean) / std
    else:
        images = images * 2 - 1
    return images, masks
