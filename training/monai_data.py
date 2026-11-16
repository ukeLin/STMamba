"""MONAI transforms with persistent deterministic caching and in-memory augmentation input."""
from pathlib import Path
import math
import re
import numpy as np
from PIL import Image
import torch
from monai.data import CacheDataset, PersistentDataset, DataLoader
from monai.transforms import (Compose, MapTransform, EnsureTyped, Resized, SomeOf,
    RandFlipd, RandRotate90d, RandAffined, Rand2DElasticd, RandGaussianNoised,
    RandGaussianSmoothd, RandAdjustContrastd, RepeatChanneld, NormalizeIntensityd)
from training.data import pairs, remap_synapse


class LoadRecord(MapTransform):
    def __init__(self):
        super().__init__(['image_path'])

    def __call__(self, record):
        result = dict(record)
        if record['kind'] == 'binary':
            with Image.open(record['image_path']) as source:
                image = np.asarray(source.convert('RGB'), dtype=np.float32).transpose(2,0,1)/255
            with Image.open(record['label_path']) as source:
                label = (np.asarray(source.convert('L')) > 127).astype(np.float32)[None]
        else:
            with np.load(record['image_path']) as source:
                image = np.asarray(source['img' if record['kind']=='acdc' else 'image'], dtype=np.float32)[None]
                label = source['label']
            if record['kind'] == 'acdc':
                image = (image+1)/2  # Server ACDC files are already stored in [-1,1].
                if label.min()<0 or label.max()>3:
                    raise ValueError('ACDC expects background/RV/myocardium/LV labels 0..3')
            else:
                label = remap_synapse(label, record['label_format'])
            label = np.asarray(label,dtype=np.float32)[None]
        result['image'],result['label'] = image,label
        return result


class ClampUnit(MapTransform):
    def __init__(self):
        super().__init__(['image'])

    def __call__(self, data):
        result=dict(data)
        result['image']=torch.as_tensor(result['image']).clamp(0,1)
        return result


def records_for(config, root, split):
    root=Path(root)
    if config['task']=='binary':
        records=[{'image_path':str(image.resolve()),'label_path':str(mask.resolve()),
                  'case_name':image.stem,'kind':'binary'} for image,mask in pairs(root,split)]
    else:
        acdc=config['dataset']=='ACDC'
        filename='valid.txt' if acdc and split=='val' else f'{split}.txt'
        names=(Path(config['list_dir'])/filename).read_text().splitlines()
        records=[]
        for name in names:
            path=root/('valid' if split=='val' else split)/name if acdc else root/'train_npz'/f'{name}.npz'
            records.append({'image_path':str(path.resolve()),'case_name':name,
                            'kind':'acdc' if acdc else 'synapse','label_format':config.get('label_format','original13')})
    for record in records:
        record['source_stats']=[(Path(record[key]).stat().st_size,Path(record[key]).stat().st_mtime_ns)
                                for key in ('image_path','label_path') if key in record]
        record['pipeline_version']=1
    return records


def build_transforms(config, train):
    size=config['image_size']
    keys=['image','label']
    resize=Resized(keys,(size,size),mode=('bicubic','nearest'),align_corners=(False,None))
    prefix=[LoadRecord(),EnsureTyped(keys,track_meta=False)]
    if config['task']=='binary':
        prefix.append(resize)  # Small RGB cache; augment online after deterministic resize.
    suffix=[]
    if train and config['task']=='binary':
        suffix += [RandFlipd(keys,prob=.5,spatial_axis=0),RandFlipd(keys,prob=.5,spatial_axis=1),
                   RandRotate90d(keys,prob=.5,max_k=3)]
    elif train:
        # The MSVM-UNet recipe's augmentation families, implemented with MONAI.
        suffix.append(SomeOf([
            RandFlipd(keys,prob=.5,spatial_axis=0),
            RandGaussianNoised('image',prob=1.,std=.005,sample_std=False),
            RandGaussianSmoothd('image',prob=1.,sigma_x=(1.,1.),sigma_y=(1.,1.)),
            RandAdjustContrastd('image',prob=1.,gamma=(.5,1.5),retain_stats=True),
            RandAffined(keys,prob=1.,scale_range=((-0.5,1.),(-0.5,1.)),mode=('bilinear','nearest'),padding_mode='zeros'),
            RandAffined(keys,prob=1.,rotate_range=math.radians(40),mode=('bilinear','nearest'),padding_mode='zeros'),
            RandAffined(keys,prob=1.,shear_range=(math.tan(math.radians(16)),)*2,mode=('bilinear','nearest'),padding_mode='zeros'),
            Rand2DElasticd(keys,prob=1.,spacing=(32,32),magnitude_range=(.5,2.),mode=('bilinear','nearest'),padding_mode='zeros'),
            RandAffined(keys,prob=1.,translate_range=(44.8,44.8) if config['dataset']=='ACDC' else (102.4,102.4),
                        mode=('bilinear','nearest'),padding_mode='zeros'),
        ],num_transforms=(0,4),replace=False))
    if config['task']!='binary':
        suffix.append(resize)  # Native-grid medical augmentation precedes resizing.
    suffix.append(ClampUnit())
    if config['task']!='binary':
        suffix.append(RepeatChanneld('image',repeats=3))
    suffix += [NormalizeIntensityd('image',subtrahend=(.485,.456,.406),divisor=(.229,.224,.225),channel_wise=True),
               EnsureTyped(keys,track_meta=False)]
    return Compose(prefix),Compose(suffix)


def make_loader(config, root, split, cache_root, workers=6, seed=42):
    prefix,suffix=build_transforms(config,train=split=='train')
    suffix.set_random_state(seed=seed)
    records=records_for(config,root,split)
    folder=Path(cache_root)/'monai_v1'/config['dataset']/f'{split}_{config["image_size"]}'
    disk=PersistentDataset(records,transform=prefix,cache_dir=folder)
    dataset=CacheDataset(data=disk,transform=suffix,cache_rate=1.,num_workers=workers,progress=False)
    print(f'MONAI cache ready: {config["dataset"]}/{split}, samples={len(dataset)}, deterministic disk+RAM cache, random transforms online',flush=True)
    return DataLoader(dataset,batch_size=config['batch_size'],shuffle=split=='train',num_workers=workers,
                      pin_memory=True,persistent_workers=workers>0)


def acdc_split_counts(config,root):
    ids={};counts={}
    for name in ('train','valid','test'):
        files=(Path(config['list_dir'])/f'{name}.txt').read_text().splitlines()
        ids[name]={re.match(r'case_(\d+)',f)[1] for f in files}
        counts[name]={'files':len(files),'subjects':len(ids[name])}
        if any(not (Path(root)/name/f).is_file() for f in files):
            raise FileNotFoundError(f'Missing ACDC {name} files')
    if [len(ids[s]) for s in ('train','valid','test')]!=[70,10,20] or any(ids[a]&ids[b] for a,b in [('train','valid'),('train','test'),('valid','test')]):
        raise ValueError('ACDC requires disjoint 70/10/20 subject splits')
    return counts


def normalize_volume_batch(images,config):
    if config['dataset']=='ACDC':
        images=(images+1)/2
    images=images.float().clamp(0,1).expand(-1,3,-1,-1)
    return (images-images.new_tensor([.485,.456,.406])[None,:,None,None])/images.new_tensor([.229,.224,.225])[None,:,None,None]
