"""Convert the Supervisely GlaS mirror to the 85/20/60 binary segmentation protocol."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image


SPLITS = {'train': ('train', 'train', 85),
          'val': ('test_b', 'testB', 20),
          'test': ('test_a', 'testA', 60)}


def decode_annotation(annotation):
    height, width = annotation['size']['height'], annotation['size']['width']
    instances = np.zeros((height, width), dtype=np.uint16)
    for index, obj in enumerate(annotation['objects'], 1):
        if obj['geometryType'] != 'bitmap' or index > 65535:
            raise ValueError('Expected bitmap gland instances')
        bitmap = obj['bitmap']
        png = zlib.decompress(base64.b64decode(bitmap['data']))
        with Image.open(BytesIO(png)) as image:
            if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
                mask = np.asarray(image.convert('RGBA'))[:, :, 3] > 0
            else:
                mask = np.asarray(image.convert('L')) > 0
        x, y = bitmap['origin']
        h, w = mask.shape
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError('Bitmap lies outside the image')
        region = instances[y:y+h, x:x+w]
        region[mask] = index
    return (instances > 0).astype(np.uint8) * 255, instances


def prepare(raw, output):
    raw, output = Path(raw), Path(output)
    # Validate all source names and destination folders before writing anything.
    for split, (source, prefix, count) in SPLITS.items():
        images = list((raw/source/'img').glob('*.bmp'))
        expected = {f'{prefix}_{i}.bmp' for i in range(1, count+1)}
        if {p.name for p in images} != expected:
            raise ValueError(f'Unexpected {source} images; expected {count} original images')
        if any(not (raw/source/'ann'/(p.name+'.json')).is_file() for p in images):
            raise FileNotFoundError(f'Missing {source} annotation')
        if (output/split).exists() or (output/'instances'/split).exists():
            raise FileExistsError(f'Destination already exists for {split}')
    manifest = {'dataset': 'GlaS', 'task': 'binary semantic segmentation',
                'protocol': {'train': 'original Train', 'val': 'original Test B',
                             'test': 'original Test A'}, 'splits': {}}
    for split, (source, prefix, count) in SPLITS.items():
        for folder in (output/split/'images', output/split/'masks', output/'instances'/split):
            folder.mkdir(parents=True)
        names = []
        for index in range(1, count+1):
            stem = f'{prefix}_{index}'
            annotation = json.loads((raw/source/'ann'/f'{stem}.bmp.json').read_text())
            mask, instances = decode_annotation(annotation)
            with Image.open(raw/source/'img'/f'{stem}.bmp') as image:
                if image.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f'Image/annotation size mismatch: {stem}')
                image.convert('RGB').save(output/split/'images'/f'{stem}.png')
            Image.fromarray(mask).save(output/split/'masks'/f'{stem}.png')
            Image.fromarray(instances).save(output/'instances'/split/f'{stem}.png')
            names.append(stem)
        manifest['splits'][split] = names
    (output/'splits_85_20_60.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'counts': {k: len(v) for k,v in manifest['splits'].items()},
                      'protocol': manifest['protocol']}, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, help='Extracted Supervisely project')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.raw, args.output)
