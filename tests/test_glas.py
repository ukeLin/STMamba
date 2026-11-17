import base64
from io import BytesIO
import sys
from pathlib import Path
import zlib
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_glas import decode_annotation, SPLITS


def bitmap(mask, origin, alpha=False):
    if alpha:
        rgba = np.full((*mask.shape, 4), 255, dtype=np.uint8)
        rgba[:, :, 3] = mask.astype(np.uint8)*255
        image = Image.fromarray(rgba)
    else:
        image = Image.fromarray(mask.astype(np.uint8)*255).convert('P')
        image.info['transparency'] = 0
    buffer = BytesIO(); image.save(buffer, format='PNG')
    return {'geometryType':'bitmap', 'bitmap':{'origin':origin,
            'data':base64.b64encode(zlib.compress(buffer.getvalue())).decode()}}


def test_bitmap_offsets_and_low_instance_ids_become_foreground():
    a = np.array([[1,0,1],[0,1,0]], dtype=bool)
    b = np.ones((2,2), dtype=bool)
    annotation = {'size':{'height':9,'width':13}, 'objects':[bitmap(a,[2,1]),bitmap(b,[8,5],True)]}
    binary, instances = decode_annotation(annotation)
    expected = np.zeros((9,13),dtype=np.uint16)
    expected[1:3,2:5] = a
    expected[5:7,8:10] = 2
    np.testing.assert_array_equal(instances, expected)
    np.testing.assert_array_equal(binary, (expected>0).astype(np.uint8)*255)
    assert binary.sum()//255 == 7
    assert SPLITS == {'train':('train','train',85),'val':('test_b','testB',20),'test':('test_a','testA',60)}


def test_empty_and_out_of_bounds_annotations():
    annotation = {'size':{'height':3,'width':4},'objects':[]}
    binary, instances = decode_annotation(annotation)
    assert not binary.any() and not instances.any()
    annotation['objects'] = [bitmap(np.ones((2,2),dtype=bool),[3,2])]
    with pytest.raises(ValueError, match='outside'):
        decode_annotation(annotation)
