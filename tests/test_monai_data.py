import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from training.monai_data import build_transforms,normalize_volume_batch


@pytest.mark.parametrize('dataset',['Synapse','ACDC'])
def test_medical_labels_and_evaluation_normalization(tmp_path,dataset):
    image=np.linspace(0,1,32*32,dtype=np.float32).reshape(32,32)
    label=np.zeros((32,32),dtype=np.uint8)
    label[:16,:16]=1;label[16:,:16]=2;label[16:,16:]=3
    if dataset=='Synapse':
        label=np.choose(label,[0,8,4,11]).astype(np.uint8)
        stored=image
        np.savez(tmp_path/'sample.npz',image=stored,label=label)
    else:
        stored=image*2-1
        np.savez(tmp_path/'sample.npz',img=stored,label=label)
    config={'dataset':dataset,'task':'multiclass','image_size':32,'num_classes':9 if dataset=='Synapse' else 4}
    record={'image_path':str(tmp_path/'sample.npz'),'kind':'acdc' if dataset=='ACDC' else 'synapse',
            'label_format':'original13'}
    prefix,suffix=build_transforms(config,False)
    result=suffix(prefix(record))
    assert result['image'].shape==(3,32,32)
    assert set(result['label'].unique().tolist())==({0.,1.,2.,6.} if dataset=='Synapse' else {0.,1.,2.,3.})
    expected=normalize_volume_batch(torch.from_numpy(stored)[None,None],config)[0]
    torch.testing.assert_close(result['image'],expected,rtol=1e-6,atol=1e-6)
