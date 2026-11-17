import random
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.checkpoints import preserve_rng, save_recent_weights


def test_retains_latest_five_loadable_weights(tmp_path):
    for epoch in range(291, 301):
        save_recent_weights(tmp_path, {'model':{'weight':torch.tensor(epoch)},
                                      'epoch':epoch, 'config':{'lr':5e-4}}, 5)
    files = sorted((tmp_path/'checkpoints').glob('*.pt'))
    assert [p.name for p in files] == [f'epoch_{n:04d}.pt' for n in range(296,301)]
    for epoch, path in zip(range(296,301), files):
        state = torch.load(path, weights_only=True)
        assert state['epoch'] == state['model']['weight'].item() == epoch
        assert state['config']['lr'] == 5e-4


@pytest.mark.parametrize('fail', [False, True])
def test_evaluation_preserves_rng_and_training_mode(fail):
    model = torch.nn.Linear(2, 2).train()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    try:
        with preserve_rng(model):
            model.eval()
            random.random(); np.random.rand(7); torch.rand(20)
            if fail:
                raise RuntimeError('evaluation failure')
    except RuntimeError:
        assert fail
    assert model.training
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Training runner requires CUDA')
def test_periodic_best_and_final_evaluation(tmp_path, monkeypatch):
    from training import runner, monai_data

    class SmallModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.encoder = torch.nn.Conv2d(3, 9, 1)

        def forward(self, x, return_aux=False):
            return self.encoder(x)

    config = json.loads((Path(__file__).resolve().parents[1]/'configs/synapse.json').read_text())
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    output = tmp_path/'run'
    batch = {'image':torch.rand(2,3,8,8), 'label':torch.randint(0,9,(2,1,8,8))}
    scores = iter((.7, .8, .6))
    calls = []

    def evaluate(*args):
        calls.append(1)
        return {'dice':next(scores)}

    monkeypatch.setattr(runner, 'STMamba', SmallModel)
    monkeypatch.setattr(runner, 'validate_splits', lambda *args: {})
    monkeypatch.setattr(monai_data, 'make_loader', lambda *args: [batch])
    monkeypatch.setattr(runner, 'synapse_evaluate', evaluate)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(config_path), '--data-root', str(tmp_path),
                                    '--output', str(output), '--epochs', '3', '--lr', '.0005',
                                    '--eval-interval', '1', '--keep-last', '2'])
    runner.main()
    assert len(calls) == 3  # Final checkpoint reuses the final periodic evaluation.
    assert len(list(output.glob('test_epoch_*.json'))) == 3
    best = json.loads((output/'best_test.json').read_text())
    assert best['epoch'] == 2 and best['test']['dice'] == .8
    assert best['selection_split'] == 'test'
    assert torch.load(output/'best_test.pt', map_location='cpu', weights_only=False)['epoch'] == 2
    last = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
    assert last['epoch'] == 3 and last['best_periodic_test_dice'] == .8
    assert last['config']['lr'] == .0005
    assert json.loads((output/'test_results.json').read_text())['test']['dice'] == .6
    assert sorted(p.name for p in (output/'checkpoints').glob('*.pt')) == ['epoch_0002.pt','epoch_0003.pt']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Training runner requires CUDA')
def test_standalone_monai_binary_evaluation(tmp_path, monkeypatch):
    from training import runner, monai_data

    class SmallBinaryModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.encoder = torch.nn.Conv2d(3, 1, 1)

        def forward(self, x, return_aux=False):
            return self.encoder(x)

    config = json.loads((Path(__file__).resolve().parents[1]/'configs/isic2017.json').read_text())
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    checkpoint = tmp_path/'selected.pt'
    torch.save({'model':SmallBinaryModel().state_dict(), 'epoch':17}, checkpoint)
    calls = []

    def loader(cfg, root, split, *args):
        calls.append(split)
        return [{'image':torch.ones(2,3,8,8), 'label':torch.ones(2,1,8,8)}]

    monkeypatch.setattr(runner, 'STMamba', SmallBinaryModel)
    monkeypatch.setattr(runner, 'validate_splits', lambda *args: {})
    monkeypatch.setattr(monai_data, 'make_loader', loader)
    output = tmp_path/'evaluation'
    monkeypatch.setattr(sys, 'argv', ['train.py','--config',str(config_path),'--data-root',str(tmp_path),
                                    '--output',str(output),'--test-only','--checkpoint',str(checkpoint)])
    runner.main()
    assert calls == ['test']
    result = json.loads((output/'test_results.json').read_text())
    assert result['status'] == 'complete' and result['epoch'] == 17
    assert result['checkpoint'] == str(checkpoint)
    assert 0 <= result['test']['dice'] <= 1
