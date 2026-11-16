"""Checkpoint retention and isolated periodic evaluation."""
from contextlib import contextmanager
import os
from pathlib import Path
import random

import numpy as np
import torch


def save_checkpoint(path, state):
    temporary = str(path) + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


def save_recent_weights(output, state, keep):
    if keep <= 0:
        return
    folder = Path(output) / 'checkpoints'
    folder.mkdir(exist_ok=True)
    weights = {key: state[key] for key in ('model', 'epoch', 'config')}
    save_checkpoint(folder / f'epoch_{state["epoch"]:04d}.pt', weights)
    files = sorted(folder.glob('epoch_[0-9][0-9][0-9][0-9].pt'))
    for path in files[:-keep]:
        path.unlink()


@contextmanager
def preserve_rng(model):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training = model.training
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(training)
