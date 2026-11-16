import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
import h5py
import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
from loguru import logger
from model import STMamba
from training.losses import DeepSupervisionLoss
from training.data import cached_split, prepare_batch, validate_splits, remap_synapse
from training.metrics import binary_metrics, volume_metrics
from torch.utils.data import DataLoader


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(temporary, path)


def save_checkpoint(path, state):
    temporary = str(path)+'.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def binary_evaluate(model, data, config, batch_size, amp):
    model.eval()
    images, masks = data
    totals = {}
    for start in range(0, len(images), batch_size):
        x, y = prepare_batch(images[start:start+batch_size], masks[start:start+batch_size], 'binary')
        with amp():
            prediction = model(x).float().sigmoid() >= .5
        for p, t in zip(prediction.cpu().numpy(), y.cpu().numpy()):
            for key, value in binary_metrics(p, t).items():
                totals[key] = totals.get(key, 0.) + value
    return {key:value/len(images) for key, value in totals.items()}


@torch.no_grad()
def synapse_evaluate(model, config, root, batch_size, device, amp):
    model.eval()
    cases = []
    names = (Path(config.get('list_dir', 'training/splits/synapse'))/'test_vol.txt').read_text().splitlines()
    for name in names:
        with h5py.File(Path(root)/'test_vol_h5'/f'{name}.npy.h5') as source:
            image, target = np.asarray(source['image'][:], dtype=np.float32), source['label'][:]
        target = remap_synapse(target, config.get('label_format', 'original13'))
        size = config['image_size']
        resized = np.stack([zoom(sl, (size/sl.shape[0], size/sl.shape[1]), order=3) for sl in image]).astype(np.float32)
        x = torch.from_numpy(resized[:, None]).to(device)
        predictions = []
        for start in range(0, len(x), batch_size):
            with amp():
                batch = x[start:start+batch_size]
                if config.get('training_protocol') == 'legacy85':
                    from training.synapse_data import normalize_ct
                    batch = normalize_ct(batch)
                else:
                    batch = batch*2-1
                result = model(batch).float().argmax(1).cpu().numpy()
            predictions.extend(zoom(sl, (target.shape[-2]/size, target.shape[-1]/size), order=0) for sl in result)
        predicted = np.stack(predictions)
        record = {'case':name, 'organs':{str(k):volume_metrics(predicted==k, target==k)
                                       for k in range(1, config['num_classes'])}}
        cases.append(record)
        print(f'test volume {name}: dice={np.mean([v["dice"] for v in record["organs"].values()]):.4f}', flush=True)
    flat = [v for case in cases for v in case['organs'].values()]
    finite_hd = [v['hd95'] for v in flat if v['hd95'] is not None]
    return {'dice':float(np.mean([v['dice'] for v in flat])),
            'hd95_voxels':float(np.mean(finite_hd)) if finite_hd else None,
            'hd95_undefined_case_organ_pairs':len(flat)-len(finite_hd), 'cases':cases,
            'organ_means':{str(k):float(np.mean([c['organs'][str(k)]['dice'] for c in cases]))
                           for k in range(1, config['num_classes'])}}


def main():
    parser = argparse.ArgumentParser(description='Train STMamba with SC-CTF and cached deterministic preprocessing')
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache-root', default='.cache/data')
    parser.add_argument('--cache-device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', choices=['fp32','bf16'])
    parser.add_argument('--resume')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    args.precision = args.precision or config.get('precision', 'bf16')
    if args.epochs is not None: config['epochs'] = args.epochs
    if args.batch_size is not None: config['batch_size'] = args.batch_size
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled = config.get('cudnn_enabled', True)
    torch.backends.cudnn.benchmark = not config.get('deterministic', False)
    torch.backends.cudnn.deterministic = config.get('deterministic', False)
    torch.set_float32_matmul_precision(config.get('matmul_precision', 'high'))
    legacy_synapse = config.get('training_protocol') == 'legacy85'
    if legacy_synapse:
        import imgaug
        imgaug.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('STMamba requires a CUDA GPU for selective scan')
    if args.precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('bf16 requires a compatible CUDA GPU; use --precision fp32')
    amp = (lambda:torch.autocast('cuda', dtype=torch.bfloat16)) if args.precision == 'bf16' else nullcontext
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    counts = validate_splits(config, args.data_root)
    atomic_json(output/'config.json', {**config, 'args':vars(args), 'actual_splits':counts,
                                       'torch':torch.__version__, 'gpu':torch.cuda.get_device_name()})
    logger.remove(); logger.add(output/'model_initialization.log')
    pretrained = Path(__file__).resolve().parents[1]/'model/pretrain/vssm1_tiny_0230s_ckpt_epoch_264.pth'
    if not pretrained.is_file():
        raise FileNotFoundError('Missing encoder checkpoint; run python tools/download_pretrained.py')
    model = STMamba(num_classes=config['num_classes'], fusion_options=config.get('fusion_options')).to(device)
    objective = DeepSupervisionLoss(config.get('aux_weight', .2), config.get('ce_weight', .4), config.get('dice_weight', .6))
    batch_size = config['batch_size']
    if not args.test_only:
        cache_device = device if args.cache_device == 'cuda' else torch.device('cpu')
        if legacy_synapse:
            from training.synapse_data import CachedSynapseDataset, seed_worker
            workers = config.get('workers', 6)
            dataset = CachedSynapseDataset(config, args.data_root, args.cache_root, workers)
            train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                               pin_memory=True, persistent_workers=workers>0, worker_init_fn=seed_worker)
        else:
            train = cached_split(config, args.data_root, 'train', args.cache_root, cache_device, args.workers)
        val = cached_split(config, args.data_root, 'val', args.cache_root, cache_device, args.workers) if config['task']=='binary' else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['min_lr'])
        start_epoch, best = 0, -1.
        if args.resume:
            state = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            start_epoch, best = state['epoch'], state['best_val_dice']
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
        print(json.dumps({'status':'training', 'dataset':config['dataset'], 'splits':counts,
                          'epochs':config['epochs'], 'batch_size':batch_size, 'aux_weight':objective.aux_weight}), flush=True)
        for epoch in range(start_epoch, config['epochs']):
            begin = time.perf_counter()
            model.train()
            if legacy_synapse:
                batches = train
                batch_count = len(train)
            else:
                order = torch.randperm(len(train[0]), device=train[0].device)
                batches = ((train[0][indices].to(device), train[1][indices].to(device))
                           for indices in order.split(batch_size))
                batch_count = math.ceil(len(order)/batch_size)
            total, seen = 0., 0
            for step, (images, masks) in enumerate(batches):
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                if not legacy_synapse:
                    images, masks = prepare_batch(images, masks, config['task'], augment=True)
                optimizer.zero_grad(set_to_none=True)
                with amp():
                    predictions = model(images, return_aux=config.get('supervision') != 'final')
                    loss = objective(predictions, masks)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}, step {step}')
                loss.backward()
                if config.get('gradient_clip_norm'):
                    gradient_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
                if epoch == 0 and step == 0:
                    encoder_parameters = list(model.encoder.parameters())
                    gradient_norm = torch.linalg.vector_norm(torch.stack([
                        parameter.grad.detach().float().norm() for parameter in encoder_parameters
                        if parameter.grad is not None]))
                    print(json.dumps({'event':'first_step_encoder_gradient',
                                      'all_encoder_parameters_trainable':all(p.requires_grad for p in encoder_parameters),
                                      'encoder_gradient_norm':gradient_norm.item(),
                                      'precision':args.precision,'gradient_clip_norm':config.get('gradient_clip_norm'),
                                      'loss_ce_weight':objective.ce_weight,'loss_dice_weight':objective.dice_weight,
                                      'aux_weight':objective.aux_weight}), flush=True)
                optimizer.step()
                total += loss.detach().item()*len(images); seen += len(images)
                if step % 20 == 0:
                    print(f'epoch={epoch+1}/{config["epochs"]} step={step+1}/{batch_count} loss={loss.item():.6f}', flush=True)
            metrics = binary_evaluate(model, val, config, batch_size, amp) if val is not None else None
            improved = metrics is not None and metrics['dice'] > best
            if improved: best = metrics['dice']
            scheduler.step()
            record = {'epoch':epoch+1, 'train_loss':total/seen, 'lr':optimizer.param_groups[0]['lr'],
                      'seconds':time.perf_counter()-begin, 'val':metrics}
            with (output/'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
                     'epoch':epoch+1, 'best_val_dice':best, 'config':config,
                     'python_rng':random.getstate(), 'numpy_rng':np.random.get_state(),
                     'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all()}
            save_checkpoint(output/'last.pt', state)
            if improved:
                save_checkpoint(output/'best.pt', {'model':state['model'], 'epoch':epoch+1, 'val':metrics, 'config':config})
            atomic_json(output/'status.json', {'status':'training', **record, 'total_epochs':config['epochs']})
            print(json.dumps(record), flush=True)
    checkpoint = args.checkpoint or str(output/'last.pt')
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    print(f'evaluating fixed checkpoint: {checkpoint}', flush=True)
    if config['task']=='binary':
        test = cached_split(config, args.data_root, 'test', args.cache_root, device, args.workers)
        metrics = binary_evaluate(model, test, config, batch_size, amp)
    else:
        metrics = synapse_evaluate(model, config, args.data_root, batch_size, device, amp)
    result = {'status':'complete', 'checkpoint':checkpoint, 'epoch':state['epoch'], 'test':metrics}
    atomic_json(output/'test_results.json', result)
    atomic_json(output/'status.json', result)
    print(json.dumps(result), flush=True)
