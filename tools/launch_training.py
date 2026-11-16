"""Launch independent cached training jobs on explicitly selected CUDA devices."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--devices', default='0,1,2')
    parser.add_argument('--datasets', default='synapse,isic2017,isic2018')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
    snapshot = run_root/'source'
    datasets, devices = args.datasets.split(','), args.devices.split(',')
    if len(datasets) != len(devices):
        raise ValueError('Provide one device for each dataset')
    if snapshot.exists():
        raise FileExistsError(f'Run already exists: {run_root}')
    snapshot.mkdir(parents=True)
    for folder in ('model', 'training', 'configs'):
        shutil.copytree(root/folder, snapshot/folder, ignore=shutil.ignore_patterns('__pycache__', 'pretrain', '*.pyc'))
    for filename in ('train.py',):
        shutil.copy2(root/filename, snapshot/filename)
    (snapshot/'model/pretrain').symlink_to(root/'model/pretrain', target_is_directory=True)
    records = []
    for name, gpu in zip(datasets, devices):
        config = json.loads((root/'configs'/f'{name}.json').read_text())
        output = run_root/name
        output.mkdir()
        command = [sys.executable, '-u', str(snapshot/'train.py'), '--config', f'configs/{name}.json',
                   '--data-root', str((root/'data'/config['dataset']).resolve()),
                   '--output', str(output), '--cache-root', str(root/'.cache/data'),
                   '--cache-device', 'cuda', '--precision', 'bf16', '--seed', '42']
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4')
        with (output/'train.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=snapshot, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        records.append({'dataset':name, 'gpu':gpu, 'pid':child.pid, 'output':str(output), 'command':command})
    (run_root/'launch.json').write_text(json.dumps({'launched_at':datetime.now().isoformat(),
                                                  'source_snapshot':str(snapshot), 'jobs':records}, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
