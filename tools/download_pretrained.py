"""Download the encoder checkpoint from the upstream VMamba release."""
import argparse
import os
from pathlib import Path
import urllib.request

URL = 'https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='model/pretrain/vssm1_tiny_0230s_ckpt_epoch_264.pth')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = str(output)+'.part'
    urllib.request.urlretrieve(URL, temporary)
    import torch
    checkpoint = torch.load(temporary, map_location='cpu', weights_only=False)
    if 'model' not in checkpoint or not checkpoint['model']:
        raise ValueError('Invalid VMamba checkpoint')
    os.replace(temporary, output)
    print(f'Downloaded {output} ({output.stat().st_size/1024**2:.1f} MiB)')


if __name__ == '__main__':
    main()
