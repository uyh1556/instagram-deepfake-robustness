"""Current local job: wait for NeuralTextures, crop originals, then package."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--crop-root',type=Path,required=True)
    p.add_argument('--anchor-manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args();scripts=Path(__file__).parent
    summary=args.crop_root/'metadata/NeuralTextures/summary.json'
    print('Waiting for active NeuralTextures job; no duplicate crop process',flush=True)
    deadline=time.monotonic()+7200
    while not summary.is_file():
        if time.monotonic()>deadline:raise RuntimeError('NeuralTextures not completed within2h; inspect existing crop job')
        time.sleep(5)
    if not json.loads(summary.read_text()).get('complete'):raise RuntimeError('NeuralTextures incomplete')
    subprocess.run([sys.executable,'-u',str(scripts/'prepare_ffpp_c23_native_roi16.py'),
        '--input-root',str(args.input_root),'--methods','original','--anchor-manifest',str(args.anchor_manifest),
        '--output-root',str(args.crop_root)],check=True)
    subprocess.run([sys.executable,'-u',str(scripts/'archive_ffpp_c23_native_roi16.py'),
        '--crop-root',str(args.crop_root),'--output-root',str(args.output_root)],check=True)


if __name__=='__main__':main()
