"""Copy/extract the two locally prepared FF++ c23 ROI16 archives in Colab."""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ARCHIVES=['ffpp_c23_4methods_roi16_trainval_v1.tar','ffpp_c23_4methods_roi16_test_v1.tar']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--drive-data',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args()
    inventory=json.loads((args.drive_data/'inventory.json').read_text())
    if not inventory['complete']:raise RuntimeError('TAR preparation has not completed')
    entries={x['name']:x for x in inventory['archives']}
    for name in ARCHIVES:
        manifest=args.data_root/'manifests'/f'{name.removesuffix(".tar")}.csv'
        if manifest.is_file():
            print('Already extracted:',name,flush=True);continue
        source=args.drive_data/name
        if not source.is_file() or source.stat().st_size!=entries[name]['bytes']:
            raise RuntimeError(f'TAR absent or upload incomplete: {source}')
        if shutil.disk_usage(args.data_root.parent).free<2*source.stat().st_size+2*2**30:
            raise RuntimeError('Insufficient Colab local disk for archive copy + extraction')
        local=args.data_root.parent/(name+'.localcopy')
        print('Copying:',name,flush=True);last=time.monotonic();done=0
        with source.open('rb') as src,local.open('wb') as dst:
            while block:=src.read(16*1024*1024):
                dst.write(block);done+=len(block)
                if time.monotonic()-last>30:
                    print(f'Copy: {done/source.stat().st_size:.0%}',flush=True);last=time.monotonic()
        if local.stat().st_size!=entries[name]['bytes']:raise RuntimeError('Incomplete local TAR copy')
        print('Extracting:',name,flush=True)
        result=subprocess.run(['tar','-xf',str(local),'-C',str(args.data_root.parent)],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(f'TAR extraction failed:\n{result.stderr[-5000:]}')
        if not manifest.is_file():raise RuntimeError('Unexpected TAR extraction root/manifest')
        local.unlink()
    print('FF++ c23 Train/Val/Test data ready:',args.data_root,flush=True)


if __name__=='__main__':main()
