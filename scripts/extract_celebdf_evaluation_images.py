"""Extract only frozen custom-test images (+two train preview images) from existing TARs."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from train_celebdf_baseline import load_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--drive-data', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifest-root', type=Path, required=True)
    args = p.parse_args()
    frame, _ = load_manifest(args)
    test = frame[frame.split.eq('test')]
    preview = frame[frame.split.eq('train') & frame.roi_status.eq('detected')].groupby('class_index').head(1)
    wanted = pd.concat([test, preview], ignore_index=True)
    inventory = json.loads((args.drive_data/'inventory.json').read_text())
    sizes = {x['name']: x['bytes'] for x in inventory['archives']}
    def archive_for(row):
        if row.provided_split == 'test':
            return 'celebdf_v2_roi16_test_v1.tar'
        return f'celebdf_v2_roi16_non_test_{row.label}_v1.tar'
    wanted['archive'] = wanted.apply(archive_for, axis=1)
    local_root = args.data_root.parent
    local_root.mkdir(parents=True, exist_ok=True)
    for name, group in wanted.groupby('archive'):
        absent = group[~group.crop_path.map(lambda x: (args.data_root/x).is_file())]
        if absent.empty:
            print('Already available:', name, flush=True)
            continue
        source = args.drive_data/name
        if not source.is_file() or source.stat().st_size != sizes[name]:
            raise ValueError(f'Archive upload incomplete: {source}')
        if shutil.disk_usage(local_root).free < sizes[name]+2*2**30:
            raise RuntimeError('Insufficient local space for temporary TAR copy')
        local = local_root/f'smoothing_eval_{name}'
        if not local.exists() or local.stat().st_size != sizes[name]:
            print('Copying:', name, flush=True)
            shutil.copy2(source, local)
        members = local_root/f'smoothing_eval_{name}.members.txt'
        paths = [f'{args.data_root.name}/{x}' for x in absent.crop_path]
        members.write_text('\n'.join(paths)+'\n')
        print('Extracting only required images:',len(paths), flush=True)
        result = subprocess.run(['tar','-xf',str(local),'-C',str(local_root),'-T',str(members)],
                                capture_output=True,text=True)
        if result.returncode:
            print(result.stderr,flush=True)
            raise RuntimeError(f'{name}: tar extraction failed; original Drive archive preserved')
        if not absent.crop_path.map(lambda x: (args.data_root/x).is_file()).all():
            raise RuntimeError('Extraction did not produce required paths')
        local.unlink()  # only this script's temporary copy; source Drive TAR preserved
        members.unlink()
    print('Ready:',len(test),'test images +',len(preview),'train previews',flush=True)


if __name__=='__main__':
    main()
