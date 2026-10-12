"""Detect frozen source ROIs, or optionally export native-resolution face crops.

Does not run automatically after extraction and never changes original frames.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import random
import io
from pathlib import Path

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['detect','export'])
    parser.add_argument('--frames-root',type=Path,required=True)
    parser.add_argument('--roi-root',type=Path,required=True)
    parser.add_argument('--crop-root',type=Path)
    parser.add_argument('--scope',choices=['test','all'],default='test')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
    args=parser.parse_args()
    source=args.frames_root/'manifest.csv'
    summary=json.loads((args.frames_root/'summary.json').read_text())
    assert summary['complete'] and not summary['errors'], 'Frame extraction must finish successfully first'
    args.roi_root.mkdir(parents=True,exist_ok=True)
    if args.action=='detect':
        # No inferred identities or new train/validation split.
        with source.open(newline='') as stream:
            reader=csv.DictReader(stream); columns=reader.fieldnames
            rows=[r for r in reader if args.scope=='all' or r['split']=='test']
        assert rows
        subset=args.roi_root/'source_manifest.csv'
        from io import StringIO
        buffer=StringIO(newline=''); writer=csv.DictWriter(buffer,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
        payload=buffer.getvalue().encode()
        if subset.exists(): assert subset.read_bytes()==payload, 'Source scope changed; use new ROI output'
        else: subset.write_bytes(payload)
        provenance={'source_manifest_sha256':digest(source),'scope':args.scope,'images':len(rows),
                    'grouping':'video IDs preserved; identity/source split not yet assigned',
                    'detector':'MTCNN1.0.0; inherited FF++ ROI validation rules',
                    'expansion':1.5,'fallback':'full-frame letterbox; no rows dropped'}
        metadata=args.roi_root/'preparation.json'
        if metadata.exists(): assert json.loads(metadata.read_text())==provenance
        else: metadata.write_text(json.dumps(provenance,indent=2)+'\n')
        command=[sys.executable,'-u',str(Path(__file__).with_name('create_frozen_face_roi_manifest.py')),
                 '--data-root',str(args.frames_root),'--manifest',str(subset),
                 '--output-dir',str(args.roi_root/'detections'),'--expansion','1.5',
                 '--batch-size',str(args.batch_size),'--device',args.device]
        print(f'Detecting frozen source ROI: scope={args.scope}, images={len(rows)}',flush=True)
        subprocess.run(command,check=True)
        return
    assert args.crop_root is not None, 'export requires --crop-root'
    manifest=args.roi_root/'detections/manifest.csv'
    assert manifest.is_file(), 'Run detect first'
    from PIL import Image
    from xception_preprocessing import crop_face_roi
    with manifest.open(newline='') as stream: rows=list(csv.DictReader(stream))
    args.crop_root.mkdir(parents=True,exist_ok=True)
    # Bounded, deterministic size estimate before making a second PNG dataset.
    missing=[r for r in rows if not (args.crop_root/r['source_path']).is_file()]
    estimate=0
    for method in sorted({r['method'] for r in missing}):
        pool=[r for r in missing if r['method']==method]
        samples=random.Random(42).sample(pool,min(24,len(pool)))
        sizes=[]
        for row in samples:
            with Image.open(args.frames_root/row['source_path']) as opened:
                image=crop_face_roi(opened.convert('RGB'),row)
            buffer=io.BytesIO(); image.save(buffer,format='PNG',compress_level=6)
            sizes.append(buffer.tell())
        estimate+=sum(sizes)/len(sizes)*len(pool)
    free=shutil.disk_usage(args.crop_root).free
    print(f'Estimated additional crop PNGs: {estimate/2**30:.2f} GiB; free: {free/2**30:.2f} GiB',flush=True)
    if free<estimate*1.2+3*2**30:
        raise RuntimeError('Insufficient disk for native crops with safety margin. Source data preserved; free space and rerun export.')
    protocol={'roi_manifest_sha256':digest(manifest),'frames_root':str(args.frames_root.resolve()),
              'crop':'frozen 1.5x square, RGB128 padding; native ROI size, no resize',
              'fallback':'same full-frame letterbox299 as imported FF++ checkpoint',
              'source_pngs_untouched':True}
    cfg=args.crop_root/'config.json'
    if cfg.exists(): assert json.loads(cfg.read_text())==protocol
    else: cfg.write_text(json.dumps(protocol,indent=2)+'\n')
    last=time.monotonic(); total=0
    records=[]
    for i,row in enumerate(rows,1):
        relative=Path(row['source_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path=args.crop_root/relative
        if not path.exists():
            assert shutil.disk_usage(args.crop_root).free>3*2**30, 'Free disk below3 GiB; rerun after freeing space'
            with Image.open(args.frames_root/relative) as opened:
                crop=crop_face_roi(opened.convert('RGB'),row)
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.tmp.png'); crop.save(temp,format='PNG',compress_level=6); temp.replace(path)
        with Image.open(path) as existing:
            width,height=existing.size; assert existing.format=='PNG'; existing.verify()
        # Keep original source_path/ROI to avoid applying ROI twice to crops.
        records.append({**row,'crop_path':str(relative),'crop_width':width,'crop_height':height})
        total+=path.stat().st_size
        if time.monotonic()-last>=30 or i==len(rows):
            print(f'Face crops: {i}/{len(rows)}',flush=True); last=time.monotonic()
    temp=args.crop_root/'crop_manifest.csv.tmp'
    with temp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    temp.replace(args.crop_root/'crop_manifest.csv')
    (args.crop_root/'summary.json').write_text(json.dumps({'complete':True,'images':len(rows),
        'fallback_images':sum(r['roi_status']=='fallback_letterbox' for r in rows),'png_bytes':total},indent=2)+'\n')
    print('Face crops ready:',args.crop_root,flush=True)

if __name__=='__main__': main()
