"""Wait for completed crop export; package three disjoint lossless PNG pools."""
import argparse
import csv
import hashlib
import io
import json
import shutil
import tarfile
import time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def add_bytes(tar,name,blob):
    info=tarfile.TarInfo(name); info.size=len(blob); info.mode=0o644
    tar.addfile(info,io.BytesIO(blob))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--crop-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    summary_path=args.crop_root/'summary.json'
    start=time.monotonic()
    if args.wait: print('Waiting for completed crop export before TAR creation',flush=True)
    while not summary_path.exists():
        if not args.wait: raise FileNotFoundError('Crop summary not ready')
        if time.monotonic()-start>7200: raise RuntimeError('Crop export not completed within2h; no TAR created')
        time.sleep(5)
    summary=json.loads(summary_path.read_text())
    assert summary['complete']
    source=args.crop_root/'crop_manifest.csv'
    with source.open(newline='') as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==summary['images'] and len({r['crop_path'] for r in rows})==len(rows)
    assert all((args.crop_root/r['crop_path']).is_file() for r in rows)
    partitions={
        'celebdf_v2_roi16_test_v1.tar':[r for r in rows if r['split']=='test'],
        'celebdf_v2_roi16_non_test_real_v1.tar':[r for r in rows if r['split']=='non_test_pool' and r['label']=='real'],
        'celebdf_v2_roi16_non_test_fake_v1.tar':[r for r in rows if r['split']=='non_test_pool' and r['label']=='fake']}
    assert sum(len(x) for x in partitions.values())==len(rows)
    args.output_root.mkdir(parents=True,exist_ok=True)
    root='celebdf_v2_face_crops16_v1'
    identity={'crop_manifest_sha256':sha(source),'crop_config_sha256':sha(args.crop_root/'config.json'),
              'pngs_only':'native frozen face ROI, no new resize/encoding; marked fallback included',
              'non_test_pool':'candidate pool only; NOT assigned train/val; UDA must not use target class labels',
              'extraction_root':root,'parts':{name:len(part) for name,part in partitions.items()}}
    cfg=args.output_root/'config.json'
    if cfg.exists(): assert json.loads(cfg.read_text())==identity
    else: cfg.write_text(json.dumps(identity,indent=2)+'\n')
    inventory=[]
    for name,part in partitions.items():
        path=args.output_root/name
        expected={f'{root}/{r["crop_path"]}' for r in part}
        if path.exists():
            print('Verifying existing TAR:',name,flush=True)
        else:
            required=sum((args.crop_root/r['crop_path']).stat().st_size+1024 for r in part)
            assert shutil.disk_usage(args.output_root).free>required+3*2**30, 'Insufficient space for TAR; source crops preserved'
            temp=path.with_suffix('.tar.partial')
            assert not temp.exists(), f'Incomplete TAR remains: {temp}; inspect before rerunning'
            print('Packing:',name,'images=',len(part),flush=True)
            last=time.monotonic()
            with tarfile.open(temp,'w') as tar:
                for i,row in enumerate(part,1):
                    tar.add(args.crop_root/row['crop_path'],arcname=f'{root}/{row["crop_path"]}',recursive=False)
                    if time.monotonic()-last>30:
                        print(name,f'{i}/{len(part)}',flush=True); last=time.monotonic()
                buffer=io.StringIO(newline=''); writer=csv.DictWriter(buffer,fieldnames=list(part[0]))
                writer.writeheader(); writer.writerows(part)
                stem=name.removesuffix('.tar')
                add_bytes(tar,f'{root}/manifests/{stem}.csv',buffer.getvalue().encode())
                add_bytes(tar,f'{root}/manifests/{stem}_protocol.json',json.dumps(identity,indent=2).encode())
                add_bytes(tar,f'{root}/manifests/{stem}_crop_config.json',(args.crop_root/'config.json').read_bytes())
            temp.replace(path)
        with tarfile.open(path,'r') as tar:
            members=tar.getmembers()
            images=[m.name for m in members if m.isfile() and m.name.endswith('.png')]
            assert len(images)==len(expected) and set(images)==expected
            assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
            assert all(m.size==(args.crop_root/Path(m.name).relative_to(root)).stat().st_size for m in members if m.name.endswith('.png'))
        item={'name':name,'images':len(part),'real':sum(r['label']=='real' for r in part),
              'fake':sum(r['label']=='fake' for r in part),
              'fallback':sum(r['roi_status']=='fallback_letterbox' for r in part),
              'bytes':path.stat().st_size,'sha256':sha(path)}
        inventory.append(item)
        print('TAR ready:',item,flush=True)
    (args.output_root/'inventory.json').write_text(json.dumps({'complete':True,'archives':inventory},indent=2)+'\n')
    (args.output_root/'SHA256SUMS.txt').write_text(''.join(f"{i['sha256']}  {i['name']}\n" for i in inventory))
    print('ALL TARs READY:',args.output_root,flush=True)

if __name__=='__main__': main()
