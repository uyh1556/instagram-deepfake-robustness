"""Package completed native c23 crops without changing source splits or data."""
import argparse
import csv
import io
import json
import math
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from archive_celebdf_v2_crops import sha,add_bytes

METHODS=['original','Deepfakes','Face2Face','FaceSwap','NeuralTextures']
PREFIX='ffpp_c23_native_roi16_v1'


def csv_blob(rows):
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return f.getvalue().encode()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--crop-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args();rows=[];configs={}
    for method in METHODS:
        meta=args.crop_root/'metadata'/method
        summary=json.loads((meta/'summary.json').read_text())
        if not summary['complete'] or summary['videos']!=1000:raise ValueError(f'{method}: extraction incomplete')
        with (meta/'crop_manifest.csv').open(newline='') as f:part=list(csv.DictReader(f))
        if len(part)!=summary['images'] or {r['method'] for r in part}!={method}:raise ValueError('Method manifest mismatch')
        if any(int(r['class_index'])!=int(method!='original') for r in part):raise ValueError('Binary label mismatch')
        rows.extend(part);configs[method]=json.loads((meta/'config.json').read_text())
    if len({r['crop_path'] for r in rows})!=len(rows):raise ValueError('Duplicate crop path')
    memberships={}
    for r in rows:
        path=Path(r['crop_path'])
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe path')
        f=args.crop_root/path
        if not f.is_file() or f.stat().st_size!=int(r['png_bytes']):raise ValueError(f'Missing/changed image: {path}')
        if r['split'] in ['train','val','test']:
            for identity in r['source_ids'].split('|'):
                if identity in memberships and memberships[identity]!=r['split']:raise ValueError('Source identity leakage')
                memberships[identity]=r['split']
    partitions={
        'ffpp_c23_4methods_roi16_trainval_v1.tar':[r for r in rows if r['split'] in ['train','val']],
        'ffpp_c23_4methods_roi16_test_v1.tar':[r for r in rows if r['split']=='test']}
    excluded=[r for r in rows if r['split'] not in ['train','val','test']]
    for part in partitions.values():
        if {r['method'] for r in part}!=set(METHODS):raise ValueError('Partition lacks a method')
    args.output_root.mkdir(parents=True,exist_ok=True)
    blob=csv_blob(rows)
    catalog=args.output_root/'all_crops_catalog.csv'
    if catalog.exists() and catalog.read_bytes()!=blob:raise ValueError('Frozen catalog differs')
    if not catalog.exists():catalog.write_bytes(blob)
    if excluded:(args.output_root/'excluded_crops.csv').write_bytes(csv_blob(excluded))
    protocol={'name':'ffpp_c23_4methods_roi16_v1','extraction_root':PREFIX,
              'crop_catalog_sha256':sha(catalog),'method_configs':configs,
              'split':'Existing frozen global_source_aware_v1, NOT official split. No new assignment.',
              'rows_total':len(rows),'excluded_images':len(excluded),
              'counts':dict(Counter(r['split']+'/'+r['method'] for r in rows)),
              'stored':'All eligible native PNG crops. No class balancing applied at packaging; epoch sampler is separate.',
              'fallback':'Letterbox299 unchanged, retained and marked; no images dropped for failed face detection.'}
    config=args.output_root/'protocol.json'
    if config.exists() and json.loads(config.read_text())!=protocol:raise ValueError('Archive protocol differs')
    if not config.exists():config.write_text(json.dumps(protocol,indent=2)+'\n')
    required=0
    for name,part in partitions.items():
        if not (args.output_root/name).exists():
            required+=sum(math.ceil(int(r['png_bytes'])/512)*512+1536 for r in part)+len(csv_blob(part))+1024*1024
    free=shutil.disk_usage(args.output_root).free
    disk={'estimated_new_tar_bytes':required,'free_bytes':free,'safety_margin_bytes':3*2**30,
          'additional_free_space_needed_bytes':max(0,required+3*2**30-free)}
    (args.output_root/'disk_preflight.json').write_text(json.dumps(disk,indent=2)+'\n')
    print('TAR disk check:',{k:round(v/2**30,2) for k,v in disk.items()},'GiB',flush=True)
    if free<required+3*2**30:raise RuntimeError('Insufficient disk for both TARs; crops preserved. Free space and rerun this script; no deletion performed.')
    inventory=[]
    for name,part in partitions.items():
        dest=args.output_root/name
        expected={PREFIX+'/'+r['crop_path']:int(r['png_bytes']) for r in part}
        subset=csv_blob(part);stem=name.removesuffix('.tar')
        if not dest.exists():
            temp=dest.with_suffix('.partial.tar')
            if temp.exists():raise FileExistsError('Partial TAR remains; inspect before retry')
            print('Packing',name,len(part),'images',flush=True)
            with tarfile.open(temp,'w') as t:
                for ix,r in enumerate(part,1):
                    t.add(args.crop_root/r['crop_path'],arcname=PREFIX+'/'+r['crop_path'],recursive=False)
                    if ix%5000==0:print(name,ix,'/',len(part),flush=True)
                add_bytes(t,PREFIX+'/manifests/'+stem+'.csv',subset)
                add_bytes(t,PREFIX+'/manifests/protocol.json',json.dumps(protocol,indent=2).encode())
            temp.replace(dest)
        with tarfile.open(dest) as t:
            images={m.name:m.size for m in t.getmembers() if m.isfile() and m.name.endswith('.png')}
            if images!=expected:raise ValueError('TAR image inventory mismatch')
            if t.extractfile(PREFIX+'/manifests/'+stem+'.csv').read()!=subset:raise ValueError('TAR manifest mismatch')
        entry={'name':name,'images':len(part),'bytes':dest.stat().st_size,'sha256':sha(dest),
               'method_counts':dict(Counter(r['method'] for r in part)),
               'fallback':sum(r['roi_status']=='fallback_letterbox' for r in part)}
        inventory.append(entry);print('Ready',name,entry['bytes'],'bytes',flush=True)
    (args.output_root/'inventory.json').write_text(json.dumps({'complete':True,'archives':inventory},indent=2)+'\n')
    (args.output_root/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in inventory))
    print('ALL TARs READY:',args.output_root,flush=True)


if __name__=='__main__':main()
