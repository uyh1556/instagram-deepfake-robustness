"""Split packaging only: preserve every original image and canonical manifest byte."""
import argparse
import csv
import io
import json
import shutil
import tarfile
import time
from pathlib import Path

from archive_celebdf_v2_crops import sha,add_bytes

SLUGS={'original':'real','Deepfakes':'deepfakes','Face2Face':'face2face','FaceSwap':'faceswap','NeuralTextures':'neuraltextures'}
PREFIX='ffpp_c23_native_roi16_v1'
MASTER='ffpp_c23_4methods_roi16_trainval_v1.csv'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--crop-root',type=Path,required=True)
    p.add_argument('--export-root',type=Path,required=True)
    args=p.parse_args()
    old=json.loads((args.export_root/'inventory.json').read_text())
    combined=args.export_root/'ffpp_c23_4methods_roi16_trainval_v1.tar'
    with tarfile.open(combined) as t:
        master=t.extractfile(PREFIX+'/manifests/'+MASTER).read()
        protocol=t.extractfile(PREFIX+'/manifests/protocol.json').read()
    rows=list(csv.DictReader(io.StringIO(master.decode())))
    if len({r['crop_path'] for r in rows})!=len(rows):raise ValueError('Duplicate canonical manifest rows')
    inventory=[]
    for method,slug in SLUGS.items():
        part=[r for r in rows if r['method']==method]
        if not part:raise ValueError('Missing method')
        name=f'ffpp_c23_{slug}_roi16_trainval_v1.tar';dest=args.export_root/name
        marker=name.removesuffix('.tar')+'.csv'
        f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(part[0]));w.writeheader();w.writerows(part)
        subset=f.getvalue().encode()
        expected={PREFIX+'/'+r['crop_path']:int(r['png_bytes']) for r in part}
        if not dest.exists():
            needed=sum(expected.values())+len(master)+len(subset)+len(part)*2048
            if shutil.disk_usage(args.export_root).free<needed+3*2**30:raise RuntimeError('Insufficient space; existing data preserved')
            temporary=dest.with_suffix('.partial.tar')
            if temporary.exists():raise FileExistsError(temporary)
            print('Packing',method,len(part),'images',flush=True)
            last=time.monotonic()
            with tarfile.open(temporary,'w') as t:
                for ix,r in enumerate(part,1):
                    path=args.crop_root/r['crop_path']
                    if not path.is_file() or path.stat().st_size!=int(r['png_bytes']):raise ValueError('Missing/changed crop')
                    t.add(path,arcname=PREFIX+'/'+r['crop_path'],recursive=False)
                    if time.monotonic()-last>20:print(method,ix,'/',len(part),flush=True);last=time.monotonic()
                add_bytes(t,PREFIX+'/manifests/'+marker,subset)
                add_bytes(t,PREFIX+'/manifests/protocol.json',protocol)
                if method=='original':add_bytes(t,PREFIX+'/manifests/'+MASTER,master)
            temporary.replace(dest)
        with tarfile.open(dest) as t:
            actual={m.name:m.size for m in t.getmembers() if m.isfile() and m.name.endswith('.png')}
            if actual!=expected or t.extractfile(PREFIX+'/manifests/'+marker).read()!=subset:raise ValueError('TAR inventory mismatch')
            if method=='original' and t.extractfile(PREFIX+'/manifests/'+MASTER).read()!=master:raise ValueError('Canonical manifest changed')
        entry={'name':name,'method':method,'images':len(part),'bytes':dest.stat().st_size,'sha256':sha(dest),'manifest':marker}
        inventory.append(entry);print('Ready:',name,entry['bytes'],'bytes',flush=True)
    test=next(x for x in old['archives'] if x['name']=='ffpp_c23_4methods_roi16_test_v1.tar')
    if not (args.export_root/test['name']).is_file():raise FileNotFoundError('Existing test TAR missing')
    inventory.append(test)
    result={'complete':True,'layout':'method_trainval_v1','canonical_trainval_manifest_sha256':__import__('hashlib').sha256(master).hexdigest(),'archives':inventory}
    (args.export_root/'method_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.export_root/'METHOD_SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in inventory))
    print('ALL FIVE METHOD TARs READY:',args.export_root,flush=True)


if __name__=='__main__':main()
