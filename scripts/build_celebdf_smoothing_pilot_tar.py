"""One self-contained native-crop TAR: FF++ source + disjoint Celeb-DF target/test."""
import argparse, csv, hashlib, io, json, re, shutil, tarfile, time
from pathlib import Path
import pandas as pd
from PIL import Image
from xception_preprocessing import crop_face_roi

def ids(row):
    return set(re.findall(r'id[0-9]+',Path(row['video_path']).stem))
def blob(tar,name,data):
    info=tarfile.TarInfo(name); info.size=len(data); info.mode=0o644
    tar.addfile(info,io.BytesIO(data))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); root=args.project_root
    ff_manifest=root/'manifests/ffpp_c0_face_roi_v1/manifest.csv'
    celeb_root=root/'data/celebdf_v2_all_face_crops16_v1'
    celeb_manifest=celeb_root/'crop_manifest.csv'
    assert json.loads((celeb_root/'summary.json').read_text())['complete']
    ff=pd.read_csv(ff_manifest,dtype=str,keep_default_na=False)
    pieces=[]
    for split,n in [('train',1000),('val',300)]:
        for method in ['original','Deepfakes','Face2Face']:
            pieces.append(ff[(ff.split==split)&(ff.method==method)].sample(n=n,random_state=42))
    ff=pd.concat(pieces).sort_values(['split','method','source_path'])
    c=pd.read_csv(celeb_manifest,dtype=str,keep_default_na=False).to_dict('records')
    test=[r for r in c if r['split']=='test']
    test_ids=set().union(*(ids(r) for r in test))
    target=[r for r in c if r['split']=='non_test_pool' and ids(r) and not ids(r)&test_ids]
    target_ids=set().union(*(ids(r) for r in target))
    assert len(test)==8288 and len(target)==1200 and not target_ids&test_ids
    assert not {r['video_id'] for r in target}&{r['video_id'] for r in test}
    rows=[]; inputs=[]
    def record(row,dataset,role,masked=False):
        origin=row['source_path']; item=hashlib.sha256((dataset+':'+origin).encode()).hexdigest()[:24]
        return {'item_id':item,'source_path':f'images/{dataset}/{item}.png',
                'original_source_path':origin,'dataset':dataset,'role':role,
                'label':'' if masked else row['label'],'method':row['method'],
                'video_id':dataset+':'+row['video_id'],
                'roi_status':row['roi_status'],'frame_index':row.get('frame_index',''),
                'group_id':dataset+':'+row.get('group_id',row['video_id'])}
    for r in ff.to_dict('records'):
        row=record(r,'ffpp','source_'+r['split']); rows.append(row)
        inputs.append(('ffpp',r,row,root/'data/ffpp_c0_v1'/r['source_path']))
    for role,pool in [('target',target),('test',test)]:
        for r in pool:
            row=record(r,'celebdf',role,masked=role=='target'); rows.append(row)
            inputs.append(('celebdf',r,row,celeb_root/r['crop_path']))
    assert len({r['item_id'] for r in rows})==len(rows)
    assert len({r['source_path'] for r in rows})==len(rows)
    protocol={'name':'celebdf_v2_smoothing_uda_pilot_v1','seed':42,
              'source_ffpp_manifest_sha256':sha(ff_manifest),'celeb_crop_manifest_sha256':sha(celeb_manifest),
              'counts':{role:sum(r['role']==role for r in rows) for role in ['source_train','source_val','target','test']},
              'target_known_ids':sorted(target_ids),'test_known_ids':sorted(test_ids),
              'target_labels':'removed from adaptation manifest; no label-based target sampling',
              'identity_audit':'disjoint filename celebrity IDs between target and test; YouTube identities unknown, no claim of full biometric audit',
              'target_limit':'only3 eligible celebrity IDs; narrow pilot, not representative full-domain result',
              'images':'native1.5x face ROI crops; fallback already letterbox299; do not crop again',
              'target_scope':'non_test_pool rows with both named IDs outside official test IDs',
              'test':'all official518 test videos,8288 frames; no test selection based on performance'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'celebdf_v2_smoothing_uda_pilot_seed42_v1.tar'
    assert not path.exists(), f'Existing TAR preserved: {path}'
    assert shutil.disk_usage(args.output_dir).free>3*2**30
    temp=path.with_suffix('.tar.partial'); assert not temp.exists()
    prefix='celebdf_v2_smoothing_uda_pilot_v1'
    print('Packing pilot:',protocol['counts'],'target IDs:',sorted(target_ids),flush=True)
    last=time.monotonic()
    with tarfile.open(temp,'w') as tar:
        for i,(dataset,original,row,image_path) in enumerate(inputs,1):
            if dataset=='celebdf':
                tar.add(image_path,arcname=f'{prefix}/{row["source_path"]}',recursive=False)
            else:
                with Image.open(image_path) as opened:
                    image=crop_face_roi(opened.convert('RGB'),original)
                buffer=io.BytesIO(); image.save(buffer,format='PNG',compress_level=6)
                blob(tar,f'{prefix}/{row["source_path"]}',buffer.getvalue())
            if time.monotonic()-last>=30:
                print(f'TAR images {i}/{len(inputs)}',flush=True); last=time.monotonic()
        text=io.StringIO(newline=''); writer=csv.DictWriter(text,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        blob(tar,f'{prefix}/manifest.csv',text.getvalue().encode())
        blob(tar,f'{prefix}/protocol.json',json.dumps(protocol,ensure_ascii=False,indent=2).encode())
    temp.replace(path)
    with tarfile.open(path) as tar:
        names=[m.name for m in tar.getmembers() if m.name.endswith('.png')]
        assert set(names)=={f'{prefix}/{r["source_path"]}' for r in rows} and len(names)==len(rows)
    inventory={'name':path.name,'bytes':path.stat().st_size,'sha256':sha(path),'images':len(rows),**protocol['counts']}
    (args.output_dir/'pilot_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (args.output_dir/'pilot_SHA256SUMS.txt').write_text(f"{inventory['sha256']}  {path.name}\n")
    print('PILOT TAR READY:',path,inventory,flush=True)
if __name__=='__main__': main()
