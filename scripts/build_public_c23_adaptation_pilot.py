"""Small frozen c23 source + identity-disjoint unlabelled Celeb-DF target TAR."""
import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import pandas as pd

PREFIX='public_c23_adaptation_custom_pilot_v2'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def sample(frame,n,cap,seed):
    pool=frame.sample(frac=1,random_state=seed).groupby('video_id',sort=False).head(cap)
    if len(pool)<n:raise ValueError(f'Insufficient capped pool {len(pool)} < {n}')
    return pool.sample(n=n,random_state=seed).sort_values(['video_id','frame_index'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--legacy-root',type=Path,required=True)
    p.add_argument('--celeb-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args()
    source_path=args.legacy_root/'data/ffpp/c23/df40_preprocessed/splits/global_source_aware_v1/manifest.csv'
    source=pd.read_csv(source_path,dtype=str,keep_default_na=False)
    if not source.compression.eq('c23').all():raise ValueError('Expected existing c23 source images')
    celeb_path=args.celeb_root/'crop_manifest.csv'
    celeb=pd.read_csv(celeb_path,dtype=str,keep_default_na=False)
    from create_celebdf_identity_split import partition
    custom,groups=partition(celeb,42)
    candidate=custom[custom.split.eq('train')].copy()
    target=sample(candidate,3000,4,42)
    test=custom[custom.split.eq('test')].sort_values(['video_id','frame_index']).copy()
    test_ids={x for v in test.video_id for x in re.findall(r'id\d+',v)}
    target_ids={x for v in target.video_id for x in re.findall(r'id\d+',v)}
    if target_ids & test_ids:raise ValueError('Target/Test identity leakage')
    rows,files=[],{}
    for split,role,quotas in [('train','source_train',{'original':1500,'Deepfakes':750,'Face2Face':750}),
                             ('val','source_val',{'original':300,'Deepfakes':150,'Face2Face':150})]:
        for method,n in quotas.items():
            selected=sample(source[source.split.eq(split)&source.method.eq(method)],n,4,42)
            for r in selected.to_dict('records'):
                path=args.legacy_root/r['local_source_path']
                name=f"images/ffpp/{hashlib.sha256(r['sample_id'].encode()).hexdigest()[:24]}.png"
                rows.append(dict(item_id=name.removesuffix('.png'),crop_path=name,role=role,label=r['label'],
                    class_index=str(int(r['label']=='fake')),method=method,video_id='ffpp/'+method+'/'+r['video_id'],
                    source_ids=r['source_ids'],group_id=r['group_id'],original_split=split,
                    roi_status='source_aligned_crop',original_path=r['local_source_path']))
                files[name]=path
    for r in target.to_dict('records'):
        name=f"images/celebdf/{hashlib.sha256(r['crop_path'].encode()).hexdigest()[:24]}.png"
        rows.append(dict(item_id=name.removesuffix('.png'),crop_path=name,role='target',label='',class_index='',
            method='unlabelled_target',video_id='celebdf/'+r['video_id'],
            source_ids='|'.join(re.findall(r'id\d+',r['video_id'])),group_id='',original_split=r['provided_split'],
            roi_status=r['roi_status'],original_path=r['crop_path']))
        files[name]=args.celeb_root/r['crop_path']
    frame=pd.DataFrame(rows)
    for col in ['source_ids','group_id']:
        a={x for v in frame[frame.role.eq('source_train')][col] for x in v.split('|') if x}
        b={x for v in frame[frame.role.eq('source_val')][col] for x in v.split('|') if x}
        if a&b:raise ValueError('Source Train/Val leakage')
    if not all(x.is_file() for x in files.values()):raise FileNotFoundError('Missing selected source/crop images')
    for ix,r in test.iterrows():
        name=f"images/test/{hashlib.sha256(r['crop_path'].encode()).hexdigest()[:24]}.png"
        files[name]=args.celeb_root/r['crop_path']
        test.at[ix,'crop_path']=name
    cfg={'name':PREFIX,'seed':42,'counts':frame.groupby('role').size().to_dict(),
         'source_manifest_sha256':sha(source_path),'celeb_crop_manifest_sha256':sha(celeb_path),
         'source_split':'Existing frozen global_source_aware_v1 custom Train/Val, NOT official FF++ split.',
         'source':'Existing c23 dlib-aligned256 originals/Deepfakes/Face2Face; no Q95 re-encoding.',
         'target_labels_used':False,'target_ids':sorted({x for v in target.video_id for x in re.findall(r'id\d+',v)}),
         'identity_groups':groups,'target_identity_disjoint_from_custom_test':True,
         'target_sampling':'Random3000 from custom train47 IDs, max4 frames/video, no class-label selection',
         'test':'All1824 custom test frames, six filename IDs; NOT official test. Test images included in this TAR.',
         'test_images':len(test),'test_videos':int(test.video_id.nunique()),
         'limitations':['Only2 source forgery methods; not original detector training reproduction.',
                       'Source already aligned256; target native MTCNN crop differs.',
                       'Custom test only6 filename identities; previous evaluation exposure, not a new untouched final holdout.',
                       'Public original checkpoints may have seen some custom source validation samples; not an independent pretraining validation claim.']}
    args.output_root.mkdir(parents=True,exist_ok=True)
    tar_path=args.output_root/'public_detector_c23_custom_source_target_test_seed42_v2.tar'
    manifest_blob=frame.to_csv(index=False,lineterminator='\n').encode()
    test_blob=test.to_csv(index=False,lineterminator='\n').encode()
    cfg['manifest_sha256']=hashlib.sha256(manifest_blob).hexdigest()
    cfg['test_manifest_sha256']=hashlib.sha256(test_blob).hexdigest()
    if tar_path.exists():raise FileExistsError('Preserve existing pilot TAR; inspect before regeneration')
    temp=tar_path.with_suffix('.partial.tar')
    if temp.exists():raise FileExistsError(temp)
    with tarfile.open(temp,'w') as tar:
        for name,path in files.items():tar.add(path,arcname=PREFIX+'/'+name,recursive=False)
        for name,blob in [('manifest.csv',manifest_blob),('test_manifest.csv',test_blob),('protocol.json',json.dumps(cfg,indent=2).encode())]:
            info=tarfile.TarInfo(PREFIX+'/'+name);info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
    temp.replace(tar_path)
    inventory={'name':tar_path.name,'bytes':tar_path.stat().st_size,'sha256':sha(tar_path),'protocol':cfg}
    (args.output_root/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(inventory,indent=2),flush=True)


if __name__=='__main__':main()
