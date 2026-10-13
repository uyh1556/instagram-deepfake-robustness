"""Frozen public RECCE/SELFI -> official Celeb-DF test original/three smoothing views."""
import argparse
import json
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import public_detector_adapters as adapters
import train_celebdf_baseline as baseline
from evaluate_celebdf_smoothing import Views, cached, key, paired_bootstrap
from smoothing_roi_pilot import smooth

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/public_detectors_smoothing_v1.json'


def setup(args):
    cfg=json.loads(PROTOCOL.read_text())
    args.asset_root.mkdir(parents=True,exist_ok=True)
    for name in ['recce','ucf','selfi']:
        spec=cfg[name]
        repo=args.asset_root/name
        if not (repo/'.git').is_dir():
            if repo.exists() and any(repo.iterdir()):
                raise RuntimeError(f'Existing nonrepository assets: {repo}')
            repo.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','init','-q',str(repo)],check=True)
            subprocess.run(['git','-C',str(repo),'remote','add','origin',spec['repo']],check=True)
            subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin',spec['commit']],check=True)
            subprocess.run(['git','-C',str(repo),'checkout','--detach','FETCH_HEAD'],check=True)
        commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if commit!=spec['commit'] or subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
            raise RuntimeError(f'Upstream source changed: {repo}')
        checkpoint=args.asset_root/f'{name}_public.pth'
        if not checkpoint.is_file():
            partial=checkpoint.with_suffix('.partial.pth')
            print('Downloading official checkpoint:',name,flush=True)
            if name in ['recce','ucf']:
                with urllib.request.urlopen(spec['checkpoint_url']) as response, partial.open('wb') as out:
                    total=int(response.headers.get('Content-Length',spec['checkpoint_bytes']))
                    with tqdm(total=total,unit='B',unit_scale=True,desc=name) as bar:
                        for block in iter(lambda:response.read(4*1024*1024),b''):
                            out.write(block);bar.update(len(block))
            else:
                import gdown
                if gdown.download(id=spec['checkpoint_drive_id'],output=str(partial),quiet=False) is None:
                    raise RuntimeError('SELFI checkpoint download failed; do not substitute pretrained CLIP')
            if partial.stat().st_size!=spec['checkpoint_bytes'] or baseline.digest(partial)!=spec['checkpoint_sha256']:
                raise RuntimeError('Downloaded public checkpoint mismatch; preserve partial and inspect')
            partial.replace(checkpoint)
        if checkpoint.stat().st_size!=spec['checkpoint_bytes'] or baseline.digest(checkpoint)!=spec['checkpoint_sha256']:
            raise RuntimeError(f'Checkpoint identity differs: {checkpoint}')
        print('Verified public weights/source:',name,flush=True)


def frame(args):
    path=args.data_root/'manifests/celebdf_v2_roi16_test_v1.csv'
    data=pd.read_csv(path,dtype=str,keep_default_na=False)
    if len(data)!=8288 or data.video_id.nunique()!=518 or not data.split.eq('test').all():
        raise ValueError('Expected complete official test TAR manifest8288/518, not custom114-video test')
    if data.crop_path.duplicated().any() or data.groupby('video_id').class_index.nunique().gt(1).any():
        raise ValueError('Invalid official test manifest')
    return data, path


def cache_identity(args):
    data,path=frame(args)
    cfg=json.loads(PROTOCOL.read_text())
    identity={'protocol':cfg,'manifest_sha256':baseline.digest(path),'opencv':cv2.__version__,
              'smoothing_code_sha256':baseline.digest(ROOT/'scripts/smoothing_roi_pilot.py'),
              'evaluator_sha256':baseline.digest(Path(__file__)),
              'adapter_sha256':baseline.digest(ROOT/'scripts/public_detector_adapters.py')}
    return data,cfg,identity


def prepare(args):
    data,cfg,identity=cache_identity(args)
    args.cache_root.mkdir(parents=True,exist_ok=True)
    config=args.cache_root/'config.json'
    if config.exists() and json.loads(config.read_text())!=identity:
        raise RuntimeError('Old cache protocol differs; use new cache root')
    baseline.json_write(config,identity)
    def one(row):
        source=args.data_root/row['crop_path']
        if not source.is_file():raise FileNotFoundError(source)
        if row['roi_status']!='detected':return
        missing=[c for c in cfg['smoothing']['conditions'] if c!='normal' and not cached(args,c,row).is_file()]
        if not missing:return
        with Image.open(source) as opened:
            original=opened.convert('RGB')
            for condition in missing:
                dest=cached(args,condition,row);dest.parent.mkdir(parents=True,exist_ok=True)
                temp=dest.with_suffix('.tmp.png')
                smooth(original,cfg['smoothing']['conditions'][condition],cfg['smoothing']['bilateral']).save(temp,format='PNG')
                temp.replace(dest)
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        for _ in tqdm(pool.map(one,data.to_dict('records')),total=len(data),desc='Native smoothing cache',mininterval=2):pass
    baseline.json_write(args.cache_root/'ready.json',{'identity':key(identity)})
    print('Ready official test:',len(data),'images,',data.video_id.nunique(),'videos; fallback unchanged:',int(data.roi_status.ne('detected').sum()),flush=True)


def evaluate(args):
    data,cfg,identity=cache_identity(args)
    if json.loads((args.cache_root/'ready.json').read_text())['identity']!=key(identity):
        raise RuntimeError('Matching smoothing cache required')
    args.output_root.mkdir(parents=True,exist_ok=True)
    records,deltas=[],[]
    for name in args.models:
        checkpoint=args.asset_root/f'{name}_public.pth'
        if baseline.digest(checkpoint)!=cfg[name]['checkpoint_sha256']:raise RuntimeError('Wrong public weights')
        model,transform=adapters.load(name,args.asset_root/name,checkpoint)
        model.to(args.device).eval()
        clean,clean_row=None,None
        run_identity=key({'cache':identity,'model':name,'torch':torch.__version__,'bootstrap':args.bootstrap_repeats})
        for condition in cfg['smoothing']['conditions']:
            dest=args.output_root/f'{name}_{condition}'
            report_path,pred_path=dest.with_suffix('.json'),dest.with_suffix('.csv')
            if report_path.exists() and pred_path.exists():
                report=json.loads(report_path.read_text())
                if report['identity']!=run_identity:raise RuntimeError('Saved result belongs to another protocol')
                metrics=report['metrics'];predictions=pd.read_csv(pred_path,dtype={'video_id':str,'crop_path':str,'roi_status':str})
                if predictions.crop_path.tolist()!=data.crop_path.tolist():raise RuntimeError('Prediction/test mismatch')
                print('Reuse completed:',name,condition,flush=True)
            else:
                loader=baseline.make_loader(Views(data,args,transform,condition),args)
                scores=np.empty(len(data))
                with torch.inference_mode():
                    for x,_,indexes in tqdm(loader,desc=f'{name} {condition}',mininterval=2):
                        # Full float32 inference: match public code, no training/no AMP.
                        scores[indexes.numpy()]=adapters.probabilities(name,model,x.to(args.device)).cpu().numpy()
                metrics,predictions=baseline.metrics(data,scores)
                predictions.to_csv(pred_path,index=False)
                baseline.json_write(report_path,{'identity':run_identity,'metrics':metrics})
            row={'model':name,'condition':condition,'images':len(data),'videos':data.video_id.nunique(),
                 'fallback_images':int(data.roi_status.ne('detected').sum())}
            row.update({f'{level}_{m}':metrics[level][m] for level in ['frame','video']
                        for m in ['auc','accuracy','f1','real_fpr','fake_fnr']})
            records.append(row)
            if condition=='normal':clean,clean_row=predictions,row
            else:
                delta={'model':name,'condition':condition}
                delta.update({f'delta_{level}_{m}':row[f'{level}_{m}']-clean_row[f'{level}_{m}']
                              for level in ['frame','video'] for m in ['auc','real_fpr','fake_fnr']})
                delta.update(paired_bootstrap(clean,predictions,args.bootstrap_repeats,args.seed))
                deltas.append(delta)
            print(f"{name} {condition}: Video AUC={row['video_auc']:.4f}, Real FPR={row['video_real_fpr']:.2%}, Fake FNR={row['video_fake_fnr']:.2%}",flush=True)
        del model
        if args.device=='cuda':torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(args.output_root/'condition_summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(args.output_root/'paired_deltas.csv',index=False)
    baseline.json_write(args.output_root/'protocol.json',identity)
    print('Saved:',args.output_root,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['setup','prepare','evaluate'])
    p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--models',nargs='+',choices=['recce','ucf','selfi'],default=['recce','ucf','selfi'])
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--bootstrap-repeats',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    if args.action!='setup' and not all([args.data_root,args.cache_root,args.output_root]):p.error('data/cache/output roots required')
    baseline.seed_all(args.seed)
    {'setup':setup,'prepare':prepare,'evaluate':evaluate}[args.action](args)


if __name__=='__main__':main()
