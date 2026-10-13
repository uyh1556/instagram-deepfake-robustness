"""c23 DF/F2F pilot: source FT vs smoothing augmentation vs two DANN controls."""
import argparse
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

import public_detector_adapters as adapters
import train_celebdf_baseline as baseline
from evaluate_celebdf_smoothing import key
from pilgram2_roi_pilot import Reverse
from smoothing_roi_pilot import smooth

ROOT=Path(__file__).resolve().parents[1]
STRATEGIES=['source_ft','source_aug','uda_normal','uda_smoothing']
CONDITIONS=['normal','smooth_weak','smooth_medium','smooth_strong']
FEATURE_DIMS={'recce':2048,'ucf':256,'selfi':768}
PRE_REPAIR_SHA='97858e9e26ef2203c7280acee6e69190eb77513db1c92de20e0bfe17b95b1dcd'


def run_configuration(args,name,strategy,cfg,public):
    current=configuration(args,name,strategy,cfg,public)
    path=args.runs_root/run_name(name,strategy)/'config.json'
    if path.is_file():
        saved=json.loads(path.read_text())
        if saved!=current:
            old=dict(saved);new=dict(current)
            old_sha=old.pop('code_sha256',None);new.pop('code_sha256',None)
            if old!=new or old_sha!=PRE_REPAIR_SHA:raise ValueError('Run config differs beyond approved numerical repair')
            # Preserve original config/checkpoint identities; record the repair separately.
            baseline.json_write(path.parent/'numerical_repair.json',{
                'original_code_sha256':old_sha,'repair_code_sha256':current['code_sha256'],
                'policy':'Retry nonfinite AMP loss on same batch in FP32; never skip a batch',
                'original_config_preserved':True})
            return saved
    return current


def batch_loss(name,model,discriminator,xs,xt,y,uda,grl,domain_weight):
    logits,features=outputs(name,model,torch.cat([xs,xt]))
    if uda:
        cls=nn.functional.cross_entropy(logits[:len(y)],y)
        domains=torch.cat([torch.zeros_like(y),torch.ones_like(y)])
        dom=nn.functional.cross_entropy(discriminator(Reverse.apply(features,grl)),domains)
        return cls+domain_weight*dom,cls,dom
    cls=nn.functional.cross_entropy(logits,torch.cat([y,y]))
    return cls,cls,cls.new_zeros(())


def inputs(args):
    cfg=json.loads((args.pilot_root/'protocol.json').read_text())
    path=args.pilot_root/'manifest.csv'
    if baseline.digest(path)!=cfg['manifest_sha256']:raise ValueError('Frozen pilot manifest mismatch')
    data=pd.read_csv(path,dtype=str,keep_default_na=False)
    target=data[data.role.eq('target')]
    if not target.label.eq('').all() or not target.class_index.eq('').all():raise ValueError('Target labels must be hidden')
    expected=cfg['counts']
    if not args.allow_tiny_fixture and data.groupby('role').size().to_dict()!=expected:raise ValueError('Incomplete pilot TAR')
    test_path=args.pilot_root/'test_manifest.csv'
    if baseline.digest(test_path)!=cfg['test_manifest_sha256']:raise ValueError('Frozen test subset mismatch')
    test=pd.read_csv(test_path,dtype=str,keep_default_na=False)
    public=json.loads((ROOT/'configs/public_detectors_smoothing_v1.json').read_text())
    return data,test,cfg,public


class PilotImages(Dataset):
    def __init__(self,data,args,transform,public,strategy=None,epoch=0,condition='normal',target=None,test=False):
        self.rows=data.to_dict('records');self.args=args;self.transform=transform
        self.public=public;self.strategy=strategy;self.epoch=epoch;self.condition=condition;self.test=test
        self.target=[] if target is None else target.to_dict('records')
    def __len__(self):return len(self.rows)
    def view(self,row,condition,flip=False):
        root=self.args.test_root if self.test else self.args.pilot_root
        with Image.open(root/row['crop_path']) as f:
            image=f.convert('RGB')
            if condition!='normal' and row['roi_status']!='fallback_letterbox':
                cfg=self.public['smoothing']
                image=smooth(image,cfg['conditions'][condition],cfg['bilateral'])
            if flip:image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return self.transform(image)
    def __getitem__(self,i):
        row=self.rows[i]
        if self.strategy is None:return self.view(row,self.condition),int(row['class_index']),i
        rng=random.Random(self.args.seed+self.epoch*100003+i)
        flip=rng.random()<.5
        source=self.view(row,'normal',flip)
        if self.strategy=='source_ft':second=source.clone()
        elif self.strategy=='source_aug':second=self.view(row,rng.choice(CONDITIONS[1:]),flip)
        else:
            target=rng.choice(self.target)
            assert target['class_index']=='' and target['label']==''
            condition='normal' if self.strategy=='uda_normal' else rng.choice(CONDITIONS[1:])
            second=self.view(target,condition,rng.random()<.5)
        return source,second,int(row['class_index'])


def outputs(name,model,images):
    if name=='recce':
        features=model.features(images)
        return model.classifier(features),features
    data={'image':images}
    if name=='ucf':data['label']=torch.zeros(len(images),dtype=torch.long,device=images.device)
    result=model(data,inference=True)
    return result['cls'],result['feat'] if name=='ucf' else result['fused_feat']


def freeze_bn(model):
    for layer in model.modules():
        if isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.eval()


def run_name(name,strategy):return f'{name}_{strategy}_c23_df_f2f_custom_v2_seed42'


def model_state(name,model):
    # ArcFace is frozen and stays in the original public checkpoint, saving per-run storage.
    return {k:v for k,v in model.state_dict().items() if not(name=='selfi' and k.startswith('iresnet.'))}


def restore(name,model,state):
    status=model.load_state_dict(state,strict=False)
    expected={k for k in model.state_dict() if name=='selfi' and k.startswith('iresnet.')}
    if set(status.missing_keys)!=expected or status.unexpected_keys:raise ValueError('Fine-tuned state mismatch')


def configuration(args,name,strategy,cfg,public):
    return {'name':cfg['name'],'model':name,'strategy':strategy,'data':cfg,
            'initial_checkpoint_sha256':public[name]['checkpoint_sha256'],
            'epochs':args.epochs,'batch_size':args.batch_size,'learning_rate':args.learning_rate,
            'weight_decay':args.weight_decay,'domain_weight':args.domain_weight,'seed':args.seed,
            'source_loss':'CE; source_ft repeated clean view, source_aug mean clean+random smoothing CE',
            'uda':'labelled clean source CE + .1 domain CE with progressive GRL; independent unlabelled target',
            'preprocessing':public[name],'smoothing':public['smoothing'],
            'batchnorm':'running statistics frozen in all strategies',
            'frozen_identity_encoder':'SELFI ArcFace frozen; original public weights reused',
            'checkpoint_selection':'maximum clean source validation frame AUC only',
            'limitations':'Classification fine-tuning, not original RECCE/UCF training losses. Custom test only six identities.',
            'code_sha256':baseline.digest(Path(__file__)),'adapter_sha256':baseline.digest(ROOT/'scripts/public_detector_adapters.py'),
            'torch':torch.__version__}


def train(args):
    data,_,cfg,public=inputs(args)
    source=data[data.role.eq('source_train')].reset_index(drop=True)
    val=data[data.role.eq('source_val')].reset_index(drop=True)
    target=data[data.role.eq('target')].reset_index(drop=True)
    for name in args.models:
        for strategy in args.strategies:
            config=run_configuration(args,name,strategy,cfg,public);identity=key(config)
            run=args.runs_root/run_name(name,strategy);run.mkdir(parents=True,exist_ok=True)
            if (run/'completed.json').exists():
                if json.loads((run/'completed.json').read_text())['identity']!=identity or not(run/'best.pt').is_file():raise ValueError('Completed run differs')
                print('Skip completed:',name,strategy,flush=True);continue
            if (run/'config.json').exists() and json.loads((run/'config.json').read_text())!=config:raise ValueError('Run config differs')
            baseline.json_write(run/'config.json',config)
            baseline.seed_all(args.seed)
            model,transform=adapters.load(name,args.asset_root/name,args.asset_root/f'{name}_public.pth')
            model.to(args.device)
            discriminator=nn.Sequential(nn.Linear(FEATURE_DIMS[name],128),nn.ReLU(),nn.Linear(128,2)).to(args.device)
            uda=strategy.startswith('uda_')
            parameters=list(p for p in model.parameters() if p.requires_grad)
            if uda:parameters+=list(discriminator.parameters())
            optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=args.weight_decay)
            scaler=torch.amp.GradScaler('cuda',enabled=args.device=='cuda')
            start,best,history=0,-1.,[]
            if (run/'last.pt').exists():
                state=torch.load(run/'last.pt',map_location=args.device,weights_only=True)
                if state['identity']!=identity:raise ValueError('Resume identity differs')
                restore(name,model,state['model_state_dict']);discriminator.load_state_dict(state['discriminator'])
                optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
                start,best,history=state['epoch'],state['best'],state['history']
                print('Resume completed epochs:',start,flush=True)
                del state
            valset=PilotImages(val,args,transform,public)
            print('Training:',name,strategy,'source',len(source),'val',len(val),'target',len(target) if uda else 0,flush=True)
            for epoch in range(start,args.epochs):
                epoch_seed=args.seed+epoch*100003;baseline.seed_all(epoch_seed)
                dataset=PilotImages(source,args,transform,public,strategy,epoch,target=target)
                loader=baseline.make_loader(dataset,args,True,epoch_seed)
                model.train();freeze_bn(model);discriminator.train()
                total,count=0.,0
                for step,(xs,xt,y) in enumerate(tqdm(loader,desc=f'{name} {strategy} epoch{epoch+1}/{args.epochs}',mininterval=2),1):
                    xs,xt,y=xs.to(args.device),xt.to(args.device),y.to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    progress=(epoch*len(loader)+step)/(args.epochs*len(loader))
                    grl=2/(1+math.exp(-10*progress))-1
                    cpu_rng=torch.get_rng_state()
                    cuda_rng=torch.cuda.get_rng_state() if args.device=='cuda' else None
                    with torch.autocast(device_type=args.device,enabled=args.device=='cuda'):
                        loss,class_loss,domain_loss=batch_loss(name,model,discriminator,xs,xt,y,uda,grl,args.domain_weight)
                    if not torch.isfinite(loss):
                        amp_values={'classification':float(class_loss.detach()),'domain':float(domain_loss.detach())}
                        del loss,class_loss,domain_loss
                        torch.set_rng_state(cpu_rng)
                        if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
                        with torch.autocast(device_type=args.device,enabled=False):
                            loss,class_loss,domain_loss=batch_loss(name,model,discriminator,xs.float(),xt.float(),y,uda,grl,args.domain_weight)
                        event={'epoch':epoch+1,'batch':step,'amp':amp_values,
                               'fp32_classification':float(class_loss.detach()),'fp32_domain':float(domain_loss.detach())}
                        with (run/'numerical_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                        if not torch.isfinite(loss):
                            raise RuntimeError(f'FP32 also nonfinite: {event}. Last completed epoch preserved; no batch skipped.')
                        print(f'FP32 retry succeeded: {name}/{strategy} epoch{epoch+1} batch{step}',flush=True)
                    scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(parameters,1.);scaler.step(optimizer);scaler.update()
                    total+=float(loss.detach())*len(y);count+=len(y)
                model.eval();scores=np.empty(len(val))
                with torch.inference_mode():
                    for x,_,ix in tqdm(baseline.make_loader(valset,args),desc='Clean source validation',mininterval=2):
                        scores[ix.numpy()]=adapters.probabilities(name,model,x.to(args.device)).cpu().numpy()
                values,_=baseline.metrics(val,scores);auc=values['frame']['auc']
                improved=auc>best;best=max(best,auc)
                history.append({'epoch':epoch+1,'train_loss':total/count,'val_frame':values['frame']})
                state={'identity':identity,'model_state_dict':model_state(name,model),'discriminator':discriminator.state_dict(),
                       'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,'best':best,'history':history}
                if improved:baseline.save(run/'best.pt',{k:state[k] for k in ['identity','model_state_dict','epoch','best']})
                baseline.save(run/'last.pt',state);baseline.json_write(run/'history.json',history)
                del state
                print('Epoch',epoch+1,'source val Frame AUC',round(auc,4),'best',round(best,4),flush=True)
            baseline.json_write(run/'completed.json',{'identity':identity,'best_source_val_frame_auc':best})
            del model,discriminator,optimizer,parameters,scaler
            if args.device=='cuda':torch.cuda.empty_cache()


def evaluate(args):
    _,test,cfg,public=inputs(args)
    args.output_root.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in args.models:
        for strategy in ['baseline',*args.strategies]:
            model,transform=adapters.load(name,args.asset_root/name,args.asset_root/f'{name}_public.pth')
            config=run_configuration(args,name,strategy,cfg,public)
            identity=key({'configuration':config,'test_manifest_sha256':cfg['test_manifest_sha256']})
            if strategy!='baseline':
                state=torch.load(args.runs_root/run_name(name,strategy)/'best.pt',map_location='cpu',weights_only=True)
                if state['identity']!=key(config):raise ValueError('Wrong trained checkpoint')
                restore(name,model,state['model_state_dict']);del state
            model.to(args.device).eval()
            for condition in CONDITIONS:
                prefix=args.output_root/f'{name}_{strategy}_{condition}'
                report,pred_path=prefix.with_suffix('.json'),prefix.with_suffix('.csv')
                if report.exists() and pred_path.exists():
                    saved=json.loads(report.read_text())
                    if saved['identity']!=identity:raise ValueError('Saved evaluation differs')
                    values=saved['frame'];print('Reuse completed:',name,strategy,condition,flush=True)
                else:
                    dataset=PilotImages(test,args,transform,public,condition=condition,test=True)
                    scores=np.empty(len(test))
                    with torch.inference_mode():
                        for x,_,ix in tqdm(baseline.make_loader(dataset,args),desc=f'{name} {strategy} {condition}',mininterval=2):
                            scores[ix.numpy()]=adapters.probabilities(name,model,x.to(args.device)).cpu().numpy()
                    result,predictions=baseline.metrics(test,scores);values=result['frame']
                    predictions.to_csv(pred_path,index=False);baseline.json_write(report,{'identity':identity,'frame':values})
                records.append({'model':name,'strategy':strategy,'condition':condition,**{k:values[k] for k in ['auc','accuracy','f1','real_fpr','fake_fnr']}})
            del model
            if args.device=='cuda':torch.cuda.empty_cache()
    table=pd.DataFrame(records);table.to_csv(args.output_root/'frame_summary.csv',index=False)
    baseline.json_write(args.output_root/'protocol.json',cfg)
    print(table.to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','evaluate'])
    for name in ['pilot-root','test-root','asset-root','runs-root','output-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=['recce','ucf','selfi'],default=['recce','ucf','selfi'])
    p.add_argument('--strategies',nargs='+',choices=STRATEGIES,default=STRATEGIES)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--domain-weight',type=float,default=.1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--allow-tiny-fixture',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.seed!=42:raise ValueError('This frozen pilot uses seed42 only')
    baseline.seed_all(args.seed)
    (train if args.action=='train' else evaluate)(args)


if __name__=='__main__':main()
