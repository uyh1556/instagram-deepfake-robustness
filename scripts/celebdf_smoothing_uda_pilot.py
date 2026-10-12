"""FF++ source -> unlabelled Celeb-DF: original vs smoothing DANN controls."""
import argparse, copy, hashlib, itertools, json, math, random, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
import pilgram2_roi_pilot as common
from smoothing_roi_pilot import smooth, PROTOCOL
from xception_preprocessing import interpolation_mode

STRATEGIES=['baseline','source_ft','source_aug','uda_normal','uda_smoothing']
CONDITIONS=['normal','smooth_weak','smooth_medium','smooth_strong']

def initial(args):
    model=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    protocol=json.loads((args.data_root/'protocol.json').read_text())
    assert model['config']['preprocessing_name']=='face_roi_mtcnn_1p5_299'
    assert model['num_classes']==2
    assert model['config']['manifest_sha256']==protocol['source_ffpp_manifest_sha256']
    return model,protocol
def path(root,row,condition):
    return root/'images'/condition/(row['item_id']+'.png')
def prepare(args):
    base,protocol=initial(args)
    frame=pd.read_csv(args.data_root/'manifest.csv',dtype=str,keep_default_na=False)
    assert frame.item_id.is_unique and frame.source_path.is_unique
    target=frame[frame.role=='target']; test=frame[frame.role=='test']
    assert not set(target.video_id)&set(test.video_id)
    assert target.label.eq('').all(), 'Target labels must be removed'
    smoothing=json.loads(PROTOCOL.read_text())
    cfg={'protocol':protocol,'data_config':base['config']['data_config'],
         'manifest_sha256':common.sha256_file(args.data_root/'manifest.csv'),
         'checkpoint_sha256':common.sha256_file(args.checkpoint),'smoothing':smoothing,
         'smoothing_code_sha256':common.sha256_file(Path(__file__).with_name('smoothing_roi_pilot.py')),
         'pilot_code_sha256':common.sha256_file(Path(__file__)),
         'opencv':cv2.__version__,'torch':torch.__version__,'seed':args.seed}
    cfg['identity']=common.key(cfg)
    args.cache_root.mkdir(parents=True,exist_ok=True)
    saved=args.cache_root/'config.json'
    if saved.exists(): assert json.loads(saved.read_text())==cfg, 'Cache identity mismatch'
    else: common.atomic_json(saved,cfg)
    resize=transforms.Resize([299,299],interpolation=interpolation_mode(cfg['data_config']['interpolation']),antialias=True)
    def one(row):
        conditions=['normal'] if row['role']=='source_val' else CONDITIONS
        missing=[c for c in conditions if not path(args.cache_root,row,c).is_file()]
        if not missing:return
        with Image.open(args.data_root/row['source_path']) as opened:
            original=opened.convert('RGB')
            for condition in missing:
                image=original
                if condition!='normal' and row['roi_status']=='detected':
                    image=smooth(original,smoothing['strengths'][condition],smoothing['bilateral'])
                image=resize(image)
                dest=path(args.cache_root,row,condition); dest.parent.mkdir(parents=True,exist_ok=True)
                temp=dest.with_suffix('.tmp.png'); image.save(temp,format='PNG'); temp.replace(dest)
    last=time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        for i,_ in enumerate(pool.map(one,frame.to_dict('records')),1):
            if time.monotonic()-last>30 or i==len(frame):
                print(f'Cache {i}/{len(frame)} native crop -> smoothing -> resize299',flush=True);last=time.monotonic()
    common.atomic_json(args.cache_root/'ready.json',{'identity':cfg['identity']})
def cache(args):
    base,protocol=initial(args)
    cfg=json.loads((args.cache_root/'config.json').read_text())
    assert cfg['protocol']==protocol and cfg['seed']==args.seed
    assert cfg['manifest_sha256']==common.sha256_file(args.data_root/'manifest.csv')
    assert cfg['checkpoint_sha256']==common.sha256_file(args.checkpoint)
    assert cfg['pilot_code_sha256']==common.sha256_file(Path(__file__))
    assert json.loads((args.cache_root/'ready.json').read_text())['identity']==cfg['identity']
    frame=pd.read_csv(args.data_root/'manifest.csv',dtype=str,keep_default_na=False)
    return base,cfg,frame

class Images(Dataset):
    def __init__(self,frame,args,cfg,condition='normal'):
        self.rows=frame.to_dict('records'); self.args=args; self.condition=condition
        self.tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize(cfg['data_config']['mean'],cfg['data_config']['std'])])
    def __len__(self): return len(self.rows)
    def view(self,row,condition,flip=False):
        with Image.open(path(self.args.cache_root,row,condition)) as opened:
            image=opened.convert('RGB')
            if flip:image=TF.hflip(image)
            return self.tensor(image)
    def __getitem__(self,i):
        row=self.rows[i]
        assert row['label'] in ['real','fake']
        return self.view(row,self.condition),int(row['label']=='fake')
class Train(Images):
    def __init__(self,source,target,args,cfg,strategy,epoch):
        super().__init__(source,args,cfg); self.target=target.to_dict('records'); self.strategy=strategy; self.epoch=epoch
        assert all(r['label']=='' for r in self.target)
    def __getitem__(self,i):
        source=self.rows[i]; rng=random.Random(self.args.seed+self.epoch*1000003+i)
        flip=rng.random()<.5
        xs=self.view(source,'normal',flip)
        if self.strategy=='source_ft':xt=xs.clone()
        elif self.strategy=='source_aug':xt=self.view(source,rng.choice(CONDITIONS[1:]),flip)
        else:
            # Independent target images. Their class labels are never returned.
            row=rng.choice(self.target)
            condition='normal' if self.strategy=='uda_normal' else rng.choice(CONDITIONS[1:])
            xt=self.view(row,condition,rng.random()<.5)
        return xs,xt,int(source['label']=='fake')

def train(args):
    base,cfg,frame=cache(args); source=frame[frame.role=='source_train']; target=frame[frame.role=='target']
    validation=Images(frame[frame.role=='source_val'],args,cfg)
    val_loader=common.loader(validation,args)
    weights=base['config']['class_weights']
    criterion=nn.CrossEntropyLoss(weight=torch.tensor([weights['real'],weights['fake']],device=args.device))
    for strategy in args.strategies:
        if strategy=='baseline':continue
        identity=common.key({'cache':cfg['identity'],'strategy':strategy,'epochs':args.epochs,'lr':args.learning_rate,
                             'batch_size':args.batch_size,'weight_decay':args.weight_decay,'domain_weight':args.domain_weight})
        folder=args.runs_root/strategy; folder.mkdir(parents=True,exist_ok=True)
        if (folder/'completed.json').exists():
            assert json.loads((folder/'completed.json').read_text())['identity']==identity and (folder/'best.pt').is_file()
            print('Skip completed:',strategy,flush=True);continue
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        if args.device=='cuda':torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        model=common.model_from(base).to(args.device)
        disc=nn.Sequential(nn.Linear(model.num_features,128),nn.ReLU(),nn.Linear(128,1)).to(args.device)
        is_uda=strategy.startswith('uda_')
        optimizer=torch.optim.AdamW(itertools.chain(model.parameters(),disc.parameters() if is_uda else []),lr=args.learning_rate,weight_decay=args.weight_decay)
        scaler=torch.amp.GradScaler('cuda',enabled=args.device=='cuda')
        start,best,history=0,-1.,[]
        if (folder/'last.pt').exists():
            state=torch.load(folder/'last.pt',map_location=args.device,weights_only=False)
            assert state['pilot_identity']==identity
            model.load_state_dict(state['model_state_dict']);disc.load_state_dict(state['discriminator_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict']);scaler.load_state_dict(state['scaler_state_dict'])
            start,best,history=state['epoch'],state['best_val_auc'],state['history']
        run_config=copy.deepcopy(base['config'])
        run_config.update(experiment_family='celebdf_v2_smoothing_uda_pilot_v1',condition_name=strategy,
                          pilot_identity=identity,cache_identity=cfg['identity'],target_labels_used=False,
                          train_images=len(source),target_images=len(target) if is_uda else 0,val_images=len(validation),
                          epochs=args.epochs,learning_rate=args.learning_rate,weight_decay=args.weight_decay,
                          batch_size=args.batch_size,seed=args.seed,domain_weight=args.domain_weight,
                          initial_checkpoint_sha256=cfg['checkpoint_sha256'],protocol=cfg['protocol'],
                          preprocessing='native frozen crop -> smoothing when requested -> bicubic299 -> normalize',
                          target_sampling='independent uniform sampling with replacement; labels hidden',
                          validation_rule='maximum FF++ original validation AUC only')
        common.atomic_json(folder/'config.json',run_config)
        for epoch in range(start,args.epochs):
            torch.manual_seed(args.seed+epoch)
            if args.device=='cuda':torch.cuda.manual_seed_all(args.seed+epoch)
            loader=common.loader(Train(source,target,args,cfg,strategy,epoch),args,True,epoch)
            model.train();disc.train();sums=np.zeros(3);done=0;last=time.monotonic()
            print(f'{strategy}: epoch {epoch+1}/{args.epochs}',flush=True)
            for step,(xs,xt,y) in enumerate(loader):
                xs,xt,y=xs.to(args.device),xt.to(args.device),y.to(args.device); n=len(y)
                optimizer.zero_grad(set_to_none=True)
                progress=(epoch*len(loader)+step)/max(1,args.epochs*len(loader)-1)
                alpha=2/(1+math.exp(-10*progress))-1
                with torch.autocast(device_type=args.device,dtype=torch.float16,enabled=args.device=='cuda'):
                    features=model.forward_features(torch.cat([xs,xt])); logits=model.forward_head(features)
                    cls=criterion(logits[:n],y)
                    if strategy=='source_aug':cls=.5*(cls+criterion(logits[n:],y))
                    domain=cls.new_zeros(())
                    if is_uda:
                        score=disc(common.Reverse.apply(features.mean((2,3)),alpha)).flatten()
                        domain=nn.functional.binary_cross_entropy_with_logits(score,torch.cat([torch.zeros(n,device=args.device),torch.ones(n,device=args.device)]))
                    loss=cls+args.domain_weight*domain
                scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                sums+=np.array([loss.item(),cls.item(),domain.item()])*n;done+=n
                if time.monotonic()-last>=30:
                    print(f'Train {done}/{len(source)} loss={sums[0]/done:.4f}',flush=True);last=time.monotonic()
            y,p=common.scores(model,val_loader,args,'FF++ validation original');auc=common.roc_auc_score(y,p)
            history.append({'epoch':epoch+1,'loss':sums[0]/done,'classification_loss':sums[1]/done,'domain_loss':sums[2]/done,'source_validation_auc':float(auc)})
            improved=auc>best;best=max(best,auc)
            state={'model_name':base['model_name'],'num_classes':2,'model_state_dict':model.state_dict(),'config':run_config,
                   'epoch':epoch+1,'best_val_auc':best,'pilot_identity':identity}
            if improved:common.atomic_checkpoint(folder/'best.pt',state)
            common.atomic_checkpoint(folder/'last.pt',dict(state,discriminator_state_dict=disc.state_dict(),optimizer_state_dict=optimizer.state_dict(),scaler_state_dict=scaler.state_dict(),history=history))
            common.atomic_json(folder/'history.json',history)
            print(f'FF++ val AUC={auc:.6f}; best={best:.6f}',flush=True)
        common.atomic_json(folder/'completed.json',{'identity':identity,'epochs':args.epochs})
        del model,disc,optimizer
        if args.device=='cuda':torch.cuda.empty_cache()

def measures(y,p):
    m=common.classification_metrics(y,(p>=.5).astype(int),p)
    m['real_fpr']=float((p[y==0]>=.5).mean());m['fake_fnr']=float((p[y==1]<.5).mean())
    return {k:v for k,v in m.items() if k!='confusion_matrix'}
def evaluate(args):
    base,cfg,frame=cache(args);test=frame[frame.role=='test'].reset_index(drop=True)
    table=[];args.output_root.mkdir(parents=True,exist_ok=True)
    for strategy in args.strategies:
        checkpoint=args.checkpoint if strategy=='baseline' else args.runs_root/strategy/'best.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if strategy!='baseline':assert state['config']['cache_identity']==cfg['identity'] and state['config']['condition_name']==strategy
        model=common.model_from(state).to(args.device);checkpoint_sha=common.sha256_file(checkpoint)
        for condition in CONDITIONS:
            folder=args.output_root/strategy/condition;folder.mkdir(parents=True,exist_ok=True)
            identity=common.key({'cache':cfg['identity'],'checkpoint':checkpoint_sha,'condition':condition})
            saved=folder/'metrics.json'
            if saved.exists():
                data=json.loads(saved.read_text());assert data['identity']==identity and (folder/'predictions.csv').exists()
                result=data['result'];print('Skip evaluation:',strategy,condition,flush=True)
            else:
                print(f'Evaluate {strategy}/{condition}: {len(test)} images',flush=True)
                y,p=common.scores(model,common.loader(Images(test,args,cfg,condition),args),args,f'{strategy}/{condition}')
                predictions=test[['item_id','video_id','method','label','roi_status']].copy()
                predictions['class_index']=y;predictions['fake_probability']=p;predictions['condition']=condition
                predictions.to_csv(folder/'predictions.csv',index=False)
                video=predictions.groupby('video_id',sort=True).agg(class_index=('class_index','first'),fake_probability=('fake_probability','mean'))
                assert predictions.groupby('video_id').class_index.nunique().max()==1
                result={'images':len(test),'videos':len(video),**measures(y,p),
                        **{'video_'+k:v for k,v in measures(video.class_index.to_numpy(),video.fake_probability.to_numpy()).items()}}
                common.atomic_json(saved,{'identity':identity,'threshold':.5,'result':result})
            table.append(dict(strategy=strategy,condition=condition,**result))
            print(f"Image AUC={result['roc_auc']:.6f} | Video AUC={result['video_roc_auc']:.6f} | FPR={result['real_fpr']:.2%} | FNR={result['fake_fnr']:.2%}",flush=True)
        del model
        if args.device=='cuda':torch.cuda.empty_cache()
    df=pd.DataFrame(table);df.to_csv(args.output_root/'condition_summary.csv',index=False)
    if 'baseline' in set(df.strategy):
        reference=df[df.strategy=='baseline'][['condition','roc_auc','video_roc_auc','real_fpr','fake_fnr']]
        delta=df.merge(reference,on='condition',suffixes=('','_baseline'))
        for name in ['roc_auc','video_roc_auc','real_fpr','fake_fnr']:delta[name+'_delta']=delta[name]-delta[name+'_baseline']
        delta.to_csv(args.output_root/'baseline_deltas.csv',index=False)
    print(df[['strategy','condition','roc_auc','video_roc_auc','real_fpr','fake_fnr']].to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','evaluate'])
    for name in ['data-root','checkpoint','cache-root','runs-root','output-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--strategies',nargs='+',choices=STRATEGIES,default=STRATEGIES)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=1e-5);p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--domain-weight',type=float,default=.1);p.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args(); assert args.workers>=0 and args.epochs>0 and args.batch_size>0
    {'prepare':prepare,'train':train,'evaluate':evaluate}[args.action](args)
if __name__=='__main__':main()
