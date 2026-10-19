"""Source-supervised ranking with bounded source/target smoothing feature deltas.

Experimental label-preservation hypothesis, not a proven nuisance decomposition.
Target labels, logits and pseudo-label confidence never supervise this method.
"""
import argparse
import copy
import json
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import ffpp_c23_smoothing_mean_teacher as engine
import ffpp_c23_smoothing_mean_teacher_soft as reference
from ffpp_c23_smoothing_ranking import balanced_batches

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_delta_transfer_v1'
STRATEGIES = ['ce_rank', 'source_delta', 'target_delta']
common, base, adaptation, evaluation = engine.common, engine.base, engine.adaptation, engine.evaluation
CONDITIONS = engine.CONDITIONS
ORIGINAL_HASHES = engine.code_hashes


def code_hashes():
    return {**ORIGINAL_HASHES(), Path(__file__).name: base.common.digest(Path(__file__)),
            'ffpp_c23_smoothing_ranking.py': base.common.digest(ROOT/'scripts/ffpp_c23_smoothing_ranking.py')}


def run_name(name, strategy, seed):
    return f'{name}_{strategy}_{PROTOCOL}_seed{seed}'


@contextmanager
def protocol_context():
    old_protocol, old_hashes = engine.PROTOCOL, engine.code_hashes
    engine.PROTOCOL, engine.code_hashes = PROTOCOL, code_hashes
    try:
        yield
    finally:
        engine.PROTOCOL, engine.code_hashes = old_protocol, old_hashes


def bounded_delta(features, delta, ratio):
    """No amplification: cap donor delta by ratio * each recipient feature norm.

    Stop gradients through donor and cap. Only the recipient/classifier learn.
    Shape is [source images, views, dimensions] and [source images, dimensions].
    """
    if features.ndim != 3 or delta.shape != (len(features), features.shape[-1]):
        raise ValueError('Expected paired source features and one donor delta per image')
    delta = delta.detach().float()[:, None, :]
    limit = features.detach().float().norm(dim=-1, keepdim=True)*ratio
    factor = (limit/delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
    return delta*factor


def classification_and_ranking(logits, labels, weight):
    """All real/fake, cross-image and cross-view pairs, with actual source GT.

    Pairwise softplus(real margin - fake margin) is an AUC surrogate, not AUC.
    Constant shifts cancel in rank loss; CE also trains the decision threshold.
    """
    n, views, classes = logits.shape
    if classes != 2 or labels.shape != (n,) or not labels.eq(0).any() or not labels.eq(1).any():
        raise ValueError('Ranking requires actual source real AND fake in every batch')
    flat = logits.float().reshape(-1, 2)
    ce = nn.functional.cross_entropy(flat, labels[:, None].expand(-1, views).reshape(-1))
    margins = logits.float()[..., 1]-logits.float()[..., 0]
    real, fake = margins[labels == 0].flatten(), margins[labels == 1].flatten()
    gaps = real[:, None]-fake[None, :]
    rank = nn.functional.softplus(gaps).mean()
    return ce+weight*rank, ce, rank, (gaps.detach() >= 0).float().mean()


class DonorViews(Dataset):
    """Observed normal/weak/medium/strong + weak re-smoothing, SAME flip.

    Dataset consumes only crop_path/video_id/roi_status/condition/flip. Neither
    source donor GT nor target GT is returned. Fallback contributes a zero delta.
    """
    def __init__(self, plan, root, transform, cfg, observed_cache, probe_cache):
        self.plan, self.root, self.transform, self.cfg = plan, root, transform, cfg
        self.observed_cache, self.probe_cache = observed_cache, probe_cache

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        row = self.plan[index]
        with Image.open(self.root/row['crop_path']) as opened:
            image = opened.convert('RGB')
        detected = row['roi_status'] == 'detected'
        observed = image
        if detected and row['condition'] != 'normal':
            observed = self.observed_cache.smooth(image, row['crop_path'], self.cfg['conditions'][row['condition']])
        probe = (self.probe_cache.smooth(observed, str(Path(row['condition'])/row['crop_path']),
                  self.cfg['conditions']['smooth_weak']) if detected else observed.copy())
        if row['flip']:
            observed = observed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            probe = probe.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.transform(observed), self.transform(probe), detected


def source_donor_plan(sampled, seed):
    # Equal source classes in the fixed epoch sample; never consume their labels.
    rng = random.Random(seed+810001)
    records = sampled[['crop_path','video_id','roi_status']].to_dict('records')
    rng.shuffle(records)
    return [{**row, 'condition': rng.choice(CONDITIONS), 'flip': rng.random() < .5} for row in records]


def configuration(args, name, strategy, data, cfg, initial, initial_hash):
    return dict(protocol=PROTOCOL, model=name, strategy=strategy, seed=args.seed,
        data=data, smoothing=cfg, initial_checkpoint_sha256=initial_hash,
        preprocessing=base.PREPROCESS[name], train_blocks=initial['config']['train_blocks'],
        epochs=args.epochs, patience=args.patience, min_delta=.0001,
        source_images=args.source_images, val_images=args.val_images, target_images=args.target_images,
        batch_size=args.batch_size, encoder_lr=args.encoder_lr, head_lr=args.head_lr,
        weight_decay=args.weight_decay, ranking_weight=args.ranking_weight,
        delta_cap_ratio=args.delta_cap_ratio if strategy != 'ce_rank' else 0.,
        ema_decay=args.ema_decay, ramp_epochs=args.ramp_epochs,
        donor_domain={'ce_rank':'none','source_delta':'FF++ train','target_delta':'CDF non-test'}[strategy],
        donor_observation='normal/weak/medium/strong uniformly; observed -> additional weak smoothing, identical flip',
        delta='stopgrad(EMA_feature(probe)-EMA_feature(observed)); no target logits or pseudo labels',
        transfer='add positive donor delta to clean AND smooth source features; cap each at ratio*recipient norm, never amplify',
        label_preservation='experimental assumption; deltas may also change forgery evidence',
        donor_limit='source/target donors differ in content and unknown class priors, not solely smoothing style',
        loss='GT source mean CE + ranking_weight*mean softplus(real_margin-fake_margin), all cross-view pairs',
        source_views='clean and random weak/medium/strong; delta variants also classify both transferred features',
        source_sampling='same rotating 6000-image balanced epoch; per-batch Real:Fake 1:1, pairs kept together',
        donor_sampling='source: shuffled same source epoch; target: frozen GT-free video-uniform replacement, 6000 exposures',
        selection='maximum FF++ four-condition mean raw-margin AUC; exact AUC tie: minimum mean BER; epoch0 eligible; no target/test selection',
        inference_model='student', bn_statistics='frozen',
        precision='BF16 if supported else FP32 source training; FP32 EMA donor/validation/test',
        torch=str(torch.__version__), baseline_versions=initial['config'].get('versions',{}),
        versions={p:engine.metadata.version(p) for p in ['torch','torchvision','Pillow','numpy','pandas','scikit-learn']},
        code_sha256=code_hashes(), hard_reference_manifest=args._hard_reference_identity)


def is_better(values, best):
    return values['mean_auc'] > best['auc'] or (values['mean_auc'] == best['auc'] and values['mean_ber'] < best['ber'])


def train(args):
    reference.verify_membership(args)
    pool, val, source_data = common.shared.source_inputs(args)
    _, _, target, cfg, target_data, blocked = adaptation.dataset_inputs(args)
    target = target[['crop_path','video_id','roi_status']]
    data = {**source_data, **target_data}
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    engine.freeze_json(args.manifest_root/'protocol.json', dict(protocol=PROTOCOL, data=data,
        seed=args.seed, target_labels_used=False, excluded_video_ids=blocked,
        source_images=args.source_images, target_images=args.target_images))
    engine.freeze_csv(args.manifest_root/'source_validation.csv', val)
    engine.freeze_csv(args.manifest_root/'target_unlabelled.csv', target)
    common.prepare_pixel_cache(args, cfg, source_data)
    common.prepare_validation(args, val, cfg, source_data)
    # Existing native caches have pixel-only identities; retain the original hashes.
    identity = dict(target_paths_sha256=data['target_paths_sha256'], smoothing=cfg,
        opencv=evaluation.cv2.__version__, code_sha256=ORIGINAL_HASHES())
    target_cache = common.pixel_cache.BilateralCache(args.target_train_cache/'observed', cfg['bilateral'], identity)
    target_probe = common.pixel_cache.BilateralCache(args.target_train_cache/'extra_weak', cfg['bilateral'], identity)
    source_probe = common.pixel_cache.BilateralCache(args.target_train_cache/'source_extra_weak_delta_v1', cfg['bilateral'],
        dict(source_trainval_sha256=data['source_trainval_sha256'], smoothing=cfg,
             opencv=evaluation.cv2.__version__, pixel_code=base.common.digest(Path(common.pixel_cache.__file__))))
    print(f'Source={args.source_images}/epoch; val={len(val)} x4; best=mean AUC; no target class labels', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root,seed=args.seed),name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256'] or
                initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Baseline/source manifest mismatch')
        for strategy in args.strategies:
            config = configuration(args,name,strategy,data,cfg,initial,initial_hash)
            identity = evaluation.key(config)
            run = args.runs_root/run_name(name,strategy,args.seed)
            run.mkdir(parents=True,exist_ok=True)
            engine.freeze_json(run/'config.json',config)
            if (run/'completed.json').is_file():
                done=json.loads((run/'completed.json').read_text())
                if done['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError('Completed run mismatch')
                print('Already completed:',name,strategy,flush=True)
                continue
            base.common.seed_all(args.seed)
            model,transform=base.build_model(name,pretrained=False)
            model.load_state_dict(initial['model_state_dict'],strict=True)
            model.to(args.device)
            blocks=2 if name=='xception' else initial['config']['train_blocks']
            common.training_mode(model,name,blocks)
            ema=None if strategy=='ce_rank' else copy.deepcopy(model).requires_grad_(False).eval()
            head=base.head(model,name)
            head_ids={id(p) for p in head.parameters()}
            groups=[{'params':[p for p in model.parameters() if p.requires_grad and id(p) not in head_ids],'lr':args.encoder_lr},
                    {'params':list(head.parameters()),'lr':args.head_lr}]
            optimizer=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
            parameters=[p for g in groups for p in g['params']]
            start,stale,history=0,0,[]
            if (run/'last.pt').is_file():
                state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
                if state['identity']!=identity:
                    raise ValueError('Resume identity differs')
                model.load_state_dict(state['model_state_dict'],strict=True)
                if ema is not None:
                    ema.load_state_dict(state['ema_state_dict'],strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start,stale,history,best,anchor=[state[k] for k in ['epoch','stale','history','best','anchor']]
                del state
                print('Resume completed epoch:',start,flush=True)
            else:
                values=engine.validate(args,model,val,transform)
                best={'auc':values['mean_auc'],'ber':values['mean_ber']}
                anchor=best['auc']
                base.common.json_write(run/'epoch0_validation.json',values)
                base.common.save(run/'best.pt',dict(identity=identity,config=config,model_state_dict=model.state_dict(),
                    epoch=0,best=best,validation=values))
            for epoch in range(start,args.epochs):
                if stale>=args.patience:
                    break
                seed=args.seed+epoch*100003
                base.common.seed_all(seed)
                common.training_mode(model,name,blocks)
                sampled=adaptation.balanced_source(pool,args.source_images,seed)
                batches=balanced_batches(torch.tensor(sampled.class_index.astype(int).to_numpy()),args.batch_size,seed)
                source_loader=DataLoader(common.PairedSource(sampled,args,transform,cfg,epoch),batch_sampler=batches,
                    num_workers=args.workers,pin_memory=args.device=='cuda',persistent_workers=False,
                    generator=torch.Generator().manual_seed(seed))
                engine.freeze_csv(args.manifest_root/f'source_epoch{epoch+1:02d}.csv',sampled)
                plan,donors=None,None
                if ema is not None:
                    if strategy=='source_delta':
                        plan=source_donor_plan(sampled,seed)
                        donor_ds=DonorViews(plan,args.source_root,transform,cfg,args._pixel_cache,source_probe)
                    else:
                        plan=engine.target_plan(target,len(sampled),seed)
                        donor_ds=DonorViews(plan,args.target_root,transform,cfg,target_cache,target_probe)
                    engine.freeze_csv(args.manifest_root/strategy/f'donor_exposures_epoch{epoch+1:02d}.csv',pd.DataFrame(plan))
                    donors=iter(base.common.make_loader(donor_ds,args,False,seed))
                totals=np.zeros(7);count=0
                for step,(clean,smooth,labels,_,_) in enumerate(tqdm(source_loader,
                        desc=f'{name} {strategy} {epoch+1}/{args.epochs}',mininterval=15,dynamic_ncols=True)):
                    base.common.seed_all(seed+step)
                    clean,smooth,labels=clean.to(args.device),smooth.to(args.device),labels.to(args.device)
                    n=len(labels)
                    delta=None
                    if donors is not None:
                        observed,probe,valid=next(donors)
                        if len(observed)!=n:
                            raise ValueError('Source/donor batch count differs')
                        with torch.no_grad():
                            _,donor_features=adaptation.outputs(name,ema,torch.cat([observed,probe]).to(args.device))
                            delta=(donor_features[n:]-donor_features[:n]).float()*valid.to(args.device)[:,None]
                        if not torch.isfinite(delta).all():
                            raise RuntimeError('Nonfinite donor features; last completed epoch preserved')
                    optimizer.zero_grad(set_to_none=True)
                    with base.amp_context(args):
                        source_logits,features=adaptation.outputs(name,model,torch.cat([clean,smooth]))
                        features=torch.stack([features[:n],features[n:]],1)
                        logits=torch.stack([source_logits[:n],source_logits[n:]],1)
                        ratio=args.delta_cap_ratio*min(1.,(epoch+(step+1)/len(source_loader))/args.ramp_epochs)
                        applied=None
                        if delta is not None:
                            applied=bounded_delta(features,delta,ratio)
                            transferred=head((features.float()+applied).reshape(-1,features.shape[-1])).reshape(n,2,2)
                            logits=torch.cat([logits,transferred],1)
                        loss,ce,rank,inversions=classification_and_ranking(logits,labels,args.ranking_weight)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                    optimizer.step()
                    if ema is not None:
                        engine.update_teacher(ema,model,args.ema_decay)
                    relative=(applied.norm(dim=-1)/features.detach().float().norm(dim=-1).clamp_min(1e-12)).mean().item() if applied is not None else 0.
                    totals += n*np.array([loss.item(),ce.item(),rank.item(),inversions.item(),relative,
                        float(delta.norm(dim=1).mean()) if delta is not None else 0.,
                        float(valid.float().mean()) if delta is not None else 0.])
                    count+=n
                values=engine.validate(args,model,val,transform)
                improved=is_better(values,best)
                if improved:
                    best={'auc':values['mean_auc'],'ber':values['mean_ber']}
                if values['mean_auc']>anchor+.0001:
                    anchor,stale=values['mean_auc'],0
                else:
                    stale+=1
                entry=dict(epoch=epoch+1,loss=float(totals[0]/count),source_ce=float(totals[1]/count),
                    source_rank=float(totals[2]/count),source_pair_inversion_fraction=float(totals[3]/count),
                    applied_delta_relative_norm=float(totals[4]/count),raw_delta_norm=float(totals[5]/count),
                    donor_detected_fraction=float(totals[6]/count),donor_exposures=count if plan is not None else 0,
                    target_exposures=count if strategy=='target_delta' else 0,
                    donor_condition_counts=dict(Counter(p['condition'] for p in plan)) if plan else {},
                    source_paths_sha256=evaluation.key(sampled.crop_path.tolist()),source_batches_sha256=evaluation.key(batches),
                    donor_plan_sha256=evaluation.key(plan) if plan is not None else None,validation=values)
                history.append(entry)
                state=dict(identity=identity,config=config,model_state_dict=model.state_dict(),
                    ema_state_dict=ema.state_dict() if ema is not None else None,optimizer=optimizer.state_dict(),
                    epoch=epoch+1,stale=stale,history=history,best=best,anchor=anchor)
                if improved:
                    base.common.save(run/'best.pt',{k:state[k] for k in ['identity','config','model_state_dict','epoch','best']}|{'validation':values})
                base.common.save(run/'last.pt',state)
                del state
                base.common.json_write(run/'history.json',history)
                print(f'CE={entry["source_ce"]:.4f} rank={entry["source_rank"]:.4f} delta/norm={entry["applied_delta_relative_norm"]:.3%} | '
                    f'val AUC={values["mean_auc"]:.4f} BER={values["mean_ber"]:.2%} best AUC={best["auc"]:.4f} patience={stale}/{args.patience}',flush=True)
            selected=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            base.common.json_write(run/'completed.json',dict(identity=identity,epochs_completed=len(history),
                selected_epoch=selected['epoch'],best_mean_auc=best['auc'],selected_mean_ber=best['ber'],inference_model='student'))
            del selected,model,ema,optimizer,parameters,groups,head
            if args.device=='cuda':
                torch.cuda.empty_cache()
        del initial


def evaluate(args):
    records,sources=reference.reference_records(args)
    copied=copy.copy(args)
    copied.reference_evaluations=args.hard_evaluations
    # Include completed sibling strategies: a later TS/strategy evaluation must not
    # silently erase a prior strategy from the summary (old Worst-view issue).
    available=[s for s in STRATEGIES if all((args.runs_root/run_name(n,s,args.seed)/'completed.json').is_file() for n in args.models)]
    if not set(args.strategies).issubset(available):
        raise FileNotFoundError('Requested training is not completed')
    copied.strategies=available
    with protocol_context():
        engine.evaluate(copied)
    table=pd.read_csv(args.output_root/'comparison.csv')
    table=table[table.strategy.isin(available)]
    combined=pd.concat([pd.DataFrame(records),table],ignore_index=True)
    keys=['model','strategy','condition']
    if combined.duplicated(keys).any() or len(combined)!=len(args.models)*(3+len(available))*4:
        raise ValueError('Incomplete comparison')
    combined.to_csv(args.output_root/'comparison_with_reference.csv',index=False)
    means=combined.groupby(['model','strategy'],sort=False)[reference.METRICS].mean().reset_index()
    means.to_csv(args.output_root/'condition_means_with_reference.csv',index=False)
    changes=[]
    for (name,strategy),group in combined.groupby(['model','strategy'],sort=False):
        by=group.set_index('condition');row=dict(model=name,strategy=strategy)
        for metric in reference.METRICS:
            row['normal_'+metric]=by.loc['normal',metric]
            row['strong_'+metric]=by.loc['smooth_strong',metric]
            row['strong_minus_normal_'+metric]=by.loc['smooth_strong',metric]-by.loc['normal',metric]
        changes.append(row)
    pd.DataFrame(changes).to_csv(args.output_root/'smoothing_changes.csv',index=False)
    differences=[]
    for _,row in means[means.strategy.isin(available)].iterrows():
        for control in ['baseline','augmentation','mean_teacher','ce_rank','source_delta']:
            match=means[means.model.eq(row.model)&means.strategy.eq(control)]
            if control==row.strategy or match.empty:
                continue
            differences.append(dict(model=row.model,strategy=row.strategy,reference=control,
                **{'delta_'+m:row[m]-match.iloc[0][m] for m in reference.METRICS}))
    pd.DataFrame(differences).to_csv(args.output_root/'deltas_vs_controls.csv',index=False)
    base.common.json_write(args.output_root/'comparison_protocol.json',dict(
        reused_reference_results=sources,new_selection='FF++ four-view mean AUC maximum',
        historical_reference_selection='Aug/TS: FF++ four-view mean BER minimum; descriptive reference, not matched ablation',
        target_labels_used=False,test_used_for_selection=False,completed_strategies=available))
    print('Saved:',args.output_root/'condition_means_with_reference.csv',flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate'])
    for field in ['source-root','target-root','baseline-runs-root','runs-root','manifest-root','validation-cache',
                  'output-root','hard-manifest-root','hard-evaluations']:
        p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--source-validation',type=Path)
    for field,default in [('train-cache','/content/smoothing_pivot_bilateral_cache_v1'),
                          ('target-train-cache','/content/smoothing_mean_teacher_target_cache_v1'),
                          ('cache-root','/content/celebdf_official_smoothing_cache_v1'),
                          ('evaluation-pixel-cache','/content/smoothing_ranking_cdf_bilateral_cache_v1')]:
        p.add_argument('--'+field,type=Path,default=Path(default))
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models',nargs='+',choices=list(base.PREPROCESS),default=['clip_b16'])
    p.add_argument('--strategies',nargs='+',choices=STRATEGIES,default=STRATEGIES)
    for field,default in [('seed',42),('epochs',10),('patience',3),('source-images',6000),('val-images',1200),
                          ('target-images',5000),('batch-size',16),('workers',2)]:
        p.add_argument('--'+field,type=int,default=default)
    for field,default in [('encoder-lr',1e-5),('head-lr',1e-5),('weight-decay',1e-4),('ranking-weight',.25),
                          ('delta-cap-ratio',.05),('ema-decay',.999),('ramp-epochs',2.)]:
        p.add_argument('--'+field,type=float,default=default)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args()
    numbers=[a.encoder_lr,a.head_lr,a.weight_decay,a.ranking_weight,a.delta_cap_ratio,a.ema_decay,a.ramp_epochs]
    if (not np.isfinite(numbers).all() or min(a.encoder_lr,a.head_lr,a.ramp_epochs)<=0 or
            min(a.weight_decay,a.ranking_weight,a.delta_cap_ratio)<0 or a.delta_cap_ratio>1 or not 0<=a.ema_decay<1 or
            min(a.epochs,a.patience,a.source_images,a.val_images,a.target_images,a.batch_size)<1 or
            a.batch_size%2 or a.source_images%8 or a.val_images%8 or a.workers<0 or
            len(set(a.models))!=len(a.models) or len(set(a.strategies))!=len(a.strategies)):
        p.error('Invalid numeric settings, batch size or duplicate models/strategies')
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__=='__main__':
    arguments=parse_args()
    (train if arguments.action=='train' else evaluate)(arguments)
