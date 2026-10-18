"""Per-image worst-view source CE, with and without hard Mean Teacher UDA.

The original source-average trainers, checkpoint hashes and pixel caches stay intact.
Only the reduction of existing clean/smoothed source losses changes.
"""
import argparse
import copy
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

import ffpp_c23_smoothing_mean_teacher as engine
import ffpp_c23_smoothing_mean_teacher_soft as reference

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_worst_view_v1'
STRATEGIES = ['worst_view_aug','worst_view_ts']
MATCHED_REFERENCE = {'worst_view_aug':'augmentation','worst_view_ts':'mean_teacher'}
common, base, adaptation, evaluation = engine.common, engine.base, engine.adaptation, engine.evaluation
CONDITIONS = engine.CONDITIONS
freeze_json, freeze_csv = engine.freeze_json, engine.freeze_csv
code_hashes, target_plan, TargetViews = engine.code_hashes, engine.target_plan, engine.TargetViews
validate, pseudo_loss, update_teacher = engine.validate, engine.pseudo_loss, engine.update_teacher
run_name = engine.run_name


def worst_source_loss(logits, labels):
    """Mean over images of max(clean CE, smoothed CE), using actual FF++ labels.

    Choose independently for each source image, never a maximum over the batch.
    The max is unscaled; at an exact tie torch.maximum splits gradients equally.
    """
    n = len(labels)
    if n < 1 or logits.shape != (2*n,2):
        raise ValueError('Expected clean+smoothed binary logits and one label per image')
    per_image = nn.functional.cross_entropy(logits.float(),labels.repeat(2),reduction='none')
    clean, smoothed = per_image[:n], per_image[n:]
    harder, ties = smoothed>clean, smoothed==clean
    real, fake = labels.eq(0), labels.eq(1)
    stats = torch.stack([clean.sum(),smoothed.sum(),harder.sum(),ties.sum(),
                         (harder&real).sum(),(harder&fake).sum(),real.sum(),fake.sum()]).detach()
    return torch.maximum(clean,smoothed).mean(), stats


def configuration(args,name,strategy,data,cfg,initial,initial_hash):
    result = engine.configuration(args,name,strategy,data,cfg,initial,initial_hash)
    result.update(
        source_reduction='per-image max(clean CE, random-smoothed CE), then mean; no extra multiplier',
        loss='source worst-view CE; worst_view_ts adds unchanged confidence-masked hard CDF CE with ramp',
        target_views='teacher observed normal/weak/medium/strong uniformly; student complementary flip; no extra smoothing',
        target_enabled=strategy=='worst_view_ts',
        matched_reference=MATCHED_REFERENCE[strategy],
        extension_code_sha256=base.common.digest(Path(__file__)),
        hard_reference_manifest=args._hard_reference_identity)
    return result


@contextmanager
def protocol_context():
    old = engine.PROTOCOL
    engine.PROTOCOL = PROTOCOL
    try:
        yield
    finally:
        engine.PROTOCOL = old


def train_loop(args):
    pool, val, source_data = common.shared.source_inputs(args)
    _, other_val, target, cfg, target_data, blocked = adaptation.dataset_inputs(args)
    if list(cfg['conditions']) != CONDITIONS or val.crop_path.tolist() != other_val.crop_path.tolist():
        raise ValueError('Source validation or smoothing differs')
    target = target[['crop_path', 'video_id', 'roi_status']]
    data = {**source_data, **target_data}
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    freeze_json(args.manifest_root/'protocol.json', {'protocol': PROTOCOL, 'data': data,
        'source_images': args.source_images, 'target_images': args.target_images,
        'seed': args.seed, 'target_labels_used': False, 'excluded_video_ids': blocked})
    freeze_csv(args.manifest_root/'source_validation.csv', val)
    freeze_csv(args.manifest_root/'target_unlabelled.csv', target)
    common.prepare_pixel_cache(args, cfg, source_data)
    target_identity = {'target_paths_sha256': data['target_paths_sha256'], 'smoothing': cfg,
        'opencv': evaluation.cv2.__version__, 'code_sha256': code_hashes()}
    args._target_cache = common.pixel_cache.BilateralCache(args.target_train_cache/'observed', cfg['bilateral'], target_identity)
    args._target_probe_cache = common.pixel_cache.BilateralCache(args.target_train_cache/'extra_weak', cfg['bilateral'], target_identity)
    common.prepare_validation(args, val, cfg, source_data)
    print(f'Source {args.source_images}/epoch; target pool {len(target)} images/{target.video_id.nunique()} videos; GT-free sampling', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Baseline source manifest mismatch')
        for strategy in args.strategies:
            config = configuration(args, name, strategy, data, cfg, initial, initial_hash)
            identity = evaluation.key(config)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            freeze_json(run/'config.json', config)
            if (run/'completed.json').is_file():
                done = json.loads((run/'completed.json').read_text())
                if done['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError('Completed run mismatch')
                print('Already completed:', name, strategy, flush=True)
                continue
            base.common.seed_all(args.seed)
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(initial['model_state_dict'], strict=True)
            model.to(args.device)
            blocks = 2 if name == 'xception' else initial['config']['train_blocks']
            common.training_mode(model, name, blocks)
            teacher = None if strategy == 'worst_view_aug' else copy.deepcopy(model).requires_grad_(False).eval()
            head_params = list(base.head(model, name).parameters())
            ids = {id(p) for p in head_params}
            groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in ids], 'lr': args.encoder_lr},
                      {'params': head_params, 'lr': args.head_lr}]
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            parameters = [p for group in groups for p in group['params']]
            start, stale, history = 0, 0, []
            if (run/'last.pt').is_file():
                state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if state['identity'] != identity:
                    raise ValueError('Resume identity differs')
                model.load_state_dict(state['model_state_dict'], strict=True)
                if teacher is not None:
                    teacher.load_state_dict(state['teacher_state_dict'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start, stale, history, best, anchor = [state[k] for k in ['epoch','stale','history','best','anchor']]
                del state
                print('Resume completed epoch:', start, flush=True)
            else:
                reference = validate(args, model, val, transform)
                best = anchor = reference['mean_ber']
                base.common.json_write(run/'epoch0_validation.json', reference)
                base.common.save(run/'best.pt', {'identity': identity, 'config': config,
                    'model_state_dict': model.state_dict(), 'epoch': 0, 'best': best, 'validation': reference})
            print(f'===== {name} / {strategy} / student selected by FF++ mean BER =====', flush=True)
            for epoch in range(start, args.epochs):
                if stale >= args.patience:
                    break
                seed = args.seed+epoch*100003
                base.common.seed_all(seed)
                common.training_mode(model, name, blocks)
                sampled = adaptation.balanced_source(pool, args.source_images, seed)
                source_loader = base.common.make_loader(common.PairedSource(sampled, args, transform, cfg, epoch), args, True, seed)
                plan = target_plan(target, len(sampled), seed)
                freeze_csv(args.manifest_root/f'target_exposures_epoch{epoch+1:02d}.csv', pd.DataFrame(plan))
                target_loader = None
                if teacher is not None:
                    target_loader = iter(base.common.make_loader(TargetViews(plan, args, transform, cfg,
                        False), args, True, seed))
                totals, stats, count = np.zeros(4), np.zeros(4, dtype=np.int64), 0
                source_totals = np.zeros(8)
                for step, (clean, changed, labels, _, _) in enumerate(tqdm(source_loader,
                        desc=f'{name} {strategy} epoch{epoch+1}/{args.epochs}', mininterval=15, dynamic_ncols=True)):
                    # Target forwards must not change the source dropout RNG in the next batch.
                    base.common.seed_all(seed+step)
                    clean, changed, labels = clean.to(args.device), changed.to(args.device), labels.to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    with base.amp_context(args):
                        logits = model(torch.cat([clean, changed])).float()
                        ce, source_stats = worst_source_loss(logits, labels)
                    target_loss, confidence_sum = ce.new_zeros(()), ce.new_zeros(())
                    ramp = min(1., (epoch+(step+1)/len(source_loader))/args.ramp_epochs)
                    if teacher is not None:
                        xt, xs = next(target_loader)
                        with torch.no_grad():
                            teacher_logits = teacher(xt.to(args.device)).float()
                        with base.amp_context(args):
                            student_logits = model(xs.to(args.device)).float()
                        if not torch.isfinite(teacher_logits).all() or not torch.isfinite(student_logits).all():
                            raise RuntimeError('Nonfinite target logits; last completed epoch preserved')
                        target_loss, batch_stats, confidence_sum = pseudo_loss(student_logits, teacher_logits, args.pseudo_threshold)
                        stats += batch_stats.cpu().numpy()
                    loss = ce+args.target_weight*ramp*target_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    if teacher is not None:
                        update_teacher(teacher, model, args.ema_decay)
                    n = len(labels)
                    totals += [float(ce.detach())*n, float(target_loss.detach())*n,
                               float(confidence_sum.detach()), (args.target_weight*ramp if teacher is not None else 0.)*n]
                    source_totals += source_stats.cpu().numpy()
                    count += n
                values = validate(args, model, val, transform)
                score, improved = values['mean_ber'], values['mean_ber'] < best
                best = min(best, score)
                if score < anchor-.0001:
                    anchor, stale = score, 0
                else:
                    stale += 1
                entry = {'epoch': epoch+1, 'source_ce': float(totals[0]/count),
                    'source_clean_ce': float(source_totals[0]/count),
                    'source_smoothed_ce': float(source_totals[1]/count),
                    'smoothed_harder_fraction': float(source_totals[2]/count),
                    'equal_loss_fraction': float(source_totals[3]/count),
                    'smoothed_harder_by_class': {label: float(source_totals[i]/max(source_totals[j],1))
                        for label,i,j in [('real',4,6),('fake',5,7)]},
                    'target_ce': float(totals[1]/count), 'mean_target_weight': float(totals[3]/count),
                    'teacher_mean_confidence': float(totals[2]/count),
                    'teacher_predicted_counts': dict(zip(['real','fake'], stats[:2].tolist())),
                    'pseudo_selected_counts': dict(zip(['real','fake'], stats[2:].tolist())),
                    'pseudo_selected_fraction': float(stats[2:].sum()/count),
                    'target_condition_counts': dict(Counter(p['condition'] for p in plan)),
                    'target_exposures': count if teacher is not None else 0,
                    'source_paths_sha256': evaluation.key(sampled.crop_path.tolist()),
                    'target_plan_sha256': evaluation.key(plan) if teacher is not None else None, 'validation': values}
                history.append(entry)
                state = {'identity': identity, 'config': config, 'model_state_dict': model.state_dict(),
                    'teacher_state_dict': teacher.state_dict() if teacher is not None else None,
                    'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'stale': stale,
                    'best': best, 'anchor': anchor, 'history': history}
                if improved:
                    base.common.save(run/'best.pt', {k: state[k] for k in ['identity','config','model_state_dict','epoch','best']} | {'validation': values})
                base.common.save(run/'last.pt', state)
                del state
                base.common.json_write(run/'history.json', history)
                print(f'CE={entry["source_ce"]:.4f} targetCE={entry["target_ce"]:.4f} | '
                    f'pseudo selected={entry["pseudo_selected_counts"]} ({entry["pseudo_selected_fraction"]:.1%}) | '
                    f'val BER={score:.2%} AUC={values["mean_auc"]:.4f} best={best:.2%} patience={stale}/{args.patience}', flush=True)
            selected = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            base.common.json_write(run/'completed.json', {'identity': identity, 'epochs_completed': len(history),
                'selected_epoch': selected['epoch'], 'best_mean_ber': best, 'inference_model': 'student'})
            del selected, teacher, model, optimizer, parameters, groups, head_params
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial



def train(args):
    reference.verify_membership(args)
    with protocol_context():
        train_loop(args)


def evaluate(args):
    # Check saved old results before spending time on new inference.
    records, sources = reference.reference_records(args)
    for name in args.models:
        for strategy in args.strategies:
            run=args.runs_root/f'{name}_{strategy}_{PROTOCOL}_seed{args.seed}'
            saved=json.loads((run/'config.json').read_text())
            if (saved.get('extension_code_sha256')!=base.common.digest(Path(__file__))
                    or saved.get('target_enabled')!=(strategy=='worst_view_ts')
                    or saved.get('matched_reference')!=MATCHED_REFERENCE[strategy]
                    or not saved.get('source_reduction','').startswith('per-image max(')):
                raise ValueError(f'Worst-view configuration/code mismatch: {run}')
    copied=copy.copy(args)
    copied.reference_evaluations=args.hard_evaluations
    with protocol_context():
        engine.evaluate(copied)
    current=pd.read_csv(args.output_root/'comparison.csv')
    current=current[current.strategy.isin(args.strategies)]
    combined=pd.concat([pd.DataFrame(records),current],ignore_index=True)
    keys=['model','strategy','condition']
    expected=pd.MultiIndex.from_product([args.models,['baseline','augmentation','mean_teacher',*args.strategies],CONDITIONS],names=keys)
    actual=pd.MultiIndex.from_frame(combined[keys])
    if actual.has_duplicates or set(actual)!=set(expected):
        raise ValueError('Incomplete comparison rows')
    metrics=reference.METRICS
    combined.to_csv(args.output_root/'comparison_with_reference.csv',index=False)
    means=combined.groupby(['model','strategy'],sort=False)[metrics].mean().reset_index()
    means.to_csv(args.output_root/'condition_means_with_reference.csv',index=False)
    changes=[]
    for (name,strategy),group in combined.groupby(['model','strategy'],sort=False):
        by=group.set_index('condition')
        row={'model':name,'strategy':strategy}
        for metric in metrics:
            row['normal_'+metric]=by.loc['normal',metric]
            row['strong_'+metric]=by.loc['smooth_strong',metric]
            row['strong_minus_normal_'+metric]=by.loc['smooth_strong',metric]-by.loc['normal',metric]
        changes.append(row)
    pd.DataFrame(changes).to_csv(args.output_root/'smoothing_changes.csv',index=False)
    old=combined[combined.strategy.isin(['augmentation','mean_teacher'])].set_index(keys)
    differences=current.copy()
    differences['reference_strategy']=differences.strategy.map(MATCHED_REFERENCE)
    for metric in metrics:
        differences['delta_'+metric]=[
            row[metric]-old.loc[(row.model,MATCHED_REFERENCE[row.strategy],row.condition),metric]
            for _,row in differences.iterrows()]
    differences.to_csv(args.output_root/'deltas_vs_matched.csv',index=False)
    base.common.json_write(args.output_root/'reused_reference_results.json',
        {'reference_protocol':reference.HARD_PROTOCOL,'sources':sources,'old_models_retrained':False})
    print('Saved average/worst-view and Aug/TS comparison:',args.output_root,flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate'])
    for name in ['source-root','target-root','baseline-runs-root','runs-root','manifest-root',
                 'validation-cache','output-root','hard-manifest-root','hard-evaluations']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-validation',type=Path)
    for name,path in {
        'train-cache':'/content/smoothing_pivot_bilateral_cache_v1',
        'target-train-cache':'/content/smoothing_mean_teacher_target_cache_v1',
        'cache-root':'/content/celebdf_official_smoothing_cache_v1',
        'evaluation-pixel-cache':'/content/smoothing_ranking_cdf_bilateral_cache_v1',
    }.items():
        p.add_argument('--'+name,type=Path,default=Path(path))
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models',nargs='+',choices=list(base.PREPROCESS),default=['xception'])
    p.add_argument('--strategies',nargs='+',choices=STRATEGIES,default=STRATEGIES)
    for name,value in [('seed',42),('epochs',10),('patience',3),('source-images',6000),
                       ('val-images',1200),('target-images',5000),('batch-size',16),('workers',2)]:
        p.add_argument('--'+name,type=int,default=value)
    for name,value in [('encoder-lr',1e-5),('head-lr',1e-5),('weight-decay',1e-4),
                       ('ema-decay',.999),('pseudo-threshold',.9),('target-weight',.5),('ramp-epochs',2.)]:
        p.add_argument('--'+name,type=float,default=value)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args()
    floats=[args.encoder_lr,args.head_lr,args.weight_decay,args.ema_decay,
            args.pseudo_threshold,args.target_weight,args.ramp_epochs]
    if (not np.isfinite(floats).all() or min(args.encoder_lr,args.head_lr,args.ramp_epochs)<=0
            or min(args.weight_decay,args.target_weight)<0
            or not 0<=args.ema_decay<1 or not .5<=args.pseudo_threshold<=1
            or min(args.epochs,args.patience,args.source_images,args.val_images,args.target_images,args.batch_size)<1
            or args.source_images%8 or args.val_images%8 or args.workers<0
            or len(set(args.models))!=len(args.models) or len(set(args.strategies))!=len(args.strategies)):
        p.error('Invalid settings or duplicate models/strategies')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return args


if __name__=='__main__':
    arguments=parse_args()
    (train if arguments.action=='train' else evaluate)(arguments)

