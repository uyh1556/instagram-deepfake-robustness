"""Stage 0/1: pooled-AUC CE+Rank control and class-balanced CDF SDA on a rebuilt pool.

Pool v2 replaces the transitive filename-component exclusion (which removed 98.6%
of non-test fakes) with a ONE-HOP family exclusion, a test-like class/source mix,
a per-scene fake cap and a component-disjoint adapt-train/adapt-val split.
SDA uses actual non-test labels: a diagnostic ceiling for later UDA, not UDA.
Official test labels never supervise training or checkpoint selection.
"""
import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import ffpp_c23_smoothing_delta_transfer as rank
import ffpp_c23_smoothing_rank_adaptation as sda_v1

engine, common, base = rank.engine, rank.common, rank.base
adaptation, evaluation, reference = rank.adaptation, rank.evaluation, rank.reference
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_rank_sda_pool_v2'
STRATEGIES = ['ce_rank_pooled', 'sda_new_balanced', 'sda_old_balanced']
CONDITIONS = engine.CONDITIONS
KINDS = ['Celeb-real', 'YouTube-real', 'Celeb-synthesis']
KIND_LABEL = {'Celeb-real': '0', 'YouTube-real': '0', 'Celeb-synthesis': '1'}
CHECKPOINTS = {'ffpp': 'best.pt', 'adapt_val': 'best_adapt_val.pt'}
POOL_RULES = dict(
    exclusion='drop a non-test video when any of its scene keys (both ids of idA_idB_NNNN, or idA_NNNN) is a key of ANY official test video; one hop, NOT transitive',
    composition='fake videos capped per primary scene (idA_NNNN); real video count/ratio copied from official test video counts',
    split='union-find over scene keys of the SELECTED videos; whole components go to adapt-train or adapt-val',
    limits='filename scene keys are candidates, not verified source/driver metadata; identities overlap with test as before')


def code_hashes():
    return {**rank.code_hashes(), Path(sda_v1.__file__).name: base.common.digest(Path(sda_v1.__file__)),
            Path(__file__).name: base.common.digest(Path(__file__))}


def run_name(model, strategy, seed):
    return f'{model}_{strategy}_{PROTOCOL}_seed{seed}'


def kind_of(video_id):
    kind = video_id.split('/', 1)[0]
    if kind not in KINDS:
        raise ValueError(f'Unknown Celeb-DF source folder: {video_id}')
    return kind


def scene_keys(video_id):
    """Primary key first. YouTube-real has no filename scene relation."""
    fake = re.fullmatch(r'Celeb-synthesis/(id\d+)_(id\d+)_(\d+)', video_id)
    if fake:
        return [f'{fake.group(1)}_{fake.group(3)}', f'{fake.group(2)}_{fake.group(3)}']
    real = re.fullmatch(r'Celeb-real/(id\d+)_(\d+)', video_id)
    if real:
        return [f'{real.group(1)}_{real.group(2)}']
    if kind_of(video_id) == 'YouTube-real':
        return []
    raise ValueError(f'Unexpected Celeb-DF filename: {video_id}')


def read_catalogs(root):
    catalog = pd.concat([pd.read_csv(root/'manifests'/f'celebdf_v2_roi16_non_test_{k}_v1.csv', dtype=str, keep_default_na=False)
                         for k in ['real', 'fake']], ignore_index=True)
    test = pd.read_csv(root/'manifests/celebdf_v2_roi16_test_v1.csv', dtype=str, keep_default_na=False)
    for name, frame, split in [('non-test', catalog, 'non_test_pool'), ('test', test, 'test')]:
        kinds = frame.video_id.map(kind_of)
        if (not frame.split.eq(split).all() or frame.crop_path.duplicated().any()
                or not frame.class_index.eq(kinds.map(KIND_LABEL)).all()):
            raise ValueError(f'Invalid {name} manifest: split, duplicate or label/source-folder mismatch')
    if set(catalog.video_id) & set(test.video_id):
        raise ValueError('Non-test/test video overlap')
    return catalog, test


def components(videos):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for video in videos:
        keys = scene_keys(video) or [video]
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])
        find(keys[0])
    return {video: find((scene_keys(video) or [video])[0]) for video in videos}


def build_pool(args, old_target):
    """Deterministic pool v2 membership. Uses folder/class metadata by DESIGN (test-like
    composition); the unlabelled export keeps only crop_path/video_id/roi_status."""
    catalog, test = read_catalogs(args.target_root)
    rng = random.Random(args.seed+520001)
    test_keys = {k for v in test.video_id.unique() for k in scene_keys(v)}
    videos = sorted(catalog.video_id.unique())
    eligible = [v for v in videos if not set(scene_keys(v)) & test_keys]
    by_kind = {k: [v for v in eligible if kind_of(v) == k] for k in KINDS}
    families = {}
    for video in by_kind['Celeb-synthesis']:
        families.setdefault(scene_keys(video)[0], []).append(video)
    fakes = []
    for family in sorted(families):
        members = sorted(families[family])
        rng.shuffle(members)
        fakes += members[:args.family_cap]
    test_videos = Counter(kind_of(v) for v in test.video_id.unique())
    test_real = test_videos['Celeb-real']+test_videos['YouTube-real']
    n_real = round(len(fakes)*test_real/test_videos['Celeb-synthesis'])
    wanted = {'Celeb-real': round(n_real*test_videos['Celeb-real']/test_real)}
    wanted['YouTube-real'] = n_real-wanted['Celeb-real']
    selected = {'Celeb-synthesis': sorted(fakes)}
    for kind, count in wanted.items():
        if count > len(by_kind[kind]):
            raise ValueError(f'Only {len(by_kind[kind])} eligible {kind} videos for {count}; lower --family-cap')
        selected[kind] = sorted(rng.sample(by_kind[kind], count))
    chosen = sorted(v for kind in KINDS for v in selected[kind])
    group = components(chosen)
    totals = Counter(kind_of(v) for v in chosen)
    target = {k: round(args.adapt_val_fraction*totals[k]) for k in KINDS}
    members = {}
    for video in chosen:
        members.setdefault(group[video], []).append(video)
    order = sorted(members)
    rng.shuffle(order)
    val_counts, partition = Counter(), {}
    for root in order:
        need = Counter(kind_of(v) for v in members[root])
        to_val = all(val_counts[k]+need[k] <= target[k] for k in need)
        if to_val:
            val_counts.update(need)
        for video in members[root]:
            partition[video] = 'adapt_val' if to_val else 'adapt_train'
    frame = catalog[catalog.video_id.isin(chosen)][['crop_path', 'video_id', 'roi_status', 'class_index']].copy()
    frame['kind'] = frame.video_id.map(kind_of)
    frame['scene_group'] = frame.video_id.map(group)
    frame['partition'] = frame.video_id.map(partition)
    frame = frame.sort_values(['partition', 'video_id', 'crop_path'], kind='stable').reset_index(drop=True)
    if frame.groupby('scene_group').partition.nunique().gt(1).any() or set(frame.video_id) != set(chosen):
        raise ValueError('Pool v2 split leaked a scene component or lost a video')
    old_videos = set(old_target.video_id)
    counts = {part: {k: dict(videos=int(g[g.kind.eq(k)].video_id.nunique()), images=int(g.kind.eq(k).sum())) for k in KINDS}
              for part, g in frame.groupby('partition')}
    summary = dict(rules=POOL_RULES, seed=args.seed, family_cap=args.family_cap, adapt_val_fraction=args.adapt_val_fraction,
        non_test_videos={k: sum(kind_of(v) == k for v in videos) for k in KINDS},
        eligible_videos={k: len(by_kind[k]) for k in KINDS}, fake_scene_families=len(families),
        official_test_videos={k: test_videos[k] for k in KINDS}, partitions=counts,
        scene_components=len(members), old_pool_videos_in_adapt_val=len(old_videos & set(frame[frame.partition.eq('adapt_val')].video_id)),
        old_pool_videos_in_pool_v2=len(old_videos & set(chosen)),
        membership_sha256=evaluation.key(frame.to_dict('records')))
    return frame, summary


def freeze_pool(args, old_target):
    frame, summary = build_pool(args, old_target)
    folder = args.manifest_root/'pool_v2'
    folder.mkdir(parents=True, exist_ok=True)
    engine.freeze_json(folder/'summary.json', summary)
    engine.freeze_csv(folder/'membership_labelled.csv', frame)
    train = frame[frame.partition.eq('adapt_train')]
    # Future UDA reads only this label-free file.
    engine.freeze_csv(folder/'adapt_train_unlabelled.csv', train[['crop_path', 'video_id', 'roi_status']])
    return frame, summary


def balanced_plan(labelled, n, batch_size, seed):
    """Exactly batch_size/2 real + batch_size/2 fake per consecutive batch; video-uniform
    within class. Rows never carry the label; SDA looks it up by crop_path."""
    if n % batch_size or batch_size % 2:
        raise ValueError('Target exposures must fill even, class-balanced batches')
    banks = {}
    for row in labelled[['crop_path', 'video_id', 'roi_status', 'class_index']].to_dict('records'):
        banks.setdefault(row.pop('class_index'), {}).setdefault(row['video_id'], []).append(row)
    if set(banks) != {'0', '1'}:
        raise ValueError('Balanced SDA requires real and fake target images')
    videos = {y: sorted(v) for y, v in banks.items()}
    rng = random.Random(seed+810001)
    plan = []
    for _ in range(n//batch_size):
        block = []
        for y in ['0', '1']:
            for _ in range(batch_size//2):
                row = rng.choice(banks[y][rng.choice(videos[y])])
                block.append({**row, 'condition': rng.choice(CONDITIONS), 'flip': rng.random() < .5})
        rng.shuffle(block)
        plan += block
    return plan


def fast_auc(labels, scores):
    labels = np.asarray(labels).astype(bool)
    ranks = rankdata(scores)
    positives = labels.sum()
    return float((ranks[labels].sum()-positives*(positives+1)/2)/(positives*(len(labels)-positives)))


def summarize(labels, banks, records):
    """Pooled = all four conditions in ONE ranking; cross[i][j] = real condition i vs fake j."""
    real, fake = labels == 0, labels == 1
    cross = [[fast_auc(np.r_[np.zeros(real.sum()), np.ones(fake.sum())], np.r_[banks[i][real], banks[j][fake]])
              for j in range(len(banks))] for i in range(len(banks))]
    return dict(pooled_auc=fast_auc(np.tile(labels, len(banks)), np.concatenate(banks)),
        mean_auc=float(np.mean([r['auc'] for r in records.values()])),
        mean_ber=float(np.mean([r['balanced_error_rate'] for r in records.values()])),
        min_cross_auc=float(np.min(cross)), strong_real_vs_normal_fake_auc=cross[-1][0],
        cross_auc=cross, conditions=records)


def score_conditions(args, model, frame, dataset, title):
    banks, records = [], {}
    for condition in CONDITIONS:
        probabilities, margins = engine.infer(args, model, frame, dataset(condition), f'{title} {condition}')
        records[condition] = engine.raw_metrics(frame, probabilities, margins)
        banks.append(margins)
    return summarize(frame.class_index.astype(int).to_numpy(), banks, records)


def validate(args, model, val, adapt_val, transform, cfg):
    model.eval()
    values = dict(ffpp=score_conditions(args, model, val, lambda c: common.ValidationViews(val, args, transform, c), 'FF++ val'))
    values['adapt_val'] = score_conditions(args, model, adapt_val,
        lambda c: engine.TestViews(adapt_val, args, transform, cfg, c, args._pool_cache, None), 'CDF adapt-val')
    return values


def is_better(values, best):
    return (values['pooled_auc'] > best['pooled_auc'] or
            (values['pooled_auc'] == best['pooled_auc'] and values['min_cross_auc'] > best['min_cross_auc']))


def best_of(values):
    return dict(pooled_auc=values['pooled_auc'], min_cross_auc=values['min_cross_auc'])


def inputs(args):
    pool, val, source_data = common.shared.source_inputs(args)
    _, other_val, old_target, cfg, old_data, _ = adaptation.dataset_inputs(args)
    if not val.equals(other_val) or list(cfg['conditions']) != CONDITIONS:
        raise ValueError('Source validation/smoothing mismatch')
    old_target = old_target[['crop_path', 'video_id', 'roi_status']]
    args._hard_reference_identity = None
    if 'sda_old_balanced' in args.strategies and args.hard_manifest_root is not None:
        reference.verify_membership(args)
    membership, summary = freeze_pool(args, old_target)
    engine.freeze_json(args.manifest_root/'protocol.json', dict(protocol=PROTOCOL, seed=args.seed,
        source_images=args.source_images, old_target_images=args.target_images, pool_v2=summary,
        target_supervision={'ce_rank_pooled': 'none', 'sda_new_balanced': 'actual labels, pool v2 adapt-train',
                            'sda_old_balanced': 'actual labels, previous 5000-image pool'}))
    engine.freeze_csv(args.manifest_root/'source_validation.csv', val)
    adapt_val = membership[membership.partition.eq('adapt_val')].reset_index(drop=True)
    targets = {'sda_new_balanced': membership[membership.partition.eq('adapt_train')].reset_index(drop=True)}
    if 'sda_old_balanced' in args.strategies:
        targets['sda_old_balanced'] = sda_v1.labelled_target(old_target, args.target_root)
    data = dict(source_data, test_manifest_sha256=old_data['test_manifest_sha256'],
        old_target_paths_sha256=old_data['target_paths_sha256'], pool_v2_membership_sha256=summary['membership_sha256'])
    for path in membership.crop_path:
        if not (args.target_root/path).is_file():
            raise FileNotFoundError(args.target_root/path)
    return pool, val, adapt_val, targets, cfg, data, summary


def configuration(args, name, strategy, data, cfg, initial, initial_hash, target, summary):
    labelled = target is not None
    return dict(protocol=PROTOCOL, model=name, strategy=strategy, seed=args.seed, data=data, smoothing=cfg,
        initial_checkpoint_sha256=initial_hash, preprocessing=base.PREPROCESS[name],
        train_blocks=initial['config']['train_blocks'], epochs=args.epochs, patience=args.patience, min_delta=.0001,
        source_images=args.source_images, val_images=args.val_images, batch_size=args.batch_size,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
        ranking_weight=args.ranking_weight, target_weight=args.target_weight if labelled else 0.,
        ramp_epochs=args.ramp_epochs,
        source_loss='CE + ranking_weight*mean softplus(real_margin-fake_margin) over all cross-view pairs (same as CE+Rank v1)',
        target_loss='actual-label CE on every exposure, whole-batch mean, linear ramp' if labelled else 'none',
        target_pool={'ce_rank_pooled': None, 'sda_new_balanced': 'pool v2 adapt-train',
                     'sda_old_balanced': 'previous 5000-image transitive-exclusion pool'}[strategy],
        target_sampling='exact Real:Fake 1:1 per batch, video-uniform within class; source-matched exposure count' if labelled else 'none',
        target_label_sha256=evaluation.key(target.class_index.tolist()) if labelled else None,
        target_class_counts=target.class_index.value_counts().sort_index().to_dict() if labelled else None,
        target_views='normal/weak/medium/strong uniformly; complementary student flip',
        selection='best.pt: maximum FF++ val pooled raw-margin AUC over 4 conditions; tie: higher minimum cross-condition AUC; epoch0 eligible',
        diagnostic_selection='best_adapt_val.pt: same rule on CDF pool-v2 adapt-val (labelled non-test); never official test',
        early_stopping='FF++ pooled AUC, patience/min_delta',
        adapt_val_overlaps_training=(strategy == 'sda_old_balanced' and summary['old_pool_videos_in_adapt_val'] > 0),
        bn_statistics='frozen', inference_model='student',
        precision='BF16 if supported else FP32 training; FP32 validation/test',
        hard_reference_manifest=args._hard_reference_identity,
        versions={p: engine.metadata.version(p) for p in ['torch', 'torchvision', 'Pillow', 'numpy', 'pandas', 'scikit-learn']},
        torch=str(torch.__version__), code_sha256=code_hashes())


def target_cache(args, cfg, tag, paths_key):
    identity = dict(pool=tag, paths_sha256=paths_key, smoothing=cfg, opencv=evaluation.cv2.__version__,
                    pixel_code=base.common.digest(Path(common.pixel_cache.__file__)))
    return common.pixel_cache.BilateralCache(args.target_train_cache/tag, cfg['bilateral'], identity)


def prepare(args):
    _, _, adapt_val, targets, _, _, summary = inputs(args)
    for name in args.models:
        evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
    print('Pool v2 videos/images by partition:', json.dumps(summary['partitions'], indent=1), flush=True)
    print('Eligible (one-hop):', summary['eligible_videos'], '| old pool videos in adapt-val:',
          summary['old_pool_videos_in_adapt_val'], flush=True)
    for strategy, frame in targets.items():
        print(f'{strategy}: {len(frame)} labelled target images, classes {frame.class_index.value_counts().sort_index().to_dict()}', flush=True)
    print('adapt-val:', len(adapt_val), 'images', flush=True)


def train(args):
    pool, val, adapt_val, targets, cfg, data, summary = inputs(args)
    common.prepare_pixel_cache(args, cfg, {k: data[k] for k in ['source_trainval_sha256']})
    common.prepare_validation(args, val, cfg, data={k: data[k] for k in
        ['source_trainval_sha256', 'source_protocol_sha256', 'val_paths_sha256', 'validation_images']})
    args._pool_cache = target_cache(args, cfg, 'pool_v2_observed', data['pool_v2_membership_sha256'])
    caches = {'sda_new_balanced': args._pool_cache,
              'sda_old_balanced': target_cache(args, cfg, 'old_pool_observed', data['old_target_paths_sha256'])}
    print(f'Source={args.source_images}/epoch; FF++ val={len(val)} x4; CDF adapt-val={len(adapt_val)} x4; best=FF++ pooled AUC', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Baseline/source mismatch')
        for strategy in args.strategies:
            target = targets.get(strategy) if strategy != 'ce_rank_pooled' else None
            labels = target.set_index('crop_path').class_index.astype(int).to_dict() if target is not None else None
            config = configuration(args, name, strategy, data, cfg, initial, initial_hash, target, summary)
            identity = evaluation.key(config)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            engine.freeze_json(run/'config.json', config)
            if (run/'completed.json').is_file():
                done = json.loads((run/'completed.json').read_text())
                if done['identity'] != identity or not all((run/f).is_file() for f in CHECKPOINTS.values()):
                    raise ValueError('Completed run mismatch')
                print('Already completed:', name, strategy, flush=True)
                continue
            base.common.seed_all(args.seed)
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(initial['model_state_dict'], strict=True)
            model.to(args.device)
            blocks = 2 if name == 'xception' else initial['config']['train_blocks']
            common.training_mode(model, name, blocks)
            head = base.head(model, name)
            head_ids = {id(p) for p in head.parameters()}
            groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr': args.encoder_lr},
                      {'params': list(head.parameters()), 'lr': args.head_lr}]
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            parameters = [p for group in groups for p in group['params']]
            start, stale, history = 0, 0, []
            if (run/'last.pt').is_file():
                state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if state['identity'] != identity:
                    raise ValueError('Resume identity differs')
                model.load_state_dict(state['model_state_dict'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start, stale, history, best, best_adapt, anchor = [state[k] for k in
                    ['epoch', 'stale', 'history', 'best', 'best_adapt', 'anchor']]
                del state
                print('Resume completed epoch:', start, flush=True)
            else:
                values = validate(args, model, val, adapt_val, transform, cfg)
                best, best_adapt = best_of(values['ffpp']), best_of(values['adapt_val'])
                anchor = best['pooled_auc']
                base.common.json_write(run/'epoch0_validation.json', values)
                for key, file in CHECKPOINTS.items():
                    base.common.save(run/file, dict(identity=identity, config=config, model_state_dict=model.state_dict(),
                        epoch=0, selection=key, best=best_of(values[key]), validation=values))
            for epoch in range(start, args.epochs):
                if stale >= args.patience:
                    break
                seed = args.seed+epoch*100003
                base.common.seed_all(seed)
                common.training_mode(model, name, blocks)
                sampled = adaptation.balanced_source(pool, args.source_images, seed)
                batches = rank.balanced_batches(torch.tensor(sampled.class_index.astype(int).to_numpy()), args.batch_size, seed)
                source_loader = DataLoader(common.PairedSource(sampled, args, transform, cfg, epoch), batch_sampler=batches,
                    num_workers=args.workers, pin_memory=args.device == 'cuda', persistent_workers=False,
                    generator=torch.Generator().manual_seed(seed))
                engine.freeze_csv(args.manifest_root/f'source_epoch{epoch+1:02d}.csv', sampled)
                plan, target_loader = None, None
                if target is not None:
                    plan = balanced_plan(target, len(sampled), args.batch_size, seed)
                    engine.freeze_csv(args.manifest_root/strategy/f'target_exposures_epoch{epoch+1:02d}.csv', pd.DataFrame(plan))
                    args._target_cache = caches[strategy]
                    target_loader = iter(base.common.make_loader(engine.TargetViews(plan, args, transform, cfg, False), args, False, seed))
                offset, totals, seen = 0, np.zeros(4), Counter()
                for step, (clean, smooth, labels_s, _, _) in enumerate(tqdm(source_loader,
                        desc=f'{name} {strategy} {epoch+1}/{args.epochs}', mininterval=15, dynamic_ncols=True)):
                    base.common.seed_all(seed+step)
                    clean, smooth, labels_s = clean.to(args.device), smooth.to(args.device), labels_s.to(args.device)
                    n = len(labels_s)
                    optimizer.zero_grad(set_to_none=True)
                    with base.amp_context(args):
                        source_logits, _ = adaptation.outputs(name, model, torch.cat([clean, smooth]))
                        logits = torch.stack([source_logits[:n], source_logits[n:]], dim=1)
                        source_loss, ce, pair_rank, _ = rank.classification_and_ranking(logits, labels_s, args.ranking_weight)
                        loss, target_loss, weight = source_loss, source_loss.new_zeros(()), 0.
                        if target_loader is not None:
                            _, student_view = next(target_loader)
                            if len(student_view) != n:
                                raise ValueError('Source/target batch size differs')
                            rows = plan[offset:offset+n]
                            target_labels = torch.tensor([labels[r['crop_path']] for r in rows], device=args.device)
                            seen.update(int(y) for y in target_labels.tolist())
                            target_loss = nn.functional.cross_entropy(model(student_view.to(args.device)).float(), target_labels)
                            weight = args.target_weight*min(1., (epoch+(step+1)/len(source_loader))/args.ramp_epochs)
                            loss = source_loss+weight*target_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    totals += n*np.array([loss.item(), ce.item(), pair_rank.item(), target_loss.item()])
                    offset += n
                if plan is not None and offset != len(plan):
                    raise ValueError('Target/source exposure count differs')
                values = validate(args, model, val, adapt_val, transform, cfg)
                improved, improved_adapt = is_better(values['ffpp'], best), is_better(values['adapt_val'], best_adapt)
                if improved:
                    best = best_of(values['ffpp'])
                if improved_adapt:
                    best_adapt = best_of(values['adapt_val'])
                if values['ffpp']['pooled_auc'] > anchor+.0001:
                    anchor, stale = values['ffpp']['pooled_auc'], 0
                else:
                    stale += 1
                entry = dict(epoch=epoch+1, loss=float(totals[0]/offset), source_ce=float(totals[1]/offset),
                    source_rank=float(totals[2]/offset), target_ce=float(totals[3]/offset),
                    target_class_exposures={'real': seen[0], 'fake': seen[1]},
                    source_paths_sha256=evaluation.key(sampled.crop_path.tolist()), source_batches_sha256=evaluation.key(batches),
                    target_plan_sha256=evaluation.key(plan) if plan is not None else None, validation=values)
                history.append(entry)
                state = dict(identity=identity, config=config, model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch+1, stale=stale, history=history, best=best, best_adapt=best_adapt, anchor=anchor)
                for key, flag in [('ffpp', improved), ('adapt_val', improved_adapt)]:
                    if flag:
                        base.common.save(run/CHECKPOINTS[key], {k: state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch']}
                            | dict(selection=key, best=best_of(values[key]), validation=values))
                base.common.save(run/'last.pt', state)
                del state
                base.common.json_write(run/'history.json', history)
                f, t = values['ffpp'], values['adapt_val']
                print(f'CE={entry["source_ce"]:.4f} rank={entry["source_rank"]:.4f} targetCE={entry["target_ce"]:.4f} | '
                    f'FF++ pooled={f["pooled_auc"]:.4f} mean={f["mean_auc"]:.4f} | CDF adapt-val pooled={t["pooled_auc"]:.4f} '
                    f'cross(strong real/normal fake)={t["strong_real_vs_normal_fake_auc"]:.4f} | best FF++={best["pooled_auc"]:.4f} '
                    f'patience={stale}/{args.patience}', flush=True)
            selected = {key: torch.load(run/file, map_location='cpu', weights_only=True)['epoch'] for key, file in CHECKPOINTS.items()}
            base.common.json_write(run/'completed.json', dict(identity=identity, epochs_completed=len(history),
                selected_epoch=selected['ffpp'], selected_epoch_adapt_val=selected['adapt_val'],
                best_ffpp=best, best_adapt_val=best_adapt, inference_model='student', target_labels_used=target is not None))
            del model, optimizer, parameters, groups, head
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial


def subgroup_rows(frame, margins):
    kinds = frame.video_id.map(lambda v: v.split('/', 1)[0])
    y = frame.class_index.astype(int).to_numpy()
    row = {}
    for kind in ['Celeb-real', 'YouTube-real']:
        mask = (kinds == kind).to_numpy() & (y == 0)
        if mask.any():
            row[f'fpr_{kind}'] = float((margins[mask] >= 0).mean())
            row[f'auc_fake_vs_{kind}'] = fast_auc(np.r_[np.zeros(mask.sum()), np.ones((y == 1).sum())], np.r_[margins[mask], margins[y == 1]])
    return row


def video_bootstrap(frame, banks_a, banks_b, repeats, seed):
    """Resample whole videos within class; frames of a video stay together."""
    y = frame.class_index.astype(int).to_numpy()
    index = {}
    for i, video in enumerate(frame.video_id):
        index.setdefault(video, []).append(i)
    pools = [[np.array(index[v]) for v in sorted(index) if y[index[v][0]] == c] for c in [0, 1]]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        rows = np.concatenate([pool[j] for pool in pools for j in rng.integers(0, len(pool), len(pool))])
        labels = y[rows]
        values = []
        for banks in [banks_a, banks_b]:
            values.append((fast_auc(np.tile(labels, len(banks)), np.concatenate([b[rows] for b in banks])),
                           float(np.mean([fast_auc(labels, b[rows]) for b in banks]))))
        out.append([values[0][0]-values[1][0], values[0][1]-values[1][1]])
    return np.array(out)


def evaluate(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    frame, cfg, provenance = evaluation.inputs(ea)
    args.output_root.mkdir(parents=True, exist_ok=True)
    engine.freeze_json(args.output_root/'evaluation_protocol.json', dict(provenance, used_for_selection=False))
    engine.freeze_csv(args.output_root/'evaluation_manifest.csv', frame)
    view_cache = None
    if (args.cache_root/'config.json').is_file():
        if json.loads((args.cache_root/'config.json').read_text()) != provenance:
            raise ValueError('CDF view cache provenance differs')
        view_cache = args.cache_root
    native_cache = common.pixel_cache.BilateralCache(args.evaluation_pixel_cache, cfg['bilateral'],
        dict(data=provenance, cache_code_sha256=base.common.digest(Path(common.pixel_cache.__file__)),
             storage='native uint8 RGB bilateral only; lossless PNG'))
    labels = frame.class_index.astype(int).to_numpy()
    eval_args = copy.copy(args)
    eval_args.batch_size = args.eval_batch_size
    records, mixtures, cross, groups, selections, bootstrap = [], [], [], [], [], []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        states = [('baseline', initial, initial_hash)]
        for strategy in args.strategies:
            run = args.runs_root/run_name(name, strategy, args.seed)
            config = json.loads((run/'config.json').read_text())
            done = json.loads((run/'completed.json').read_text())
            if (done['identity'] != evaluation.key(config) or config['protocol'] != PROTOCOL or config['model'] != name
                    or config['initial_checkpoint_sha256'] != initial_hash or config['smoothing'] != cfg
                    or config['data']['test_manifest_sha256'] != provenance['manifest_sha256']
                    or config['code_sha256'] != code_hashes()):
                raise ValueError(f'Checkpoint/evaluation mismatch: {run}')
            for key, file in CHECKPOINTS.items():
                if key == 'adapt_val' and config['adapt_val_overlaps_training']:
                    continue
                state = torch.load(run/file, map_location='cpu', weights_only=True)
                if state['identity'] != done['identity'] or state['config'] != config:
                    raise ValueError(f'Checkpoint identity differs: {run/file}')
                label = strategy if key == 'ffpp' else f'{strategy}@adapt_val'
                states.append((label, state, base.common.digest(run/file)))
                selections.append(dict(model=name, strategy=label, selection=key, selected_epoch=state['epoch'],
                    epochs_completed=done['epochs_completed']))
        banks_by = {}
        for label, state, checkpoint_hash in states:
            identity = evaluation.key({'data': provenance, 'checkpoint': checkpoint_hash, 'precision': 'FP32',
                                       'code_sha256': code_hashes(), 'torch': str(torch.__version__)})
            model, banks = None, []
            for condition in CONDITIONS:
                prefix = args.output_root/f'{name}_{label}_{condition}'
                report, pred = prefix.with_suffix('.json'), prefix.with_suffix('.csv')
                if report.is_file() and pred.is_file():
                    values = json.loads(report.read_text())
                    saved = pd.read_csv(pred, dtype={'crop_path': str, 'video_id': str, 'class_index': str})
                    if (values['identity'] != identity or saved.crop_path.tolist() != frame.crop_path.tolist()
                            or saved.class_index.tolist() != frame.class_index.tolist() or not np.isfinite(saved.raw_margin).all()):
                        raise ValueError('Saved evaluation differs; use a new output root')
                    print('Reuse:', name, label, condition, flush=True)
                else:
                    previous = (engine.previous_baseline(args, name, condition, frame, provenance, checkpoint_hash)
                                if label == 'baseline' else None)
                    if label != 'baseline' and state['epoch'] == 0:
                        path = args.output_root/f'{name}_baseline_{condition}.csv'
                        previous = (pd.read_csv(path), path)
                    if previous is not None:
                        old, old_path = previous
                        probabilities, margins = old.probability_fake.to_numpy(), old.raw_margin.to_numpy()
                    else:
                        if model is None:
                            model, transform = base.build_model(name, pretrained=False)
                            model.load_state_dict(state['model_state_dict'], strict=True)
                            model.to(args.device).eval()
                        probabilities, margins = engine.infer(eval_args, model, frame,
                            engine.TestViews(frame, args, transform, cfg, condition, native_cache, view_cache), f'{name} {label} {condition}')
                    values = engine.raw_metrics(frame, probabilities, margins)
                    values.update(identity=identity, images=len(frame), precision='FP32', threshold=.5, checkpoint_sha256=checkpoint_hash,
                        manifest_sha256=provenance['manifest_sha256'], selected_epoch=state.get('epoch', 0), unit='frame')
                    if previous is not None:
                        values.update(reused_predictions=str(old_path), reused_predictions_sha256=base.common.digest(old_path))
                    saved = frame[['crop_path', 'video_id', 'class_index', 'roi_status']].copy()
                    saved['probability_fake'], saved['raw_margin'] = probabilities, margins
                    temp = pred.with_suffix('.tmp.csv'); saved.to_csv(temp, index=False); temp.replace(pred)
                    base.common.json_write(report, values)
                margins = saved.raw_margin.to_numpy()
                banks.append(margins)
                records.append(dict(model=name, strategy=label, condition=condition,
                    **{k: values[k] for k in ['real_fpr', 'fake_fnr', 'balanced_error_rate', 'auc', 'auc_probability']}))
                groups.append(dict(model=name, strategy=label, condition=condition, **subgroup_rows(frame, margins)))
                print(f'{name} {label} {condition}: FPR={values["real_fpr"]:.2%} FNR={values["fake_fnr"]:.2%} AUC={values["auc"]:.4f}', flush=True)
            summary = summarize(labels, banks, {c: r for c, r in zip(CONDITIONS, [r for r in records if r['model'] == name and r['strategy'] == label])})
            mixtures.append(dict(model=name, strategy=label, **{k: summary[k] for k in
                ['pooled_auc', 'mean_auc', 'mean_ber', 'min_cross_auc', 'strong_real_vs_normal_fake_auc']},
                mean_fpr=float(np.mean([r['real_fpr'] for r in records if r['model'] == name and r['strategy'] == label])),
                mean_fnr=float(np.mean([r['fake_fnr'] for r in records if r['model'] == name and r['strategy'] == label]))))
            for i, real_condition in enumerate(CONDITIONS):
                for j, fake_condition in enumerate(CONDITIONS):
                    cross.append(dict(model=name, strategy=label, real_condition=real_condition,
                                      fake_condition=fake_condition, auc=summary['cross_auc'][i][j]))
            banks_by[label] = banks
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        if 'ce_rank_pooled' in banks_by and args.bootstrap > 0:
            for label, banks in banks_by.items():
                if label == 'ce_rank_pooled':
                    continue
                diffs = video_bootstrap(frame, banks, banks_by['ce_rank_pooled'], args.bootstrap, args.seed)
                for k, metric in enumerate(['pooled_auc', 'mean_auc']):
                    observed = (fast_auc(np.tile(labels, 4), np.concatenate(banks))-fast_auc(np.tile(labels, 4), np.concatenate(banks_by['ce_rank_pooled']))
                                if metric == 'pooled_auc' else
                                float(np.mean([fast_auc(labels, b) for b in banks])-np.mean([fast_auc(labels, b) for b in banks_by['ce_rank_pooled']])))
                    bootstrap.append(dict(model=name, strategy=label, reference='ce_rank_pooled', metric=metric, observed=observed,
                        ci_low=float(np.quantile(diffs[:, k], .025)), ci_high=float(np.quantile(diffs[:, k], .975)),
                        fraction_not_positive=float((diffs[:, k] <= 0).mean()), repeats=args.bootstrap, unit='video cluster within class'))
        del states, initial
    table = pd.DataFrame(records)
    table.to_csv(args.output_root/'comparison.csv', index=False)
    table.groupby(['model', 'strategy'], sort=False)[['real_fpr', 'fake_fnr', 'balanced_error_rate', 'auc']].mean().reset_index().to_csv(
        args.output_root/'condition_means.csv', index=False)
    pd.DataFrame(mixtures).to_csv(args.output_root/'mixture_summary.csv', index=False)
    pd.DataFrame(cross).to_csv(args.output_root/'cross_condition_auc.csv', index=False)
    pd.DataFrame(groups).to_csv(args.output_root/'subgroups.csv', index=False)
    pd.DataFrame(selections).to_csv(args.output_root/'selections.csv', index=False)
    pd.DataFrame(bootstrap).to_csv(args.output_root/'bootstrap_vs_ce_rank_pooled.csv', index=False)
    base.common.json_write(args.output_root/'comparison_protocol.json', dict(protocol=PROTOCOL,
        primary_selection='FF++ val pooled AUC', diagnostic_selection='CDF pool-v2 adapt-val pooled AUC (labelled non-test)',
        test_used_for_selection=False, bootstrap='test sampling variance only; not seed variance', strategies=args.strategies))
    print('Saved:', args.output_root, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'train', 'evaluate'])
    for field in ['source-root', 'target-root', 'baseline-runs-root', 'runs-root', 'manifest-root', 'validation-cache', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    for field in ['source-validation', 'hard-manifest-root', 'reference-evaluations']:
        p.add_argument('--'+field, type=Path)
    for field, default in [('train-cache', '/content/smoothing_pivot_bilateral_cache_v1'),
                          ('target-train-cache', '/content/smoothing_rank_sda_pool_v2_target_cache'),
                          ('cache-root', '/content/celebdf_official_smoothing_cache_v1'),
                          ('evaluation-pixel-cache', '/content/smoothing_ranking_cdf_bilateral_cache_v1')]:
        p.add_argument('--'+field, type=Path, default=Path(default))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['dinov2_b14'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=['ce_rank_pooled', 'sda_new_balanced'])
    for field, default in [('seed', 42), ('epochs', 10), ('patience', 3), ('source-images', 6000), ('val-images', 1200),
                          ('target-images', 5000), ('batch-size', 16), ('eval-batch-size', 32), ('workers', 2),
                          ('family-cap', 3), ('bootstrap', 1000)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr', 1e-5), ('head-lr', 1e-5), ('weight-decay', 1e-4), ('ranking-weight', .25),
                          ('target-weight', .5), ('ramp-epochs', 2.), ('adapt-val-fraction', .2)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    numbers = [a.encoder_lr, a.head_lr, a.weight_decay, a.ranking_weight, a.target_weight, a.ramp_epochs, a.adapt_val_fraction]
    if (not np.isfinite(numbers).all() or min(a.encoder_lr, a.head_lr, a.ramp_epochs) <= 0
            or min(a.weight_decay, a.ranking_weight, a.target_weight) < 0 or not 0 < a.adapt_val_fraction < 1
            or min(a.epochs, a.patience, a.source_images, a.val_images, a.target_images, a.batch_size, a.family_cap) < 1
            or a.batch_size % 2 or a.eval_batch_size < 1 or a.source_images % 8 or a.val_images % 8
            or a.source_images % a.batch_size or a.workers < 0 or a.bootstrap < 0
            or len(set(a.models)) != len(a.models) or len(set(a.strategies)) != len(a.strategies)):
        p.error('Invalid settings or duplicate models/strategies')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    arguments = parse_args()
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[arguments.action](arguments)
