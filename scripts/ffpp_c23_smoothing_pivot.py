"""Matched Aug vs class-conditional smoothing-score adversary; source-only training."""
import argparse
import json
import math
import random
from importlib import metadata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

import ffpp_c23_smoothing_adaptation as adaptation
import ffpp_c23_smoothing_preservation as shared
import evaluate_ffpp_baselines_celebdf_smoothing as evaluation
import train_ffpp_c23_baseline as base
import smoothing_pixel_cache as pixel_cache

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_pivot_v1'
STRATEGIES = ['augmentation', 'conditional_pivot']
CONDITIONS = ['normal', 'smooth_weak', 'smooth_medium', 'smooth_strong']
# Exact previous released trainer. Only the pixel-identical cache optimization is compatible.
LEGACY_TRAINER_SHA256 = 'b6ca3918f14ec866ff4746a993082c84b81ce110b0eca089a162322b3b562125'
LEGACY_SMOOTHING_SHA256 = 'eb32c7e7f5e116ecf269732a2d83b64f3a86b30bcc6f60a643c64f7cce443e70'


def compatible_config(run, proposed):
    path=run/'config.json'
    if not path.is_file():
        return proposed
    saved=json.loads(path.read_text())
    if saved==proposed:
        return saved
    normalized=dict(saved,code_sha256=proposed['code_sha256'])
    if saved.get('code_sha256')==LEGACY_TRAINER_SHA256 and normalized==proposed:
        print('Compatible speed optimization: preserving original config/checkpoint identity',flush=True)
        return saved
    raise ValueError(f'Run settings differ: {run}; only the verified cache-only trainer change is compatible')


def prepare_pixel_cache(args, cfg, data):
    args._pixel_cache=None
    root=getattr(args,'train_cache',None)
    if root is None:
        return
    digest=base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py')
    if digest!=LEGACY_SMOOTHING_SHA256:
        raise ValueError('Smoothing implementation changed; pixel-equivalent optimization must be reverified')
    identity={'source_trainval_sha256':data['source_trainval_sha256'],
        'bilateral':cfg['bilateral'],'opencv':evaluation.cv2.__version__,'smoothing_code_sha256':digest,
        'cache_code_sha256':base.common.digest(Path(pixel_cache.__file__)),
        'storage':'native uint8 RGB bilateral result only; lossless PNG; blend/flip/resize unchanged'}
    args._pixel_cache=pixel_cache.BilateralCache(root,cfg['bilateral'],identity)
    print('Shared native bilateral cache:',root,'| first encounters compute, later epochs/strategies reuse',flush=True)


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, strength):
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class ConditionalCritic(nn.Module):
    """Training-only heads: predict strength from real/fake margin, separately by GT class."""
    def __init__(self, hidden=32):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 4))
            for _ in range(2)])

    def forward(self, margin, labels, reversal):
        value = Reverse.apply(margin[:, None] / 10., reversal)
        outputs = torch.stack([head(value) for head in self.heads], dim=1)
        return outputs[torch.arange(len(labels), device=labels.device), labels]


def critic_loss(critic, logits, labels, strengths, detected, reversal):
    # Fallbacks have no actual smoothing. They retain CE but are excluded here.
    valid = detected.bool()
    if not valid.any():
        return logits.sum() * 0., logits.new_zeros(())
    margin = logits[valid, 1].float() - logits[valid, 0].float()
    outputs = critic(margin, labels[valid], reversal)
    target = strengths[valid]
    # Each pair is clean + uniform weak/medium/strong: priors (1/2,1/6,1/6,1/6).
    # Fixed inverse-prior weighting gives each strength equal expected weight.
    weights = outputs.new_tensor([.5, 1.5, 1.5, 1.5])[target]
    loss = (nn.functional.cross_entropy(outputs, target, reduction='none') * weights).mean()
    correct = ((outputs.argmax(1) == target).float() * weights).sum() / weights.sum()
    return loss, correct


def run_name(model, strategy, seed):
    return f'{model}_{strategy}_ffpp_c23_smoothing_pivot_v1_seed{seed}'


def training_mode(model, name, blocks):
    base.stage(model, name, blocks, False)
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()


class PairedSource(Dataset):
    def __init__(self, frame, args, transform, cfg, epoch):
        self.rows = frame.to_dict('records')
        self.args, self.transform, self.cfg, self.epoch = args, transform, cfg, epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rng = random.Random(self.args.seed + self.epoch * 100003 + index)
        flip, strength = rng.random() < .5, rng.randrange(1, 4)
        with Image.open(self.args.source_root / row['crop_path']) as opened:
            clean = opened.convert('RGB')
        detected = row['roi_status'] == 'detected'
        if not detected:
            changed=clean.copy()
        elif getattr(self.args,'_pixel_cache',None) is not None:
            changed=self.args._pixel_cache.smooth(clean,row['crop_path'],self.cfg['conditions'][CONDITIONS[strength]])
        else:
            changed=adaptation.smooth(clean,self.cfg['conditions'][CONDITIONS[strength]],self.cfg['bilateral'])
        if flip:
            clean = clean.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            changed = changed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.transform(clean), self.transform(changed), int(row['class_index']), strength, detected


def prepare_validation(args, frame, cfg, data):
    cache = args.validation_cache
    cache.mkdir(parents=True, exist_ok=True)
    identity = {'data': data, 'smoothing': cfg, 'opencv': evaluation.cv2.__version__,
                'smoothing_code': base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py')}
    path = cache / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Validation cache differs; use a new --validation-cache')
    base.common.json_write(path, identity)

    def one(row):
        if row['roi_status'] != 'detected':
            return
        missing = [c for c in CONDITIONS[1:] if not (cache/c/row['crop_path']).is_file()]
        if not missing:
            return
        with Image.open(args.source_root/row['crop_path']) as opened:
            image = opened.convert('RGB')
        native_cache=getattr(args,'_pixel_cache',None)
        filtered=(native_cache.filtered(image,row['crop_path']) if native_cache is not None
                  else pixel_cache.bilateral(image,cfg['bilateral']))
        for condition in missing:
            dest = cache/condition/row['crop_path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix('.tmp.png')
            pixel_cache.blend(image,filtered,cfg['conditions'][condition]).save(temp,format='PNG',compress_level=1)
            temp.replace(dest)

    with ThreadPoolExecutor(max_workers= max(1, args.workers)) as executor:
        for _ in tqdm(executor.map(one, frame.to_dict('records')), total=len(frame),
                      desc='Validation smoothing cache (once)', mininterval=10):
            pass


class ValidationViews(Dataset):
    def __init__(self, frame, args, transform, condition):
        self.rows, self.args, self.transform, self.condition = frame.to_dict('records'), args, transform, condition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        root = self.args.source_root
        if self.condition != 'normal' and row['roi_status'] == 'detected':
            root = self.args.validation_cache/self.condition
        with Image.open(root/row['crop_path']) as opened:
            return self.transform(opened.convert('RGB')), index


def metrics(frame, scores):
    result = base.binary_metrics(frame.class_index.astype(int).to_numpy(), scores)
    result['balanced_error_rate'] = .5*(result['real_fpr']+result['fake_fnr'])
    return result


def validate(args, model, frame, transform):
    model.eval()
    records = {}
    for condition in CONDITIONS:
        scores, margins = np.empty(len(frame)), np.empty(len(frame))
        loader = base.common.make_loader(ValidationViews(frame, args, transform, condition), args)
        with torch.inference_mode():
            for images, indexes in tqdm(loader, desc=f'Val {condition}', mininterval=20, dynamic_ncols=True):
                logits = model(images.to(args.device)).float()  # FP32, no autocast.
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Nonfinite validation logits')
                scores[indexes.numpy()] = logits.softmax(1)[:, 1].cpu().numpy()
                margins[indexes.numpy()] = (logits[:, 1]-logits[:, 0]).cpu().numpy()
        result = metrics(frame, scores)
        result['class_mean_margin'] = {
            str(label): float(margins[frame.class_index.astype(int).to_numpy() == label].mean())
            for label in [0, 1]}
        records[condition] = result
    return {'mean_ber': float(np.mean([x['balanced_error_rate'] for x in records.values()])),
            'conditions': records}


def configuration(args, name, strategy, data, cfg, initial, initial_hash):
    versions={}
    for package in ['torch','torchvision','timm','Pillow','numpy','pandas','scikit-learn',
                    'opencv-python-headless','clip','transformers','pilgram2']:
        try:
            versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package]=None
    return {'protocol': PROTOCOL, 'model': name, 'strategy': strategy, 'seed': args.seed,
            'data': data, 'smoothing': cfg, 'preprocessing': base.PREPROCESS[name],
            'initial_checkpoint_sha256': initial_hash, 'train_blocks': initial['config']['train_blocks'],
            'source_images_per_epoch': args.source_images, 'val_images': args.val_images,
            'epochs': args.epochs, 'patience': args.patience, 'min_delta': .0001,
            'batch_size': args.batch_size, 'encoder_lr': args.encoder_lr, 'head_lr': args.head_lr,
            'critic_lr': args.critic_lr, 'weight_decay': args.weight_decay, 'hidden': args.hidden,
            'adversarial_weight': args.adversarial_weight if strategy == 'conditional_pivot' else 0.,
            'margin_divisor': 10., 'strength_loss_weights': [.5, 1.5, 1.5, 1.5],
            'loss': 'mean clean+random-smoothing CE; candidate adds class-conditioned strength CE with encoder gradient reversal, linear 2-epoch ramp; fallback excluded only from critic',
            'reversal_warmup_epochs': 2,
            'selection': 'minimum four-condition FF++ validation mean BER @0.5, epoch0 eligible, no guards',
            'trainable': 'same as previous Aug: Xception all; CLIP/DINO last baseline-configured blocks plus classifier',
            'target_training': False, 'batchnorm_running_statistics': 'frozen',
            'precision': 'BF16 if available else FP32 training; FP32 validation/test',
            'torch': str(torch.__version__), 'opencv': evaluation.cv2.__version__,
            'versions': versions, 'baseline_versions': initial['config'].get('versions', {}),
            'code_sha256': base.common.digest(Path(__file__)),
            'shared_code_sha256': {p: base.common.digest(ROOT/'scripts'/p) for p in [
                'train_ffpp_c23_baseline.py', 'train_celebdf_baseline.py',
                'ffpp_c23_smoothing_adaptation.py', 'ffpp_c23_smoothing_preservation.py',
                'smoothing_roi_pilot.py', 'evaluate_ffpp_baselines_celebdf_smoothing.py']}}


def train(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    pool, val, data = shared.source_inputs(args)
    cfg = json.loads(args.protocol.read_text())
    if list(cfg['conditions']) != CONDITIONS:
        raise ValueError('Expected frozen normal/weak/medium/strong smoothing protocol')
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    frozen = args.manifest_root/'protocol.json'
    value = {'protocol': PROTOCOL, 'data': data, 'seed': args.seed,
             'source_images_per_epoch': args.source_images, 'val_images': args.val_images}
    if frozen.exists() and json.loads(frozen.read_text()) != value:
        raise ValueError('Manifest settings differ; use a new manifest root')
    base.common.json_write(frozen, value)
    val.to_csv(args.manifest_root/'source_validation.csv', index=False)
    prepare_pixel_cache(args,cfg,data)
    prepare_validation(args, val, cfg, data)
    print(f'Source={args.source_images}/epoch, validation={len(val)} x4; source-only training', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Baseline source manifest mismatch')
        for strategy in args.strategies:
            config = configuration(args, name, strategy, data, cfg, initial, initial_hash)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            config=compatible_config(run,config)
            identity = evaluation.key(config)
            if (run/'completed.json').is_file():
                done = json.loads((run/'completed.json').read_text())
                if done['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError('Completed run mismatch')
                print('Already completed:', name, strategy, flush=True)
                continue
            if not (run/'config.json').is_file():
                base.common.json_write(run/'config.json', config)
            base.common.json_write(run/'runtime_optimization.json',{
                'learning_identity_preserved':identity,'executed_code_sha256':base.common.digest(Path(__file__)),
                'cache_helper_sha256':base.common.digest(Path(pixel_cache.__file__)),
                'train_cache':str(getattr(args,'train_cache',None)),
                'behavior':'pixel-identical native bilateral cache; same seeds, views, losses, optimizer and selection'})
            base.common.seed_all(args.seed)
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(initial['model_state_dict'], strict=True)
            model.to(args.device)
            blocks = initial['config']['train_blocks'] if name != 'xception' else 2
            training_mode(model, name, blocks)
            critic = ConditionalCritic(args.hidden).to(args.device)
            head_params = list(base.head(model, name).parameters())
            head_ids = {id(p) for p in head_params}
            groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr': args.encoder_lr},
                      {'params': head_params, 'lr': args.head_lr}]
            if strategy == 'conditional_pivot':
                groups.append({'params': list(critic.parameters()), 'lr': args.critic_lr})
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            parameters = [p for group in groups for p in group['params']]
            start, stale, history = 0, 0, []
            last = run/'last.pt'
            if last.is_file():
                state = torch.load(last, map_location='cpu', weights_only=True)
                if state['identity'] != identity:
                    raise ValueError('Resume identity mismatch')
                model.load_state_dict(state['model_state_dict'], strict=True)
                critic.load_state_dict(state['critic_state_dict'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start, stale, history, best, anchor = [state[k] for k in ['epoch', 'stale', 'history', 'best', 'anchor']]
                del state
                print('Resume completed epochs:', start, flush=True)
            else:
                reference = validate(args, model, val, transform)
                best = anchor = reference['mean_ber']
                base.common.json_write(run/'epoch0_validation.json', reference)
                base.common.save(run/'best.pt', {'identity': identity, 'config': config,
                    'model_state_dict': model.state_dict(), 'epoch': 0, 'best': best, 'validation': reference})
            print(f'===== {name} / {strategy} / epoch0 eligible / mean BER =====', flush=True)
            for epoch in range(start, args.epochs):
                if stale >= args.patience:
                    break
                epoch_seed = args.seed+epoch*100003
                base.common.seed_all(epoch_seed)
                training_mode(model, name, blocks)
                critic.train()
                sampled = adaptation.balanced_source(pool, args.source_images, epoch_seed)
                dataset = PairedSource(sampled, args, transform, cfg, epoch)
                loader = base.common.make_loader(dataset, args, True, epoch_seed)
                totals, count = np.zeros(3), 0
                for step, (clean, changed, labels, strength, detected) in enumerate(tqdm(
                        loader, desc=f'{name} {strategy} epoch{epoch+1}/{args.epochs}', mininterval=15, dynamic_ncols=True)):
                    clean, changed, labels = clean.to(args.device), changed.to(args.device), labels.to(args.device)
                    labels2 = labels.repeat(2)
                    strength = torch.cat([torch.zeros_like(strength), strength]).to(args.device)
                    detected = detected.repeat(2).to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    ramp = min(1., (epoch+(step+1)/len(loader))/2.)
                    with base.amp_context(args):
                        logits = model(torch.cat([clean, changed]))
                        ce = nn.functional.cross_entropy(logits.float(), labels2)
                    adv, accuracy = logits.float().sum()*0., logits.new_zeros((), dtype=torch.float32)
                    if strategy == 'conditional_pivot':
                        adv, accuracy = critic_loss(critic, logits.float(), labels2, strength,
                                                    detected, args.adversarial_weight*ramp)
                    loss = ce+adv
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                    if strategy == 'conditional_pivot':
                        nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    n = len(labels)
                    totals += np.array([ce.detach().item(), adv.detach().item(), accuracy.detach().item()])*n
                    count += n
                values = validate(args, model, val, transform)
                score = values['mean_ber']
                improved = score < best
                best = min(best, score)
                if score < anchor-.0001:
                    anchor, stale = score, 0
                else:
                    stale += 1
                history.append({'epoch': epoch+1, 'ce': float(totals[0]/count),
                    'critic_ce': float(totals[1]/count), 'critic_weighted_accuracy': float(totals[2]/count),
                    'reversal_max_this_epoch': args.adversarial_weight*ramp if strategy == 'conditional_pivot' else 0.,
                    'epoch_sample_sha256': evaluation.key(sampled.crop_path.tolist()),
                    'validation': values})
                state = {'identity': identity, 'config': config, 'model_state_dict': model.state_dict(),
                    'critic_state_dict': critic.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch+1, 'best': best, 'anchor': anchor, 'stale': stale, 'history': history}
                if improved:
                    base.common.save(run/'best.pt', {k: state[k] for k in ['identity','config','model_state_dict','epoch','best']} | {'validation': values})
                base.common.save(last, state)
                del state
                base.common.json_write(run/'history.json', history)
                print(f'CE={totals[0]/count:.4f} critic={totals[1]/count:.4f} | val BER={score:.2%} '
                      f'best={best:.2%} patience={stale}/{args.patience}', flush=True)
            selected = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            base.common.json_write(run/'completed.json', {'identity': identity, 'epochs_completed': len(history),
                'best_epoch': selected['epoch'], 'best_mean_ber': best})
            del selected, model, critic, optimizer, groups, parameters, head_params
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial


def mixture_summary(args, frame, records):
    """Equal-condition mixture; correlated views are not independent test samples."""
    pooled, cross = [], []
    labels = frame.class_index.astype(int).to_numpy()
    for name, strategy in dict.fromkeys((r['model'],r['strategy']) for r in records):
        banks=[]
        for condition in CONDITIONS:
            path = (args.baseline_evaluation_root/f'{name}_{condition}.csv' if strategy=='baseline'
                    else args.output_root/f'{name}_{strategy}_{condition}.csv')
            if not path.is_file():
                banks=[]
                print('Mixture summary skipped (no saved predictions):',name,strategy,flush=True)
                break
            saved=pd.read_csv(path,dtype={'crop_path':str,'class_index':str,'video_id':str})
            if (saved.crop_path.tolist()!=frame.crop_path.tolist()
                    or saved.class_index.tolist()!=frame.class_index.tolist()
                    or saved.video_id.tolist()!=frame.video_id.tolist()
                    or not np.isfinite(saved.probability_fake).all()
                    or not saved.probability_fake.between(0,1).all()):
                raise ValueError('Mixture prediction membership/probabilities mismatch')
            banks.append(saved.probability_fake.to_numpy())
        if not banks:
            continue
        result=base.binary_metrics(np.tile(labels,4),np.concatenate(banks))
        result['balanced_error_rate']=.5*(result['real_fpr']+result['fake_fnr'])
        diagonal=np.mean([roc_auc_score(labels,scores) for scores in banks])
        pooled.append({'model':name,'strategy':strategy,'condition_mean_auc':float(diagonal),
            'pooled_auc':result['auc'],'pooled_minus_condition_mean_auc':result['auc']-diagonal,
            'real_fpr':result['real_fpr'],'fake_fnr':result['fake_fnr'],'balanced_error_rate':result['balanced_error_rate']})
        for i,real_condition in enumerate(CONDITIONS):
            for j,fake_condition in enumerate(CONDITIONS):
                scores=np.concatenate([banks[i][labels==0],banks[j][labels==1]])
                y=np.concatenate([np.zeros((labels==0).sum()),np.ones((labels==1).sum())])
                cross.append({'model':name,'strategy':strategy,'real_condition':real_condition,
                              'fake_condition':fake_condition,'auc':float(roc_auc_score(y,scores))})
    pd.DataFrame(pooled).to_csv(args.output_root/'mixture_summary.csv',index=False)
    pd.DataFrame(cross).to_csv(args.output_root/'cross_condition_auc.csv',index=False)


def evaluate(args):
    if args.target_root is None or args.output_root is None:
        raise ValueError('evaluate requires --target-root and --output-root')
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    evaluation.prepare(ea)
    frame, cfg, provenance = evaluation.inputs(ea)
    records = []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        del initial
        # Existing canonical baseline is optional; validate it before adding reference rows.
        if args.baseline_evaluation_root and (args.baseline_evaluation_root/'evaluation_protocol.json').is_file():
            old_protocol = json.loads((args.baseline_evaluation_root/'evaluation_protocol.json').read_text())
            if any(old_protocol.get(k) != v for k,v in provenance.items()):
                raise ValueError('Baseline evaluation protocol differs')
            for condition in CONDITIONS:
                old = json.loads((args.baseline_evaluation_root/f'{name}_{condition}.json').read_text())
                if old['checkpoint_sha256'] != initial_hash or old['manifest_sha256'] != provenance['manifest_sha256']:
                    raise ValueError('Baseline checkpoint/manifest mismatch')
                records.append({'model':name, 'strategy':'baseline', 'condition':condition,
                    **{k:old[k] for k in ['auc','real_fpr','fake_fnr']},
                    'balanced_error_rate': .5*(old['real_fpr']+old['fake_fnr'])})
        for strategy in args.strategies:
            run = args.runs_root/run_name(name, strategy, args.seed)
            done = json.loads((run/'completed.json').read_text())
            state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            config = json.loads((run/'config.json').read_text())
            if (state['config'] != config or state['identity'] != evaluation.key(config)
                    or done['identity'] != state['identity'] or config['protocol'] != PROTOCOL
                    or config['model'] != name or config['strategy'] != strategy or config['seed'] != args.seed
                    or config['initial_checkpoint_sha256'] != initial_hash or config['smoothing'] != cfg
                    or config['preprocessing'] != base.PREPROCESS[name]):
                raise ValueError('Checkpoint/evaluation protocol mismatch')
            checkpoint_hash = base.common.digest(run/'best.pt')
            identity_fields={'data':provenance,'checkpoint':checkpoint_hash,'precision':'FP32',
                             'torch':str(torch.__version__),'code':base.common.digest(Path(__file__))}
            identity=evaluation.key(identity_fields)
            compatible_evaluations={identity,evaluation.key(dict(identity_fields,code=LEGACY_TRAINER_SHA256))}
            model, transform = None, None
            for condition in CONDITIONS:
                prefix = args.output_root/f'{name}_{strategy}_{condition}'
                report, pred = prefix.with_suffix('.json'), prefix.with_suffix('.csv')
                values = None
                if report.is_file() and pred.is_file():
                    values = json.loads(report.read_text())
                    saved = pd.read_csv(pred, dtype={'crop_path':str, 'class_index':str})
                    if (values['identity'] not in compatible_evaluations or saved.crop_path.tolist() != frame.crop_path.tolist()
                            or saved.class_index.tolist() != frame.class_index.tolist()
                            or not np.isfinite(saved.probability_fake).all()
                            or not saved.probability_fake.between(0,1).all()):
                        raise ValueError('Saved evaluation differs; use a new output root')
                    print('Reuse:', name, strategy, condition, flush=True)
                if values is None:
                    if model is None:
                        model, transform = base.build_model(name, pretrained=False)
                        model.load_state_dict(state['model_state_dict'], strict=True)
                        model.to(args.device).eval()
                    loader = base.common.make_loader(evaluation.Views(frame, ea, transform, condition), args)
                    scores, margins = np.empty(len(frame)), np.empty(len(frame))
                    with torch.inference_mode():
                        for images,indexes in tqdm(loader, desc=f'{name} {strategy} {condition}', mininterval=15, dynamic_ncols=True):
                            logits = model(images.to(args.device)).float()
                            if not torch.isfinite(logits).all():
                                raise RuntimeError('Nonfinite test logits')
                            scores[indexes.numpy()] = logits.softmax(1)[:,1].cpu().numpy()
                            margins[indexes.numpy()] = (logits[:,1]-logits[:,0]).cpu().numpy()
                    saved = frame[['crop_path','video_id','class_index','roi_status']].copy()
                    saved['probability_fake'], saved['raw_margin'] = scores, margins
                    temp = pred.with_suffix('.tmp.csv'); saved.to_csv(temp,index=False); temp.replace(pred)
                    values = metrics(frame,scores)
                    values.update(identity=identity, images=len(frame), precision='FP32', threshold=.5,
                                  checkpoint_sha256=checkpoint_hash, best_epoch=state['epoch'], unit='frame')
                    base.common.json_write(report,values)
                records.append({'model':name, 'strategy':strategy, 'condition':condition,
                    **{k:values[k] for k in ['auc','real_fpr','fake_fnr','balanced_error_rate']}})
                print(f'{name} {strategy} {condition}: BER={values["balanced_error_rate"]:.2%} '
                      f'FPR={values["real_fpr"]:.2%} FNR={values["fake_fnr"]:.2%} AUC={values["auc"]:.4f}',flush=True)
            del state, model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    table=pd.DataFrame(records)
    table.to_csv(args.output_root/'comparison.csv',index=False)
    table.groupby(['model','strategy'],sort=False)[['auc','real_fpr','fake_fnr','balanced_error_rate']].mean().reset_index().to_csv(
        args.output_root/'condition_means.csv',index=False)
    mixture_summary(args,frame,records)
    print('Saved:',args.output_root,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate'])
    for field in ['source-root','baseline-runs-root','runs-root','manifest-root','validation-cache']:
        p.add_argument('--'+field,type=Path,required=True)
    for field in ['target-root','output-root','baseline-evaluation-root','source-validation']:
        p.add_argument('--'+field,type=Path)
    p.add_argument('--cache-root',type=Path,default=Path('/content/celebdf_official_smoothing_cache_v1'))
    p.add_argument('--train-cache',type=Path,default=Path('/content/smoothing_pivot_bilateral_cache_v1'))
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models',nargs='+',choices=list(base.PREPROCESS),default=['clip_b16'])
    p.add_argument('--strategies',nargs='+',choices=STRATEGIES,default=STRATEGIES)
    for field,default in [('seed',42),('epochs',10),('patience',3),('source-images',6000),('val-images',1200),
                          ('batch-size',16),('workers',2),('hidden',32)]:
        p.add_argument('--'+field,type=int,default=default)
    for field,default in [('encoder-lr',1e-5),('head-lr',1e-5),('critic-lr',1e-3),
                          ('weight-decay',1e-4),('adversarial-weight',.1)]:
        p.add_argument('--'+field,type=float,default=default)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args()
    if (min(args.epochs,args.patience,args.source_images,args.val_images,args.batch_size,args.hidden)<1
            or args.workers<0 or args.adversarial_weight<0 or args.source_images%8 or args.val_images%8):
        p.error('Positive counts, counts divisible by8, nonnegative workers/adversarial weight required')
    (train if args.action=='train' else evaluate)(args)


if __name__=='__main__':
    main()
