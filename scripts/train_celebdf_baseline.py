"""Supervised native-crop baselines: ImageNet Xception and OpenAI CLIP ViT-B/16.

Shared custom identity split, epoch-balanced source data, clean validation.
No imported FF++ detector, smoothing, JPEG augmentation or UDA in training.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from create_celebdf_identity_split import audit


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save(path, state):
    temp = path.with_suffix('.tmp.pt')
    torch.save(state, temp)
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClipClassifier(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.head = nn.Linear(visual.output_dim, 2)

    def forward(self, images):
        # Raw image-encoder features: no text encoder or extra feature L2 normalization.
        return self.head(self.visual(images))


def model_and_transform(name, pretrained=True):
    if name == 'xception':
        import timm
        model = timm.create_model('legacy_xception', pretrained=pretrained, num_classes=2)
        transform = transforms.Compose([
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(), transforms.Normalize((.5,)*3, (.5,)*3)])
    else:
        import clip
        if pretrained:
            full, transform = clip.load('ViT-B/16', device='cpu', jit=False)
            visual = full.visual.float()
        else:
            from clip.model import VisionTransformer
            visual = VisionTransformer(224, 16, 768, 12, 12, 512)
            transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
        model = ClipClassifier(visual)
    return model, transform


class Crops(Dataset):
    def __init__(self, frame, root, transform, training=False, epoch_seed=42):
        self.rows = frame.to_dict('records')
        self.root, self.transform = root, transform
        self.training, self.seed = training, epoch_seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['crop_path']) as f:
            image = f.convert('RGB')
            if self.training and random.Random(self.seed + index * 100003).random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            x = self.transform(image)
        return x, int(row['class_index']), index


def balanced_epoch(frame, seed):
    real, fake = [frame[frame.class_index.astype(int).eq(c)] for c in [0, 1]]
    n = min(len(real), len(fake))
    if n == 0:
        raise ValueError('Training needs both classes')
    return pd.concat([g.sample(n=n, random_state=seed+c) for c, g in enumerate([real, fake])],
                     ignore_index=True).sample(frac=1, random_state=seed+2).reset_index(drop=True)


def make_loader(dataset, args, training=False, seed=42):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                      num_workers=args.workers, pin_memory=args.device == 'cuda', persistent_workers=False,
                      generator=torch.Generator().manual_seed(seed))


def metrics(frame, probabilities):
    def calculate(y, p):
        pred = p >= .5
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {'auc': float(roc_auc_score(y, p)), 'accuracy': float(accuracy_score(y, pred)),
                'f1': float(f1_score(y, pred, zero_division=0)), 'real_fpr': float(fp/(tn+fp)),
                'fake_fnr': float(fn/(tp+fn)), 'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}
    out = frame[['video_id', 'class_index', 'crop_path', 'roi_status']].copy()
    out['probability_fake'] = probabilities
    videos = out.groupby('video_id', sort=True).agg(class_index=('class_index', 'first'),
                                                 probability_fake=('probability_fake', 'mean'))
    return {'frame': calculate(out.class_index.astype(int), out.probability_fake.to_numpy()),
            'video': calculate(videos.class_index.astype(int), videos.probability_fake.to_numpy()),
            'images': len(out), 'videos': len(videos), 'threshold': .5}, out


def evaluate(model, loader, frame, args, title):
    model.eval()
    p = np.empty(len(frame), dtype=np.float64)
    loss_sum, count, last = 0., 0, time.monotonic()
    with torch.inference_mode():
        for step, (x, y, indexes) in enumerate(loader, 1):
            x, y = x.to(args.device), y.to(args.device)
            with torch.autocast(device_type=args.device, enabled=args.device == 'cuda'):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            p[indexes.numpy()] = logits.float().softmax(1)[:, 1].cpu().numpy()
            loss_sum += float(loss)*len(y)
            count += len(y)
            if time.monotonic()-last >= args.log_interval_seconds or step == len(loader):
                print(f'{title}: {step}/{len(loader)}', flush=True)
                last = time.monotonic()
    values, predictions = metrics(frame, p)
    values['loss'] = loss_sum/count
    return values, predictions


def load_manifest(args):
    cfg = json.loads((args.manifest_root / 'protocol.json').read_text())
    manifest = args.manifest_root / 'manifest.csv'
    if digest(manifest) != cfg['manifest_sha256']:
        raise RuntimeError('Frozen split manifest hash mismatch')
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    audit(frame)
    return frame, cfg


def run_identity(args, split):
    lr = args.learning_rate or (1e-4 if args.model == 'xception' else 1e-5)
    config = {'model': args.model, 'seed': args.seed, 'split': split, 'epochs': args.epochs,
              'patience': args.patience, 'batch_size': args.batch_size, 'learning_rate': lr,
              'head_learning_rate': lr if args.model == 'xception' else args.head_learning_rate,
              'weight_decay': args.weight_decay, 'augmentation': 'train horizontal flip probability0.5 only',
              'training': 'epoch balanced without replacement: all minority-class frames + random equally many majority-class frames',
              'loss': 'unweighted cross entropy', 'checkpoint_selection': 'maximum clean validation video AUC; min_delta0.0001 for early stopping',
              'initial': 'ImageNet legacy_xception' if args.model == 'xception' else 'OpenAI CLIP ViT-B/16',
              'encoder': 'all trainable; no FF++ detection checkpoint; no text encoder',
              'preprocessing': 'native frozen crop -> bicubic299 -> Normalize0.5' if args.model == 'xception'
                    else 'native frozen crop -> official CLIP bicubic224/center crop -> CLIP normalization',
              'source_code_sha256': digest(Path(__file__)),
              'split_code_sha256': digest(Path(__file__).with_name('create_celebdf_identity_split.py')),
              'packages': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'timm', 'Pillow', 'pandas', 'scikit-learn']}}
    if args.model == 'clip_b16':
        config['packages']['clip'] = importlib.metadata.version('clip')
    return config, hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def train(args):
    frame, split = load_manifest(args)
    cfg, identity = run_identity(args, split)
    run = args.output_root / f'{args.model}_celebdf_identity_v1_seed{args.seed}'
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise RuntimeError('Existing run config differs; preserve run and use new output root')
    if (run / 'completed.json').exists():
        if json.loads((run / 'completed.json').read_text())['identity'] != identity or not (run/'best.pt').is_file():
            raise RuntimeError('Completed run identity/checkpoint mismatch')
        print('Already completed:', run, flush=True)
        return
    if config_path.exists() and not (run/'last.pt').is_file() and (run/'best.pt').exists():
        raise RuntimeError('Incomplete checkpoint set: preserve run, inspect before restarting')
    json_write(config_path, cfg)
    seed_all(args.seed)
    resuming = (run/'last.pt').is_file()
    model, transform = model_and_transform(args.model, pretrained=not resuming)
    model.to(args.device)
    params = model.parameters() if args.model == 'xception' else [
        {'params': model.visual.parameters(), 'lr': cfg['learning_rate']},
        {'params': model.head.parameters(), 'lr': cfg['head_learning_rate']}]
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device == 'cuda')
    start, best, anchor, stale, history = 0, -1., -1., 0, []
    if resuming:
        state = torch.load(run/'last.pt', map_location=args.device, weights_only=False)
        if state['identity'] != identity:
            raise RuntimeError('Resume identity mismatch')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        start, best, anchor, stale, history = [state[k] for k in ['epoch', 'best', 'anchor', 'stale', 'history']]
        print('Resume after epoch:', start, flush=True)
    pool = frame[frame.split.eq('train')].reset_index(drop=True)
    val = frame[frame.split.eq('val')].reset_index(drop=True)
    val_loader = make_loader(Crops(val, args.data_root, transform), args)
    print(f'{args.model}: train pool={len(pool)}, balanced epoch={2*int(pool.class_index.astype(int).value_counts().min())}, val={len(val)}', flush=True)
    for epoch in range(start, args.epochs):
        if stale >= args.patience:
            break
        epoch_seed = args.seed + epoch * 100003
        seed_all(epoch_seed)
        sampled = balanced_epoch(pool, epoch_seed)
        loader = make_loader(Crops(sampled, args.data_root, transform, True, epoch_seed), args, True, epoch_seed)
        model.train()
        total, count, last = 0., 0, time.monotonic()
        for step, (x, y, _) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(args.device), y.to(args.device)
            with torch.autocast(device_type=args.device, enabled=args.device == 'cuda'):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())*len(y)
            count += len(y)
            if time.monotonic()-last >= args.log_interval_seconds or step == len(loader):
                print(f'Epoch{epoch+1} Train: {step}/{len(loader)} loss={total/count:.4f}', flush=True)
                last = time.monotonic()
        values, _ = evaluate(model, val_loader, val, args, f'Epoch{epoch+1} Validation')
        auc = values['video']['auc']
        improved = auc > best
        best = max(best, auc)
        if auc > anchor + 1e-4:
            anchor, stale = auc, 0
        else:
            stale += 1
        history.append({'epoch': epoch+1, 'train_loss': total/count, 'val': values})
        state = {'identity': identity, 'config': cfg, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                 'epoch': epoch+1, 'best': best, 'anchor': anchor, 'stale': stale, 'history': history}
        if improved:
            save(run/'best.pt', {k: state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch', 'best']})
        save(run/'last.pt', state)
        json_write(run/'history.json', history)
        pd.DataFrame([{'epoch': h['epoch'], 'train_loss': h['train_loss'], 'val_loss': h['val']['loss'],
                       'val_video_auc': h['val']['video']['auc'], 'val_frame_auc': h['val']['frame']['auc']} for h in history]).to_csv(run/'history.csv', index=False)
        print(f'Epoch{epoch+1}: val video AUC={auc:.4f} best={best:.4f} patience={stale}/{args.patience}', flush=True)
    if not (run/'best.pt').is_file():
        raise RuntimeError('No completed training epoch')
    json_write(run/'completed.json', {'identity': identity, 'best_val_video_auc': best, 'epochs_completed': len(history)})
    print('Training completed:', run, flush=True)


def test(args):
    frame, split = load_manifest(args)
    run = args.output_root / f'{args.model}_celebdf_identity_v1_seed{args.seed}'
    checkpoint = run / 'best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['config']['split'] != split or state['config']['model'] != args.model or state['config']['seed'] != args.seed:
        raise RuntimeError('Evaluation split/model/seed mismatch')
    seed_all(args.seed)
    model, transform = model_and_transform(args.model, pretrained=False)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.to(args.device)
    test_frame = frame[frame.split.eq('test')].reset_index(drop=True)
    # Clean source baseline first. Never choose checkpoint using test performance.
    values, predictions = evaluate(model, make_loader(Crops(test_frame, args.data_root, transform), args),
                                   test_frame, args, 'Clean Test')
    values.update(model=args.model, checkpoint_epoch=state['epoch'], manifest_sha256=split['manifest_sha256'],
                  checkpoint_sha256=digest(checkpoint), protocol=split['protocol'],
                  warning='Custom identity test, NOT official Celeb-DF test; previously examined data not fully untouched.')
    predictions.to_csv(run/'test_predictions.csv', index=False)
    json_write(run/'test_metrics.json', values)
    print(json.dumps(values, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['train', 'test'])
    p.add_argument('--model', choices=['xception', 'clip_b16'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifest-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--head-learning-rate', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--log-interval-seconds', type=float, default=60)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        p.error('epochs/batch-size/patience must be positive')
    (train if args.action == 'train' else test)(args)


if __name__ == '__main__':
    main()
