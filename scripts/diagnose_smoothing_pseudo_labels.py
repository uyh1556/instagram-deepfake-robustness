"""Diagnostic-only non-test CDF labels and saved final EMA snapshots; never train."""
import argparse
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_smoothing_uda_failures as errors

PROTOCOL = 'smoothing_pseudo_label_audit_v1'
PREFIX = 'celebdf_v2_face_crops16_v1/'
ARCHIVES = {'real':826142720, 'fake':5679564800}
SNAPSHOTS = {'mean_teacher':'ffpp_c23_smoothing_mean_teacher_v1',
             'stable_agreement':'ffpp_c23_smoothing_stable_uda_v1'}


class AuditViews:
    """A label-free, module-level wrapper also usable by spawned loader workers."""
    def __init__(self, views):
        self.views = views

    def __len__(self):
        return len(self.views)

    def __getitem__(self, index):
        teacher, _ = self.views[index]
        return teacher, index


def json_read(path):
    return json.loads(path.read_text())


def safe_relative(path):
    relative = Path(path)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'Unsafe relative path: {path}')
    return relative


def frozen_target(args):
    folders = []
    for path in sorted(args.hard_manifest_root.rglob('protocol.json')):
        cfg = json_read(path)
        if (cfg.get('protocol') == SNAPSHOTS['mean_teacher'] and cfg.get('seed') == args.seed
                and (path.parent/'target_unlabelled.csv').is_file()):
            folders.append(path.parent)
    if not folders:
        raise FileNotFoundError(f'Frozen hard UDA manifests absent: {args.hard_manifest_root}')
    signatures = [{name:errors.saved.digest(folder/name) for name in
                   ['protocol.json','target_unlabelled.csv']} for folder in folders]
    if any(item != signatures[0] for item in signatures):
        raise ValueError('Different hard UDA manifests; narrow --hard-manifest-root')
    folder = folders[0]
    protocol = json_read(folder/'protocol.json')
    frame = pd.read_csv(folder/'target_unlabelled.csv', dtype=str, keep_default_na=False)
    if (list(frame) != ['crop_path','video_id','roi_status'] or frame.crop_path.duplicated().any()
            or frame.eq('').any().any()):
        raise ValueError('Expected frozen unique, unlabelled target membership')
    import hashlib
    paths_hash = hashlib.sha256(json.dumps(frame.crop_path.tolist(), sort_keys=True).encode()).hexdigest()
    if paths_hash != protocol['data']['target_paths_sha256']:
        raise ValueError('Frozen target paths hash differs')
    analysis = json_read(args.analysis_root/'analysis_protocol.json')
    test_path = args.analysis_root/'test_membership.csv'
    if errors.saved.digest(test_path) != analysis['test_membership_sha256']:
        raise ValueError('Saved test membership differs')
    if protocol['data']['test_manifest_sha256'] != analysis['evaluation_signature']['manifest_sha256']:
        raise ValueError('Hard UDA and error diagnostic use different CDF test sets')
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
    if (set(frame.video_id) & set(test.video_id) or set(frame.crop_path) & set(test.crop_path)
            or set(frame.video_id) & set(protocol['excluded_video_ids'])):
        raise ValueError('Target overlaps test or frozen conservative exclusion list')
    for path in frame.crop_path:
        safe_relative(path)
    return frame, protocol, folder


def prepare(args):
    """Extract only frozen target PNGs and original non-test manifests; no test TAR."""
    frame, _, _ = frozen_target(args)
    wanted = {PREFIX+path for path in frame.crop_path if not (args.target_root/path).is_file()}
    metadata = {PREFIX+f'manifests/celebdf_v2_roi16_non_test_{kind}_v1.csv' for kind in ARCHIVES
                if not (args.target_root/'manifests'/f'celebdf_v2_roi16_non_test_{kind}_v1.csv').is_file()}
    if not wanted and not metadata:
        print('Frozen target crops and non-test manifests already available.', flush=True)
        labelled_target(args)
        return
    args.archive_cache.mkdir(parents=True, exist_ok=True)
    for kind, size in ARCHIVES.items():
        filename = f'celebdf_v2_roi16_non_test_{kind}_v1.tar'
        # Existing target folders often already contain one archive's complete subset.
        if not wanted and not metadata:
            break
        source, local = args.drive_data/filename, args.archive_cache/filename
        if not source.is_file() or source.stat().st_size != size:
            raise FileNotFoundError(f'Existing non-test TAR missing/incomplete: {source}')
        owned = False
        if not local.is_file() or local.stat().st_size != size:
            if local.exists():
                raise FileExistsError(f'Incomplete local TAR; choose another --archive-cache: {local}')
            if shutil.disk_usage(args.archive_cache).free < size+2*2**30:
                raise RuntimeError('Not enough Colab disk to copy this TAR and retain target crops')
            print('Copying existing TAR:', filename, flush=True)
            copying = local.with_suffix('.copying')
            if copying.exists():
                raise FileExistsError(f'Interrupted copy exists; choose another archive cache: {copying}')
            shutil.copy2(source, copying)
            if copying.stat().st_size != size:
                raise RuntimeError('Incomplete TAR copy')
            copying.replace(local)
            owned = True
        print('Extracting frozen target members only:', filename, flush=True)
        extracted = 0
        with tarfile.open(local, 'r|') as archive:
            for member in archive:
                name = member.name.removeprefix('./')
                if name not in wanted and name not in metadata:
                    continue
                if not member.isfile():
                    raise ValueError(f'Expected ordinary archive member: {name}')
                relative = safe_relative(name.removeprefix(PREFIX))
                dest = args.target_root/relative
                if dest.exists():
                    raise FileExistsError(f'Preserve existing target data: {dest}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = dest.with_suffix(dest.suffix+'.extracting')
                if temp.exists():
                    raise FileExistsError(temp)
                with archive.extractfile(member) as src, temp.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                temp.replace(dest)
                wanted.discard(name); metadata.discard(name)
                extracted += 1
                if extracted % 500 == 0:
                    print('Extracted:', extracted, '| target members remaining:', len(wanted), flush=True)
        # Only a temporary TAR copied by THIS call is removed; Drive and prior copies stay.
        if owned:
            local.unlink()
        print('Prepared members:', extracted, flush=True)
    if wanted or metadata:
        raise FileNotFoundError(f'Frozen archive members absent: {sorted(wanted | metadata)[:5]}')
    labelled_target(args)


def labelled_target(args):
    """GT is restored for reporting only. No resampling or learning uses these labels."""
    frame, frozen, folder = frozen_target(args)
    paths = [args.target_root/'manifests'/f'celebdf_v2_roi16_non_test_{kind}_v1.csv' for kind in ARCHIVES]
    if [errors.saved.digest(path) for path in paths] != frozen['data']['non_test_manifest_sha256']:
        raise ValueError('Original non-test manifest hashes differ from training')
    catalog = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths], ignore_index=True)
    if catalog.crop_path.duplicated().any() or not catalog.split.eq('non_test_pool').all():
        raise ValueError('Non-test label lookup must be unique and non-test only')
    lookup = catalog.set_index('crop_path')
    if not set(frame.crop_path).issubset(lookup.index):
        raise ValueError('Frozen target path missing from original non-test metadata')
    labels = lookup.loc[frame.crop_path].reset_index()
    if (not labels[['video_id','roi_status']].equals(frame[['video_id','roi_status']])
            or not labels.class_index.isin(['0','1']).all()
            or labels.groupby('video_id').class_index.nunique().gt(1).any()):
        raise ValueError('Target metadata/labels conflict')
    missing = [path for path in frame.crop_path if not (args.target_root/path).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} target crops missing; run prepare: {missing[:3]}')
    return frame, labels.class_index.astype(int).to_numpy(), frozen, folder


def snapshot(args, model, strategy, frozen):
    """Read FINAL saved EMA, not selected best student and not historical EMA states."""
    import torch
    import ffpp_c23_smoothing_mean_teacher as engine
    protocol = SNAPSHOTS[strategy]
    root = args.hard_runs_root if strategy == 'mean_teacher' else args.stable_runs_root
    run = root/f'{model}_{strategy}_{protocol}_seed{args.seed}'
    cfg, done = json_read(run/'config.json'), json_read(run/'completed.json')
    path = run/'last.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state['config'] != cfg or state['identity'] != engine.evaluation.key(cfg)
            or done['identity'] != state['identity'] or cfg['model'] != model
            or cfg['strategy'] != strategy or cfg['protocol'] != protocol or cfg['seed'] != args.seed
            or state['epoch'] != done['epochs_completed']
            or cfg['data'] != frozen['data'] or cfg['code_sha256'] != engine.code_hashes()
            or cfg['preprocessing'] != engine.base.PREPROCESS[model]
            or not state.get('teacher_state_dict')):
        raise ValueError(f'Final EMA snapshot/config/membership mismatch: {run}')
    if strategy == 'stable_agreement':
        import ffpp_c23_smoothing_stable_uda as stable
        if cfg.get('extension_code_sha256') != errors.saved.digest(Path(stable.__file__)):
            raise ValueError('Stable UDA extension code changed')
    weights = state['teacher_state_dict']
    if strategy == 'stable_agreement':
        if not all(key.startswith('model.') for key in weights):
            raise ValueError('Expected wrapped four-view EMA state')
        weights = {key.removeprefix('model.'):value for key,value in weights.items()}
    info = dict(checkpoint=str(path), checkpoint_sha256=errors.saved.digest(path),
                epoch=int(state['epoch']), selected_student_epoch=int(done['selected_epoch']),
                config_sha256=errors.saved.digest(run/'config.json'),
                completed_sha256=errors.saved.digest(run/'completed.json'),
                teacher='final saved EMA; retrospective diagnostic, not per-step training replay')
    return weights, cfg, info


def quality_tables(frame):
    """Report BOTH pool-frame statistics and expected video-uniform exposure statistics."""
    records = []
    for (model, strategy, condition), group in frame.groupby(['model','strategy','condition'], sort=False):
        for rule in ['confidence_selected','agreement_selected','protocol_selected']:
            for grouping, column in [('all',None), ('true_class','class_index'), ('predicted_class','pseudo_label')]:
                parts = [('all',group)] if column is None else list(group.groupby(column))
                for label, part in parts:
                    selected = part[part[rule]]
                    wrong = selected.pseudo_label.ne(selected.class_index)
                    weight = selected.video_uniform_weight
                    records.append(dict(model=model, strategy=strategy, condition=condition, rule=rule,
                        grouping=grouping, class_label=str(label), images=len(part), selected=len(selected),
                        selected_wrong=int(wrong.sum()), selected_correct=int((~wrong).sum()),
                        selected_accuracy=float((~wrong).mean()) if len(selected) else None,
                        selected_wrong_fake_as_real=int((wrong & selected.class_index.eq(1)).sum()),
                        selected_wrong_real_as_fake=int((wrong & selected.class_index.eq(0)).sum()),
                        expected_video_uniform_selection_mass=float(weight.sum()),
                        video_uniform_selected_accuracy=float(((~wrong)*weight).sum()/weight.sum()) if weight.sum() else None))
    return pd.DataFrame(records)


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import ffpp_c23_smoothing_mean_teacher as engine
    import ffpp_c23_smoothing_stable_uda as stable
    frame, labels, frozen, folder = labelled_target(args)
    cfg = json_read(args.protocol)
    if list(cfg['conditions']) != errors.CONDITIONS or cfg['conditions'] != dict(zip(errors.CONDITIONS,[0,.25,.5,.75])):
        raise ValueError('Unexpected smoothing protocol')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Part B requires a GPU runtime; Part A is CPU-only')
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = [{**row, 'condition':'normal', 'flip':False} for row in frame.to_dict('records')]
    tables, snapshots = [], []
    for model_name in args.models:
        for strategy in args.strategies:
            weights, saved_config, info = snapshot(args, model_name, strategy, frozen)
            if saved_config['smoothing'] != cfg or saved_config['opencv'] != engine.evaluation.cv2.__version__:
                raise ValueError('Smoothing configuration/OpenCV differs from saved training')
            identity = dict(protocol=PROTOCOL, snapshot=info, target_sha256=errors.saved.digest(folder/'target_unlabelled.csv'),
                non_test_manifest_sha256=frozen['data']['non_test_manifest_sha256'],
                script_sha256=errors.saved.digest(Path(__file__)), model=model_name, strategy=strategy,
                smoothing=cfg, threshold=float(saved_config['pseudo_threshold']),
                views='each frozen target image once in normal/weak/medium/strong, no flip; GT only for reporting',
                torch=str(torch.__version__), opencv=engine.evaluation.cv2.__version__)
            prefix = args.output_root/f'{model_name}_{strategy}'
            csv_path, report_path = prefix.with_suffix('.csv'), prefix.with_suffix('.json')
            if csv_path.exists() or report_path.exists():
                if not csv_path.is_file() or not report_path.is_file():
                    raise FileExistsError(f'Incomplete diagnostic pair; choose a new output root: {prefix}')
                old = json_read(report_path)
                if old['identity'] != identity or old['prediction_sha256'] != errors.saved.digest(csv_path):
                    raise ValueError('Saved snapshot diagnostic differs; choose a new output root')
                tables.append(pd.read_csv(csv_path, dtype={'video_id':str,'crop_path':str}))
                snapshots.append(info)
                del weights
                print('Reuse snapshot diagnostic:', model_name, strategy, flush=True)
                continue
            model, transform = engine.base.build_model(model_name, pretrained=False)
            model.load_state_dict(weights, strict=True); del weights
            model.to(args.device).eval().requires_grad_(False)
            cache_identity = dict(target_paths_sha256=frozen['data']['target_paths_sha256'], smoothing=cfg,
                opencv=engine.evaluation.cv2.__version__, code_sha256=engine.code_hashes())
            args._target_cache = engine.common.pixel_cache.BilateralCache(
                args.target_train_cache/'observed', cfg['bilateral'], cache_identity)
            views = stable.TargetViews(plan, args, transform, cfg, False)

            loader = DataLoader(AuditViews(views), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=args.device == 'cuda')
            probabilities = np.empty((len(frame),4,2), dtype=np.float32)
            with torch.inference_mode():
                for images, indexes in tqdm(loader, desc=f'{model_name} {strategy} final EMA x4',
                                            mininterval=10, dynamic_ncols=True):
                    flat = images.flatten(0,1)
                    logits = torch.cat([model(part.to(args.device)).float() for part in flat.split(args.batch_size)])
                    if not torch.isfinite(logits).all():
                        raise RuntimeError('Nonfinite final EMA predictions')
                    probabilities[indexes.numpy()] = logits.reshape(-1,4,2).softmax(-1).cpu().numpy()
            pseudo = probabilities.argmax(-1)
            confidence = probabilities.max(-1)
            agreement = (pseudo == pseudo[:, :1]).all(1)
            videos = frame.video_id.value_counts()
            video_weights = 1/(frame.video_id.map(videos).to_numpy()*len(videos))
            rows = []
            for index, condition in enumerate(errors.CONDITIONS):
                part = frame.copy()
                part['model'], part['strategy'], part['condition'] = model_name, strategy, condition
                # Diagnostic GT is attached only AFTER model inference; never fed to Dataset/model.
                part['class_index'], part['pseudo_label'] = labels, pseudo[:,index]
                part['confidence'], part['probability_fake'] = confidence[:,index], probabilities[:,index,1]
                part['confidence_selected'] = confidence[:,index] >= saved_config['pseudo_threshold']
                part['four_view_agreement'] = agreement
                part['agreement_selected'] = part.confidence_selected & agreement
                part['protocol_selected'] = part[('confidence_selected' if strategy == 'mean_teacher' else 'agreement_selected')]
                part['smoothing_induced_wrong'] = (pseudo[:,0] == labels) & (pseudo[:,index] != labels)
                part['video_uniform_weight'] = video_weights
                rows.append(part)
            result = pd.concat(rows, ignore_index=True)
            temp = csv_path.with_suffix('.tmp.csv'); result.to_csv(temp,index=False); temp.replace(csv_path)
            errors.write_json(report_path, dict(identity=identity, prediction_sha256=errors.saved.digest(csv_path)))
            tables.append(result); snapshots.append(info)
            del model, views, loader
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    combined = pd.concat(tables, ignore_index=True)
    quality_tables(combined).to_csv(args.output_root/'pseudo_quality.csv', index=False)
    errors.write_json(args.output_root/'audit_protocol.json', dict(protocol=PROTOCOL, snapshots=snapshots,
        diagnostic_labels_used=True, training=False, checkpoint_or_threshold_selection=False,
        final_snapshot_not_training_history=True, independent_causal_evidence=False,
        note='Final EMA snapshots have different epochs and lack historical per-step teacher states. '
             'Uniform pool statistics and expected video-uniform weights are reported separately. '
             'Fixed no-flip views differ from randomized training exposures; counts do not replay training. '
             'No test labels are used in this target audit. Ground truth is diagnostic only.'))
    print('Saved diagnostic-only pseudo-label quality:', args.output_root/'pseudo_quality.csv', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','evaluate'])
    for name in ['hard-manifest-root','analysis-root','target-root','output-root']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--hard-runs-root', type=Path)
    p.add_argument('--stable-runs-root', type=Path)
    p.add_argument('--drive-data', type=Path)
    p.add_argument('--archive-cache', type=Path, default=Path('/content/smoothing_audit_archives'))
    p.add_argument('--target-train-cache', type=Path, default=Path('/content/smoothing_mean_teacher_target_cache_v1'))
    p.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parents[1]/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=errors.saved.MODELS, default=['xception'])
    p.add_argument('--strategies', nargs='+', choices=list(SNAPSHOTS), default=list(SNAPSHOTS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = p.parse_args()
    if min(args.batch_size,args.seed+1) < 1 or args.workers < 0 or len(set(args.models)) != len(args.models) or len(set(args.strategies)) != len(args.strategies):
        p.error('Invalid counts or duplicate choices')
    if args.action == 'prepare' and args.drive_data is None:
        p.error('prepare requires --drive-data')
    if args.action == 'evaluate' and (args.hard_runs_root is None or 'stable_agreement' in args.strategies and args.stable_runs_root is None):
        p.error('evaluate requires saved run roots')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    (prepare if arguments.action == 'prepare' else evaluate)(arguments)
