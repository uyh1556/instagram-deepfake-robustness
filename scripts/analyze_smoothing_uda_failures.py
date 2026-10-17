"""Paired error accounting from saved predictions; no models, images or training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_smoothing_error_tradeoff as saved

PROTOCOL = 'smoothing_uda_failure_v1'
CONDITIONS = saved.CONDITIONS
STRATEGIES = ['mean_teacher', 'stable_agreement', 'random_agreement_count', 'augmentation']


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def discover(args):
    records, sources = {}, []
    wanted = ['baseline', *args.strategies]
    for root in args.evaluation_roots:
        if not root.is_dir():
            continue
        for protocol in sorted(root.rglob('evaluation_protocol.json')):
            for model in args.models:
                for strategy in wanted:
                    for condition in CONDITIONS:
                        report = protocol.parent/f'{model}_{strategy}_{condition}.json'
                        if not report.is_file() or not report.with_suffix('.csv').is_file():
                            continue
                        item = saved.load_record(report, args.expected_images)
                        key = model, strategy, condition
                        if key in records:
                            old = records[key]
                            if (item['protocol'] != old['protocol']
                                    or item['report']['checkpoint_sha256'] != old['report']['checkpoint_sha256']
                                    or not item['frame'][saved.META].equals(old['frame'][saved.META])
                                    or not np.allclose(item['frame'][['raw_margin', 'probability_fake']],
                                                       old['frame'][['raw_margin', 'probability_fake']],
                                                       rtol=0, atol=1e-12)):
                                raise ValueError(f'Conflicting saved predictions: {key}; narrow evaluation roots')
                        else:
                            records[key] = item
                        sources.append(item['source'])
    for model in args.models:
        for strategy in wanted:
            group = [records.get((model, strategy, c)) for c in CONDITIONS]
            if any(item is None for item in group):
                raise FileNotFoundError(f'{model}/{strategy}: four complete JSON/CSV pairs required')
            if len({item['report']['checkpoint_sha256'] for item in group}) != 1:
                raise ValueError(f'Checkpoint changes across conditions: {model}/{strategy}')
        anchor = records[model, 'baseline', 'normal']
        for (name, strategy, condition), item in records.items():
            if name == model and (item['protocol'] != anchor['protocol']
                    or not item['frame'][saved.META].equals(anchor['frame'][saved.META])):
                raise ValueError(f'Membership, labels, ROI or smoothing differs: {model}/{strategy}/{condition}')
    # Pseudo-label audit uses one common target/test exclusion list across models.
    first = records[args.models[0], 'baseline', 'normal']
    for model in args.models[1:]:
        other = records[model, 'baseline', 'normal']
        if other['protocol'] != first['protocol'] or not other['frame'][saved.META].equals(first['frame'][saved.META]):
            raise ValueError('Different CDF test membership across models')
    return records, sources


def paired_rows(model, strategy, condition, baseline, candidate, normal):
    """All arrays are joined/sorted by crop_path before entering this function."""
    y = baseline.class_index.astype(int).to_numpy()
    old = baseline.raw_margin.to_numpy() >= 0
    new = candidate.raw_margin.to_numpy() >= 0
    clean = normal.raw_margin.to_numpy() >= 0
    result = baseline[saved.META].reset_index(drop=True).copy()
    result['model'], result['strategy'], result['condition'] = model, strategy, condition
    result['baseline_margin'] = baseline.raw_margin.to_numpy()
    result['candidate_margin'] = candidate.raw_margin.to_numpy()
    result['baseline_normal_margin'] = normal.raw_margin.to_numpy()
    result['baseline_correct'], result['candidate_correct'] = old == y, new == y
    result['repaired'] = (old != y) & (new == y)
    result['new_error'] = (old == y) & (new != y)
    result['baseline_smoothing_induced_error'] = (clean == y) & (old != y)
    result['baseline_already_wrong_normal'] = clean != y
    result['baseline_probability_fake'] = baseline.probability_fake.to_numpy()
    result['candidate_probability_fake'] = candidate.probability_fake.to_numpy()
    return result


def transition_summary(frame):
    n = len(frame)
    before = int((~frame.baseline_correct).sum())
    after = int((~frame.candidate_correct).sum())
    repaired, harmed = int(frame.repaired.sum()), int(frame.new_error.sum())
    if before-after != repaired-harmed:
        raise ValueError('Error accounting identity failed')
    return dict(images=n, baseline_errors=before, candidate_errors=after,
                repaired_errors=repaired, new_errors=harmed, net_errors_removed=repaired-harmed,
                baseline_error_rate=before/n, candidate_error_rate=after/n,
                error_reduction_pp=100*(before-after)/n,
                repair_fraction_of_old_errors=repaired/before if before else None,
                harm_fraction_of_old_correct=harmed/(n-before) if n > before else None)


def analyze(args):
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f'Choose an empty output root: {args.output_root}')
    records, sources = discover(args)
    comparisons, transitions, induced, margins, fixed, flips = [], [], [], [], [], []
    for model in args.models:
        normal = records[model, 'baseline', 'normal']['frame']
        for strategy in ['baseline', *args.strategies]:
            anchor = records[model, strategy, 'normal']['frame']
            for condition in CONDITIONS:
                item = records[model, strategy, condition]
                f = item['fixed']
                fixed.append(dict(model=model, strategy=strategy, condition=condition,
                                  fpr=f['fpr'], fnr=f['fnr'], ber=f['ber'], auc=item['auc'],
                                  selected_epoch=item['report'].get('selected_epoch'),
                                  checkpoint_sha256=item['report']['checkpoint_sha256']))
                y = anchor.class_index.astype(int).to_numpy()
                was = anchor.raw_margin.to_numpy() >= 0
                now = item['frame'].raw_margin.to_numpy() >= 0
                for label in [0, 1]:
                    mask = y == label
                    flips.append(dict(model=model, strategy=strategy, condition=condition,
                        class_name='real' if label == 0 else 'fake', images=int(mask.sum()),
                        correct_normal_wrong_condition=int(((was == y) & (now != y) & mask).sum()),
                        wrong_normal_correct_condition=int(((was != y) & (now == y) & mask).sum())))
        for strategy in args.strategies:
            candidate_normal = records[model, strategy, 'normal']['frame']
            for condition in CONDITIONS:
                baseline = records[model, 'baseline', condition]['frame']
                candidate = records[model, strategy, condition]['frame']
                paired = paired_rows(model, strategy, condition, baseline, candidate, normal)
                paired['candidate_normal_correct'] = (
                    (candidate_normal.raw_margin.to_numpy() >= 0) == paired.class_index.astype(int).to_numpy())
                comparisons.append(paired)
                for label in [0, 1]:
                    group = paired[paired.class_index.astype(int).eq(label)]
                    info = dict(model=model, strategy=strategy, condition=condition,
                                class_name='real' if label == 0 else 'fake')
                    transitions.append({**info, **transition_summary(group)})
                    selected = group[group.baseline_smoothing_induced_error]
                    repaired = selected.candidate_correct
                    induced.append({**info, 'baseline_smoothing_induced_errors':len(selected),
                        'repaired':int(repaired.sum()), 'still_wrong':int((~repaired).sum()),
                        'repaired_and_candidate_normal_correct':int((repaired & selected.candidate_normal_correct).sum()),
                        'repaired_but_candidate_normal_wrong':int((repaired & ~selected.candidate_normal_correct).sum()),
                        'repair_rate':float(repaired.mean()) if len(selected) else None})
                    delta = group.candidate_margin-group.baseline_margin
                    margins.append({**info, 'images':len(group), 'mean_margin_change':float(delta.mean()),
                        'median_margin_change':float(delta.median()),
                        'margin_change_q25':float(delta.quantile(.25)),
                        'margin_change_q75':float(delta.quantile(.75)),
                        'fraction_margin_moved_toward_fake':float(delta.gt(0).mean()),
                        'fraction_margin_moved_toward_real':float(delta.lt(0).mean())})
    args.output_root.mkdir(parents=True, exist_ok=True)
    tables = {'error_transitions.csv':pd.DataFrame(transitions),
              'smoothing_induced_errors.csv':pd.DataFrame(induced),
              'margin_changes.csv':pd.DataFrame(margins), 'smoothing_flips.csv':pd.DataFrame(flips),
              'paired_image_errors.csv':pd.concat(comparisons, ignore_index=True),
              'fixed_metrics.csv':pd.DataFrame(fixed)}
    tables['condition_means.csv'] = tables['fixed_metrics.csv'].groupby(
        ['model','strategy'], sort=False)[['fpr','fnr','ber','auc']].mean().reset_index()
    for filename, frame in tables.items():
        frame.to_csv(args.output_root/filename, index=False)
    first = records[args.models[0], 'baseline', 'normal']
    first['frame'][saved.META].to_csv(args.output_root/'test_membership.csv', index=False)
    write_json(args.output_root/'analysis_protocol.json', dict(protocol=PROTOCOL,
        script_sha256=saved.digest(Path(__file__)), reader_sha256=saved.digest(Path(saved.__file__)),
        models=args.models, strategies=args.strategies, sources=sources,
        evaluation_signature=first['protocol'], decision_rule='raw_margin >= 0; fixed cutoff',
        test_membership_sha256=saved.digest(args.output_root/'test_membership.csv'),
        diagnostic_only=True, threshold_or_checkpoint_selection=False, causal_claim=False,
        limitation='Paired error accounting and score changes do not identify the training cause; '
                   'no claim of uniform offset, independent frames, or pseudo-label accuracy.'))
    print(tables['condition_means.csv'].to_string(index=False), flush=True)
    print('Saved paired error diagnostic:', args.output_root, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-roots', type=Path, nargs='+', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=saved.MODELS, default=['xception'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES,
                   default=['mean_teacher','stable_agreement'])
    p.add_argument('--expected-images', type=int, default=8288)
    args = p.parse_args()
    if args.expected_images < 0 or len(set(args.models)) != len(args.models) or len(set(args.strategies)) != len(args.strategies):
        p.error('Invalid image count or duplicate choices')
    return args


if __name__ == '__main__':
    analyze(parse_args())
