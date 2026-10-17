"""CPU-only operating-point diagnostic from saved CDF raw-margin predictions."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PROTOCOL = 'smoothing_error_tradeoff_v1'
CONDITIONS = ['normal', 'smooth_weak', 'smooth_medium', 'smooth_strong']
MODELS = ['xception', 'clip_b16', 'dinov2_b14']
STRATEGIES = ['baseline', 'augmentation', 'mean_teacher', 'mean_teacher_smoothing',
              'soft_w05', 'soft_w025', 'normal_only', 'clean_to_mixed']
META = ['crop_path', 'video_id', 'class_index', 'roi_status']
WARNING = ('TEST-LABEL ORACLE DIAGNOSTIC ONLY. Thresholds use CDF test labels; '
           'not validation-selected thresholds, deployable performance or a method-selection rule. '
           'Condition-wise thresholds may differ. Frames within a video are correlated; '
           'no significance or confidence interval is claimed.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def load_record(report_path, expected_images):
    """Validate saved scores without importing training code, Torch, PIL or models."""
    report = read_json(report_path)
    protocol_path = report_path.parent/'evaluation_protocol.json'
    protocol = read_json(protocol_path)
    prediction = report_path.with_suffix('.csv')
    frame = pd.read_csv(prediction, dtype={k:str for k in META}, keep_default_na=False)
    required = set(META+['raw_margin', 'probability_fake'])
    if not required.issubset(frame) or frame.crop_path.duplicated().any():
        raise ValueError(f'Missing columns or duplicate images: {prediction}')
    if expected_images and len(frame) != expected_images:
        raise ValueError(f'Expected {expected_images} images, got {len(frame)}: {prediction}')
    if (set(frame.class_index) != {'0', '1'} or frame[META].eq('').any().any()
            or not np.isfinite(frame.raw_margin).all()
            or not np.isfinite(frame.probability_fake).all()
            or not frame.probability_fake.between(0, 1).all()):
        raise ValueError(f'Invalid metadata, labels or scores: {prediction}')
    if (report.get('precision') != 'FP32' or report.get('unit') != 'frame'
            or report.get('threshold') != .5 or report.get('images') != len(frame)
            or not report.get('checkpoint_sha256') or not protocol.get('smoothing_code_sha256')
            or not protocol.get('manifest_sha256')
            or report.get('manifest_sha256') != protocol['manifest_sha256']):
        raise ValueError(f'Incompatible saved evaluation: {report_path}')
    if protocol.get('config', {}).get('conditions') != dict(zip(CONDITIONS, [0, .25, .5, .75])):
        raise ValueError(f'Different smoothing conditions: {protocol_path}')
    y, scores = frame.class_index.astype(int).to_numpy(), frame.raw_margin.to_numpy()
    fixed = counts(y, scores, 0.)
    values = {'real_fpr':fixed['fpr'], 'fake_fnr':fixed['fnr'],
              'balanced_error_rate':fixed['ber'], 'auc':float(roc_auc_score(y, scores))}
    if any(not np.isclose(report.get(k, np.nan), v, atol=1e-10, rtol=0) for k,v in values.items()):
        raise ValueError(f'Saved metrics differ from raw-margin predictions: {report_path}')
    signature = {key:protocol[key] for key in ['manifest_sha256','config','smoothing_code_sha256']}
    # OpenCV is part of the smoothing implementation provenance, when recorded.
    signature['opencv'] = protocol.get('opencv')
    return {'frame':frame.set_index('crop_path', drop=False).sort_index(), 'report':report,
            'protocol':signature, 'auc':values['auc'], 'fixed':fixed,
            'source':{'report':str(report_path), 'report_sha256':digest(report_path),
                      'prediction':str(prediction), 'prediction_sha256':digest(prediction),
                      'protocol_sha256':digest(protocol_path)}}


def counts(y, scores, threshold):
    real, fake = y == 0, y == 1
    fp = int(np.count_nonzero(scores[real] >= threshold))
    fn = int(np.count_nonzero(scores[fake] < threshold))
    fpr, fnr = fp/int(real.sum()), fn/int(fake.sum())
    return {'fp':fp, 'fn':fn, 'real_images':int(real.sum()), 'fake_images':int(fake.sum()),
            'fpr':fpr, 'fnr':fnr, 'ber':(fpr+fnr)/2}


def curve(y, scores):
    """All realizable deterministic cutoffs: preserve ties, no interpolation."""
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    nr, nf = int((y == 0).sum()), int((y == 1).sum())
    return pd.DataFrame({'diagnostic_margin_threshold':thresholds,
                         'fp':np.rint(fpr*nr).astype(int),
                         'fn':np.rint((1-tpr)*nf).astype(int),
                         'fpr':fpr, 'fnr':1-tpr})


def compare_condition(candidate, reference):
    frame = candidate['frame']
    y, scores = frame.class_index.astype(int).to_numpy(), frame.raw_margin.to_numpy()
    points, base = curve(y, scores), reference['fixed']
    # At MOST the reference FN/FP count, since ties can make exact equality impossible.
    at_fnr = points[points.fn.le(base['fn'])].sort_values(['fp','fn']).iloc[0]
    at_fpr = points[points.fp.le(base['fp'])].sort_values(['fn','fp']).iloc[0]
    weak = points.fp.le(base['fp']) & points.fn.le(base['fn'])
    strict = points.fp.lt(base['fp']) & points.fn.lt(base['fn'])
    fixed = candidate['fixed']
    out = {'reference_fpr':base['fpr'], 'reference_fnr':base['fnr'],
           'reference_auc':reference['auc'], 'candidate_auc':candidate['auc'],
           'delta_auc':candidate['auc']-reference['auc'],
           'auc_maintained':candidate['auc'] >= reference['auc']-1e-10,
           'fixed_fpr':fixed['fpr'], 'fixed_fnr':fixed['fnr'], 'fixed_ber':fixed['ber'],
           'fixed_both_strictly_lower':fixed['fp'] < base['fp'] and fixed['fn'] < base['fn'],
           'oracle_nonworse_both':bool(weak.any()),
           'oracle_strictly_lower_both':bool(strict.any())}
    for tag, row in [('fnr',at_fnr), ('fpr',at_fpr)]:
        out.update({f'oracle_at_reference_{tag}_{k}':float(row[k]) for k in
                    ['diagnostic_margin_threshold','fpr','fnr']})
    out['oracle_fpr_reduction_pp_at_reference_fnr'] = 100*(base['fpr']-at_fnr.fpr)
    out['oracle_fnr_reduction_pp_at_reference_fpr'] = 100*(base['fnr']-at_fpr.fnr)
    return out, points


def common_threshold(candidates, references):
    """Can ONE cutoff satisfy each of the four baseline operating points?"""
    thresholds = np.r_[np.inf, np.unique(np.concatenate(
        [record['frame'].raw_margin.to_numpy() for record in candidates]))[::-1]]
    nonworse, strict = np.ones(len(thresholds), bool), np.ones(len(thresholds), bool)
    fprs, fnrs = [], []
    for candidate, reference in zip(candidates, references):
        frame, base = candidate['frame'], reference['fixed']
        real = np.sort(frame.loc[frame.class_index.eq('0'), 'raw_margin'].to_numpy())
        fake = np.sort(frame.loc[frame.class_index.eq('1'), 'raw_margin'].to_numpy())
        fp = len(real)-np.searchsorted(real, thresholds, side='left')
        fn = np.searchsorted(fake, thresholds, side='left')
        nonworse &= (fp <= base['fp']) & (fn <= base['fn'])
        strict &= (fp < base['fp']) & (fn < base['fn'])
        fprs.append(fp/len(real)); fnrs.append(fn/len(fake))
    mfpr, mfnr = np.mean(fprs, axis=0), np.mean(fnrs, axis=0)
    feasible = np.flatnonzero(nonworse)
    best = feasible[np.argmin((mfpr+mfnr)[feasible])] if len(feasible) else None
    return {'oracle_one_threshold_nonworse_every_condition':bool(nonworse.any()),
            'oracle_one_threshold_strictly_lower_both_every_condition':bool(strict.any()),
            'auc_maintained_every_condition':all(c['auc'] >= r['auc']-1e-10
                                               for c,r in zip(candidates,references)),
            'oracle_common_margin_threshold':float(thresholds[best]) if best is not None else np.nan,
            'oracle_common_mean_fpr':float(mfpr[best]) if best is not None else np.nan,
            'oracle_common_mean_fnr':float(mfnr[best]) if best is not None else np.nan}


def discover(args):
    records, skipped, sources = {}, [], []
    for root in args.evaluation_roots:
        if not root.is_dir():
            skipped.append({'path':str(root), 'reason':'optional evaluation root absent'})
            continue
        # Only saved evaluation folders; never scan training checkpoints or datasets.
        folders = sorted({p.parent for p in root.rglob('evaluation_protocol.json')})
        for folder in folders:
            for model in args.models:
                for strategy in args.strategies:
                    for condition in CONDITIONS:
                        report = folder/f'{model}_{strategy}_{condition}.json'
                        if not report.is_file():
                            continue
                        if not report.with_suffix('.csv').is_file():
                            skipped.append({'path':str(report), 'reason':'prediction CSV absent'})
                            continue
                        record = load_record(report, args.expected_images)
                        key = model, strategy, condition
                        if key in records:
                            previous = records[key]
                            if (record['protocol'] != previous['protocol']
                                    or record['report']['checkpoint_sha256'] != previous['report']['checkpoint_sha256']
                                    or not record['frame'][META].equals(previous['frame'][META])
                                    or not np.allclose(record['frame'][['raw_margin','probability_fake']],
                                                       previous['frame'][['raw_margin','probability_fake']],
                                                       atol=1e-12, rtol=0)):
                                raise ValueError(f'Conflicting saved results for {key}: '
                                                 f'{previous["source"]["report"]} and {report}. Narrow --evaluation-roots.')
                        else:
                            records[key] = record
                        sources.append(record['source'])
    for model in args.models:
        for strategy in args.strategies:
            present = [c for c in CONDITIONS if (model,strategy,c) in records]
            if len(present) != len(CONDITIONS):
                if strategy in ['baseline','augmentation','mean_teacher']:
                    raise FileNotFoundError(f'{model}/{strategy}: missing '
                                            f'{sorted(set(CONDITIONS)-set(present))}; check evaluation roots.')
                for condition in present:
                    del records[model,strategy,condition]
                skipped.append({'model':model, 'strategy':strategy, 'reason':'optional strategy incomplete or absent',
                                'present_conditions':present})
                continue
            group = [records[model,strategy,c] for c in CONDITIONS]
            if len({r['report']['checkpoint_sha256'] for r in group}) != 1:
                raise ValueError(f'Different checkpoints across conditions: {model}/{strategy}')
        baseline = records[model,'baseline','normal']
        for key, record in records.items():
            if key[0] == model and (record['protocol'] != baseline['protocol']
                    or not record['frame'][META].equals(baseline['frame'][META])):
                raise ValueError(f'Evaluation protocol or image membership differs: {key}')
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f'Preserve previous analysis; choose an empty output root: {args.output_root}')
    return records, skipped, sources


def figures(records, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for model in args.models:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        for ax, condition in zip(axes, CONDITIONS):
            for strategy in args.strategies:
                record = records.get((model,strategy,condition))
                if record is None:
                    continue
                frame = record['frame']
                points = curve(frame.class_index.astype(int).to_numpy(), frame.raw_margin.to_numpy())
                line, = ax.plot(points.fpr, points.fnr, label=strategy, linewidth=1.2)
                ax.scatter(record['fixed']['fpr'], record['fixed']['fnr'], color=line.get_color(), s=15)
            ax.set(xlabel='Real FPR', ylabel='Fake FNR', title=condition, xlim=(0,1), ylim=(0,1))
            ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7)
        fig.suptitle(f'{model}: lower-left is better; dots = margin 0 / probability 0.5')
        fig.savefig(args.output_root/f'{model}_error_curves.png', dpi=140)
        plt.close(fig)


def analyze(args):
    records, skipped, sources = discover(args)
    fixed_rows, rows, common_rows = [], [], []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for strategy in args.strategies:
            group = [records.get((model,strategy,c)) for c in CONDITIONS]
            if any(r is None for r in group):
                continue
            for condition, candidate in zip(CONDITIONS, group):
                fixed_rows.append({'model':model,'strategy':strategy,'condition':condition,
                                   **candidate['fixed'],'auc':candidate['auc']})
                for ref_name in args.references:
                    result, _ = compare_condition(candidate, records[model,ref_name,condition])
                    rows.append({'model':model,'strategy':strategy,'condition':condition,
                                 'reference':ref_name, **result})
            for ref_name in args.references:
                result = common_threshold(group, [records[model,ref_name,c] for c in CONDITIONS])
                common_rows.append({'model':model,'strategy':strategy,'reference':ref_name, **result})
            print(f'Analyzed: {model} / {strategy} / 4 conditions', flush=True)
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(args.output_root/'fixed_threshold.csv', index=False)
    pd.DataFrame(rows).to_csv(args.output_root/'operating_points.csv', index=False)
    pd.DataFrame(common_rows).to_csv(args.output_root/'one_threshold_all_conditions.csv', index=False)
    fixed.groupby(['model','strategy'], sort=False)[['fpr','fnr','ber','auc']].mean().reset_index().to_csv(
        args.output_root/'fixed_condition_means.csv', index=False)
    audit = {'protocol':PROTOCOL,'warning':WARNING,'analysis_code_sha256':digest(Path(__file__)),
             'models':args.models,'strategies':args.strategies,'references':args.references,
             'conditions':CONDITIONS,'expected_images':args.expected_images,
             'score':'stored FP32 raw_margin = fake logit - real logit; fake iff margin >= threshold',
             'fixed_threshold':0,'curve':'all deterministic tie-preserving cutoffs; no interpolation/randomization',
             'matching':'FNR <= reference FNR (integer FN count); likewise FPR <= reference FPR',
             'strict_improvement':'at least one fewer FP AND one fewer FN on the same images',
             'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':version('scikit-learn'),
             'duplicate_score_absolute_tolerance':1e-12,
             'input_files':sources,'skipped':skipped,'training':False,'inference':False}
    (args.output_root/'analysis_protocol.json').write_text(json.dumps(audit,indent=2)+'\n')
    if args.plots:
        figures(records, args)
    print(WARNING, flush=True)
    print('Saved:', args.output_root, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-roots', nargs='+', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    p.add_argument('--references', nargs='+', choices=['baseline','augmentation'], default=['baseline','augmentation'])
    p.add_argument('--expected-images', type=int, default=8288)
    p.add_argument('--plots', action='store_true')
    args = p.parse_args()
    for field in ['models','strategies','references']:
        if len(getattr(args,field)) != len(set(getattr(args,field))):
            p.error(f'Duplicate --{field}')
    if args.expected_images < 0 or not {'baseline','augmentation','mean_teacher'}.issubset(args.strategies):
        p.error('Require baseline, augmentation, mean_teacher; expected-images must be nonnegative')
    analyze(args)


if __name__ == '__main__':
    main()
