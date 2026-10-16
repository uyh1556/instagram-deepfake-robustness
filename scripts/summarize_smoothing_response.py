"""Build response comparison tables from saved predictions; no model loading/inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

METRICS = ['auc', 'accuracy', 'f1', 'real_fpr', 'fake_fnr', 'balanced_error_rate']
PRIMARY = ['baseline', 'current_only', 'response']


def read_json(path):
    return json.loads(path.read_text())


def prediction_membership(path, expected):
    """Older reports lack image count; validate the actual saved predictions instead."""
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(expected):
        raise ValueError(f'Prediction count differs: {path}')
    found = {}
    for row in rows:
        key = row['crop_path']
        value = float(row['probability_fake'])
        if key in found or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Duplicate path or invalid probability: {path}')
        found[key] = (row['video_id'], row['class_index'], row['roi_status'])
    if found != expected:
        raise ValueError(f'Test image/video/label membership differs: {path}')


def metric_row(report, model, strategy, condition):
    values = {k: float(report[k]) for k in METRICS if k != 'balanced_error_rate'}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
        raise ValueError('Invalid saved metric')
    values['balanced_error_rate'] = .5*(values['real_fpr']+values['fake_fnr'])
    return dict(model=model, strategy=strategy, condition=condition, **values)


def write_table(path, rows, fields):
    temp = path.with_suffix('.tmp.csv')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def summarize(args):
    root = args.output_root
    protocol = read_json(root/'evaluation_protocol.json')
    manifest = root/'evaluation_manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != protocol['manifest_sha256']:
        raise ValueError('Saved evaluation manifest differs from protocol')
    with manifest.open(newline='') as stream:
        source_rows = list(csv.DictReader(stream))
    expected = {r['crop_path']: (r['video_id'], r['class_index'], r['roi_status']) for r in source_rows}
    if len(expected) != len(source_rows):
        raise ValueError('Duplicate test paths')
    conditions = list(protocol['config']['conditions'])
    rows, audit = [], []
    for model in args.models:
        for strategy in [*PRIMARY, 'current_only_last_diagnostic', 'response_last_diagnostic']:
            paths = [root/f'{model}_{strategy}_{c}.json' for c in conditions]
            if strategy not in PRIMARY and not any(p.exists() for p in paths):
                continue
            for condition, path in zip(conditions, paths):
                report = read_json(path)
                if (report.get('model') != model or report.get('strategy') != strategy
                        or report.get('condition') != condition or report.get('images') != len(expected)
                        or report.get('manifest_sha256') != protocol['manifest_sha256']
                        or report.get('precision') != 'FP32' or report.get('unit') != 'frame'
                        or report.get('threshold') != .5):
                    raise ValueError(f'Primary evaluation metadata differs: {path}')
                prediction_membership(path.with_suffix('.csv'), expected)
                rows.append(metric_row(report, model, strategy, condition))
        found = False
        if args.reference_evaluations is not None and args.reference_evaluations.is_dir():
            for normal in sorted(args.reference_evaluations.rglob(f'{model}_augmentation_normal.json')):
                try:
                    old_protocol = read_json(normal.parent/'evaluation_protocol.json')
                    for field in ['manifest_sha256', 'config', 'opencv', 'smoothing_code_sha256']:
                        if old_protocol.get(field) != protocol[field]:
                            raise ValueError(f'Aug test protocol differs: {field}')
                    reference_rows, formats = [], []
                    for condition in conditions:
                        path = normal.parent/f'{model}_augmentation_{condition}.json'
                        report = read_json(path)
                        if report.get('unit') != 'frame' or report.get('threshold') != .5:
                            raise ValueError('Aug unit/threshold differs')
                        if 'images' in report and report['images'] != len(expected):
                            raise ValueError('Aug image count differs')
                        if 'precision' in report and report['precision'] != 'FP32':
                            raise ValueError('Aug precision differs')
                        # Legacy adaptation evaluator used FP32 inference but did not store
                        # images/precision. Never invent those fields in the original file.
                        prediction_membership(path.with_suffix('.csv'), expected)
                        reference_rows.append(metric_row(report, model, 'augmentation', condition))
                        formats.append(dict(file=str(path), images_field=report.get('images'),
                                            precision_field=report.get('precision'),
                                            checked_images=len(expected)))
                    rows.extend(reference_rows)
                    audit.append(dict(model=model, status='included', directory=str(normal.parent), reports=formats))
                    found = True
                    print('Included saved Aug:', model, normal.parent, flush=True)
                    break
                except (OSError, ValueError, KeyError) as error:
                    audit.append(dict(model=model, status='skipped', directory=str(normal.parent), reason=str(error)))
                    print('Skip incompatible Aug reference:', model, str(error), flush=True)
        if not found:
            print('No compatible Aug reference; primary results are still summarized:', model, flush=True)
    means = []
    groups = dict.fromkeys((r['model'], r['strategy']) for r in rows)
    for model, strategy in groups:
        group = [r for r in rows if r['model'] == model and r['strategy'] == strategy]
        means.append(dict(model=model, strategy=strategy,
                          **{k: sum(r[k] for r in group)/len(group) for k in METRICS}))
    write_table(root/'comparison.csv', rows, ['model','strategy','condition',*METRICS])
    write_table(root/'condition_means.csv', means, ['model','strategy',*METRICS])
    record = dict(inference='none; saved reports/predictions only', references=audit)
    temp = root/'reference_comparison_provenance.tmp.json'
    temp.write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    temp.replace(root/'reference_comparison_provenance.json')
    print('Saved comparison:', root/'comparison.csv', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['xception','clip_b16','dinov2_b14'], default=['clip_b16'])
    parser.add_argument('--reference-evaluations', type=Path)
    summarize(parser.parse_args())


if __name__ == '__main__':
    main()
