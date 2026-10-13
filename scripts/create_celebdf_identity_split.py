"""New Celeb-DF V2 celebrity-ID split; never overwrite imported manifests."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

PROTOCOL = 'celebdf_v2_identity_disjoint_v1'


def identity_tokens(video):
    # Both filename IDs are protected, without inferring manipulation direction.
    return tuple(sorted(set(re.findall(r'id\d+', Path(video).name)), key=lambda x: int(x[2:])))


def partition(frame, seed=42):
    frame = frame.copy()
    frame['provided_split'] = frame['split']
    ids = sorted({i for v in frame.loc[frame.method.ne('YouTube-real'), 'video_id']
                  for i in identity_tokens(v)}, key=lambda x: int(x[2:]))
    random.Random(seed).shuffle(ids)
    ntrain, nval = round(.8 * len(ids)), round(.1 * len(ids))
    groups = {'train': ids[:ntrain], 'val': ids[ntrain:ntrain+nval], 'test': ids[ntrain+nval:]}
    membership = {i: s for s, values in groups.items() for i in values}
    def assign(row):
        if row.method == 'YouTube-real':
            return 'excluded_unknown_identity'
        tokens = identity_tokens(row.video_id)
        if not tokens:
            raise ValueError(f'Missing identity: {row.video_id}')
        splits = {membership[i] for i in tokens}
        return next(iter(splits)) if len(splits) == 1 else 'excluded_cross_identity'
    frame['identity_ids'] = frame.video_id.map(lambda v: '|'.join(identity_tokens(v)))
    frame['split'] = frame.apply(assign, axis=1)
    audit(frame)
    return frame, groups


def audit(frame):
    if frame.crop_path.duplicated().any():
        raise ValueError('Duplicate crop path')
    active = frame[frame.split.isin(['train', 'val', 'test'])]
    if active.groupby('video_id').split.nunique().gt(1).any():
        raise ValueError('Video leakage')
    idsets = {s: {i for v in g.video_id for i in identity_tokens(v)}
              for s, g in active.groupby('split')}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if idsets.get(a, set()) & idsets.get(b, set()):
            raise ValueError('Filename identity leakage')
    for s in ['train', 'val', 'test']:
        if set(active[active.split.eq(s)].class_index.astype(int)) != {0, 1}:
            raise ValueError(f'{s} needs both classes')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--allow-custom-repartition', action='store_true', required=True,
                   help='Explicitly replace official role assignments only in this new protocol.')
    args = p.parse_args()
    root = args.data_root
    full = root / 'crop_manifest.csv'
    sources = [full] if full.is_file() else sorted((root / 'manifests').glob('celebdf_v2_roi16_*.csv'))
    if not sources:
        raise FileNotFoundError(f'Full crop manifest or all three TAR manifests required: {root}')
    frame = pd.concat([pd.read_csv(x, dtype=str, keep_default_na=False) for x in sources], ignore_index=True)
    if len(frame) != 104449 or frame.video_id.nunique() != 6529:
        raise ValueError('Incomplete V2 crop pool: extract all three whole-dataset TARs, not pilot TAR')
    result, groups = partition(frame, args.split_seed)
    result = result.sort_values(['video_id', 'frame_index'], kind='stable').reset_index(drop=True)
    blob = result.to_csv(index=False, lineterminator='\n').encode()
    dest = args.output_root / 'manifest.csv'
    args.output_root.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.read_bytes() != blob:
        raise RuntimeError(f'Frozen manifest differs; use a new output directory: {dest}')
    if not dest.exists():
        dest.write_bytes(blob)
    records = []
    for (split, label), g in result.groupby(['split', 'label']):
        records.append({'split': split, 'label': label, 'images': len(g), 'videos': g.video_id.nunique()})
    metadata = {'protocol': PROTOCOL, 'split_seed': args.split_seed, 'requested_identity_ratio': [80, 10, 10], 'identity_groups': groups,
                'manifest_sha256': hashlib.sha256(blob).hexdigest(), 'counts': records,
                'sources': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
                'official_test': 'Not an evaluation set in this protocol. Its original role is preserved as provided_split.',
                'warning': 'Custom repartition includes some former official-test samples in training; not comparable to official-test results.',
                'exclusions': 'YouTube-real unknown identities; fake videos connecting different identity splits.',
                'identity_limit': 'Filename IDs only, not biometric verification; prior pilot evaluation exposure is disclosed.'}
    config = args.output_root / 'protocol.json'
    if config.exists() and json.loads(config.read_text()) != metadata:
        raise RuntimeError('Existing split protocol differs; use new output directory')
    config.write_text(json.dumps(metadata, indent=2) + '\n')
    print(pd.DataFrame(records).to_string(index=False), flush=True)
    print('Frozen custom split:', dest, flush=True)


if __name__ == '__main__':
    main()
