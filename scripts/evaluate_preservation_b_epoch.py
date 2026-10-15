"""Read-only fixed-epoch diagnostic using the unchanged B evaluation engine."""
import copy
import json
import shutil
import tempfile
from pathlib import Path

import torch

import ffpp_c23_smoothing_preservation as b


def evaluate(args):
    selected_root = args.output_root
    output = selected_root/f'diagnostic_epoch{args.epoch}'
    metadata = {}
    # Stage only tiny correction checkpoints locally. Original best/last/config
    # files stay untouched; the legacy evaluator reads the staged fixed epoch.
    with tempfile.TemporaryDirectory(prefix='preservation-fixed-epoch-') as directory:
        staged = Path(directory)
        for name in args.models:
            original = args.runs_root/b.run_name(name, args.seed)
            state = torch.load(original/'last.pt', map_location='cpu', weights_only=True)
            config = json.loads((original/'config.json').read_text())
            done = json.loads((original/'completed.json').read_text())
            if state['epoch'] != args.epoch or done['epochs_completed'] != args.epoch:
                raise ValueError(f'{name}: requested epoch {args.epoch}, last.pt is epoch {state["epoch"]}. No original files changed.')
            if (state['config'] != config or state['identity'] != b.evaluation.key(config)
                    or done['identity'] != state['identity']):
                raise ValueError('Fixed-epoch checkpoint/config mismatch')
            dest = staged/b.run_name(name, args.seed)
            dest.mkdir()
            shutil.copy2(original/'last.pt', dest/'best.pt')
            shutil.copy2(original/'config.json', dest/'config.json')
            # Only identity is consumed from this temporary completion file.
            (dest/'completed.json').write_text(json.dumps({'identity': state['identity']}))
            metadata[name] = {'checkpoint_kind': 'last.pt', 'evaluated_epoch': args.epoch,
                'training_selected_epoch': done['selected_epoch'],
                'evaluation_role': 'fixed_epoch_diagnostic',
                'source_checkpoint_sha256': b.base.common.digest(original/'last.pt')}
            del state
        adjusted = copy.copy(args)
        adjusted.runs_root, adjusted.output_root = staged, output
        b.evaluate(adjusted)
    for name in args.models:
        for report in output.glob(f'{name}_{b.STRATEGY}_*.json'):
            values = json.loads(report.read_text())
            values.update(metadata[name])
            b.base.common.json_write(report, values)
    b.base.common.json_write(output/'fixed_epoch_diagnostic.json', metadata)
    print(f'Epoch {args.epoch} diagnostic saved: {output}', flush=True)


def main():
    p = b.parser()
    p.description = __doc__
    p.add_argument('--epoch', type=int, default=3)
    args = p.parse_args()
    if args.action != 'evaluate':
        p.error('Only evaluate is supported')
    for field in ['target_root', 'cache_root', 'output_root', 'baseline_evaluation_root']:
        if getattr(args, field) is None:
            p.error(f'--{field.replace("_", "-")} required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    evaluate(args)


if __name__ == '__main__':
    main()
