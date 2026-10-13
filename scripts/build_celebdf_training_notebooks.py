"""Mechanical MD -> Colab notebook conversion (no experiment execution)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for stem in ['train_celebdf_xception_seed42', 'train_celebdf_clip_b16_seed42', 'evaluate_celebdf_smoothing_seed42']:
        source = ROOT / 'docs/colab' / f'{stem}.md'
        cells = []
        for i, text in enumerate(re.split(r'^```python\n|^```[ \t]*$', source.read_text(), flags=re.MULTILINE)):
            if not text.strip():
                continue
            kind = 'code' if i % 2 else 'markdown'
            cell = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
            if kind == 'code':
                if text.lstrip().splitlines()[0] == 'python':
                    raise ValueError('Markdown language tag leaked into code cell')
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)
        notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
                    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                 'colab': {'name': f'{stem}.ipynb'}, 'accelerator': 'GPU'}}
        dest = ROOT / 'notebooks' / f'{stem}.ipynb'
        dest.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
        print(dest)


if __name__ == '__main__':
    main()
