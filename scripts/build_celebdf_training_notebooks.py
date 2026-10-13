"""Mechanical MD -> Colab notebook conversion (no experiment execution)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for model in ['xception', 'clip_b16']:
        source = ROOT / 'docs/colab' / f'train_celebdf_{model}_seed42.md'
        cells = []
        for i, text in enumerate(re.split(r'```python\n|\n```', source.read_text())):
            if not text.strip():
                continue
            kind = 'code' if i % 2 else 'markdown'
            cell = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
            if kind == 'code':
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)
        notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
                    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                 'colab': {'name': f'train_celebdf_{model}_seed42.ipynb'}, 'accelerator': 'GPU'}}
        dest = ROOT / 'notebooks' / f'train_celebdf_{model}_seed42.ipynb'
        dest.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
        print(dest)


if __name__ == '__main__':
    main()
