# Pair_NSF_xception_Seeds42_43_44

기존 pair 모델에 NSF만 결합하는 평가. 가중치 재탐색·추가 학습 없음.

## Cell 1 — 코드 설치

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys
PROJECT = Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
SCRIPT = PROJECT/'scripts/evaluate_pair_nsf_v2.py'
assert SCRIPT.is_file(), '새 코드를 GitHub에 push한 후 Cell 1을 다시 실행하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(), "GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)

```

## Cell 2 — 경로 설정

```python
MODEL = 'xception'
SEEDS = [42, 43, 44]
FILTERS = ['bilateral','weighted_median','fast_global','domain_transform']
ROOT = Path('/content/drive/MyDrive/instagram-deepfake-robustness')
RUNS = ROOT/'runs'/f'ffpp_c23_smoothing_v2_diagonal_{MODEL}'
REFERENCE = ROOT/'evaluations'/f'ffpp_c23_smoothing_v2_diagonal_{MODEL}'
OUTPUT = ROOT/'evaluations'/f'ffpp_c23_pair_nsf_v2_{MODEL}'
# Original validation images only: separate local root, no training cache or CDF TAR extraction.
FF_ROOT = Path('/content/pair_nsf_val_only')
COMMON = [
    '--model',MODEL,'--seeds',*map(str,SEEDS),'--train-filters',*FILTERS,
    '--runs-root',str(RUNS),'--evaluation-root',str(REFERENCE),'--output-root',str(OUTPUT),
    '--source-root',str(FF_ROOT),'--drive-data',str(ROOT/'data/ffpp_c23_4methods_roi16_v1'),
    '--batch-size','64','--workers','2',
]
print('Model:', MODEL, '| seeds:',SEEDS,'| output:',OUTPUT,flush=True)

```

## Cell 3 — 기존 결과 확인 및 FF++ 검증 이미지 1,200장 준비

```python
# No package/model imports in notebook kernel: Pillow installation stays isolated to subprocesses.
subprocess.run([sys.executable,'-u',str(SCRIPT),'prepare',*COMMON],check=True)

```

## Cell 4 — Pair + NSF 평가

```python
# 12 pair checkpoints: original FF++ val inference + fusion of saved CDF predictions.
# Progress is streamed by the subprocess. Completed validation statistics are reused on rerun.
subprocess.run([sys.executable,'-u',str(SCRIPT),'evaluate',*COMMON],check=True)

```

## Cell 5 — 평균 결과 및 PPT용 표

```python
import pandas as pd
from IPython.display import display, Markdown
h = pd.read_csv(OUTPUT/'pair_nsf_headline.csv')
columns = ['method','seeds','clean_auc_mean','clean_auc_sd','diag_worse_mean','heldout_worse_mean','all_worse_mean']
display(Markdown('### All methods (seed 42/43/44 mean; SD is across seed aggregates)'))
display(h[columns].round(4))
display(Markdown('### Pair + NSF vs rank + NSF, and each fusion vs its source model'))
display(pd.read_csv(OUTPUT/'ppt_tables/fusion_comparisons.csv').round(4))
for file in ['slide25_same_filter.csv','slide26_seen_unseen.csv']:
    display(Markdown('### '+file)); display(pd.read_csv(OUTPUT/'ppt_tables'/file))

```

## Cell 6 — 4×4 비교

```python
from IPython.display import Image as DisplayImage, display
# Chart was rendered by the script subprocess, avoiding Pillow state conflicts in this kernel.
display(DisplayImage(filename=str(OUTPUT/'pair_nsf_comparison.png')))
print('PPT copy/paste CSVs:', OUTPUT/'ppt_tables')
print('All three completed? The last notebook also exports slide25/26_three_backbones.csv.')

```

