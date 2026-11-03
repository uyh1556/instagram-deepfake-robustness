# End2End_Retouch_GFPGAN_xception

GFPGAN으로 보정한 CDF test에서 Baseline / pair / pair+NSF 평가. **학습 없음**, 체크포인트·w=.2·점수기·z 통계 모두 기존 그대로.
- 먼저 `build_end2end_retouch_v1` 노트북으로 `cdf_test_gfpgan.tar`를 만들어 둘 것
- Baseline seed42 1개 + pair 4필터 × seed 42/43/44 = 13개 모델의 GFPGAN 보기만 추론(원본 예측은 기존 평가 파일 재사용, 원본 AUC 재현 자동 확인)
- pair+NSF 점수 B 3가지: 원본 이미지(낙관적) / 보정 픽셀 + 원본 마스크 / **보정 이미지에서 다시 파싱(실제 사용)**
- 지표: 원본 AUC, 교차 AUC 두 방향과 나쁜 쪽(주), 둘 다 보정했을 때 AUC, 점수 이동, 뒤집힘. 피부 vs 눈·입 변화량도 기록
- 끊기면 Cell 1부터 다시 실행하면 끝난 추론은 건너뜀. 세 백본 노트북을 동시에 실행해도 됨

## Cell 1 — 코드 설치

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, os, shutil
PROJECT = Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
SCRIPT = PROJECT/'scripts/end2end_retouch_v1.py'
assert SCRIPT.is_file(), '새 코드를 GitHub에 push한 후 Cell 1을 다시 실행하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(), "GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
print('CPU cores:',os.cpu_count(),'| local free GB:',shutil.disk_usage('/content').free//2**30)

```

## Cell 2 — 설정

```python
MODEL = 'xception'
ROOT = Path('/content/drive/MyDrive/instagram-deepfake-robustness')
RUNS = ROOT/'runs'/f'ffpp_c23_smoothing_v2_diagonal_{MODEL}'
REFERENCE = ROOT/'evaluations'/f'ffpp_c23_smoothing_v2_diagonal_{MODEL}'
PAIR_NSF = ROOT/'evaluations'/f'ffpp_c23_pair_nsf_v2_{MODEL}'
OUTPUT = ROOT/'evaluations'/f'ffpp_c23_end2end_retouch_v1_{MODEL}'
BASELINE = ROOT/'runs/ffpp_c23_4methods_baseline_v1'
ARCHIVE = ROOT/'manifests/end2end_retouch_v1/cache'
V2_ARCHIVE = ROOT/'manifests/beauty_smoothing_v2/cache'
CDF_ROOT = Path('/content/celebdf_v2_face_crops16_v1')
WORKERS = max(2, min(8, os.cpu_count()))
assert (ARCHIVE/'cdf_test_gfpgan.tar').is_file(), 'build_end2end_retouch_v1 노트북을 먼저 실행하세요'
assert (V2_ARCHIVE/'cdf_test_mask.tar').is_file(), 'v2 피부 마스크 캐시가 필요합니다'
for seed in [42, 43, 44]:
    assert (PAIR_NSF/'calibration'/f'seed{seed}_bilateral.json').is_file(), f'pair+nsf seed{seed} 결과가 먼저 필요합니다'
print('Model:', MODEL, '| workers:', WORKERS, flush=True)

```

## Cell 3 — CDF test 준비

```python
name, size = 'celebdf_v2_roi16_test_v1.tar', 557015040
if not (CDF_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv').is_file():
    source = ROOT/'data/celebdf_v2_face_crops16_v1'/name
    assert source.is_file() and source.stat().st_size==size, f'TAR 확인: {source}'
    local = Path('/content')/name; print('Copying:',name,flush=True); shutil.copy2(source,local)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
print('CDF test ready', flush=True)

```

## Cell 4 — 평가 (GFPGAN 보기 추론 13개 + 점수 B + 융합)

```python
!python -u "{SCRIPT}" evaluate --model {MODEL} --runs-root "{RUNS}" --evaluation-root "{REFERENCE}" --pair-nsf-root "{PAIR_NSF}" --output-root "{OUTPUT}" --baseline-runs-root "{BASELINE}" --archive-dir "{ARCHIVE}" --v2-archive-dir "{V2_ARCHIVE}" --cdf-root "{CDF_ROOT}" --seeds 42 43 44 --batch-size 32 --workers {WORKERS}
assert _exit_code==0, '평가 실패'

```

## Cell 5 — 결과 표

```python
import pandas as pd
from IPython.display import display, Markdown
display(Markdown(f'### {MODEL} — GFPGAN 보정 CDF test (Baseline seed42, pair/pair+NSF는 4필터 × 3 seed 평균)'))
display(pd.read_csv(OUTPUT/'headline.csv').set_index('method').round(4))
display(Markdown('### 학습 필터별 (3 seed 평균)'))
display(pd.read_csv(OUTPUT/'by_train_filter.csv').pivot(index='train_filter', columns='method', values='worse_cross_auc').round(4))
display(Markdown('### 점수 B 단독: GFPGAN이 B를 얼마나 움직였나, 마스크 IoU'))
display(pd.read_csv(OUTPUT/'b_compare.csv').set_index('variant').round(4))
r = pd.read_csv(OUTPUT/'region_change.csv')
display(Markdown(f'### 영역별 평균 RGB 변화 (0~255): 피부 {r.skin.mean():.2f} | 눈·입 {r.eyes_mouth.mean():.2f} | 나머지 {r.rest.mean():.2f}'))
print('Saved:', OUTPUT)

```
