# NSF_Smoothed_B_xception

pair+nsf의 점수 B를 **보정된 이미지**에서 다시 구해, 기존 결과(B=원본 이미지)와 같은지 확인. 학습 없음, 점수기·w=.2·z 통계·pair 예측은 기존 그대로.
- 기존: 원본 픽셀 + 원본에서 파싱한 마스크 (13개 조건 모두 같은 B)
- smooth_pix: 보정 픽셀 + 원본 마스크 → 픽셀 영향만
- smooth_all: 보정 픽셀 + 보정 이미지에서 다시 파싱한 마스크 → 실제 사용 상황
- 원본 조건에서 다시 계산한 B가 저장된 B와 같은지, 기존 pair+NSF 숫자가 그대로 나오는지 자동 확인
- 점수 B는 seed와 무관해서 한 번만 계산(Drive 캐시). Cell 5는 seed 42/43/44를 한 번에 평가
- 처음 실행 시간: 얼굴 분할(8,288장 × 5개 보기) + 추론, 대략 30~60분. 끊기면 Cell 1부터 다시 실행하면 끝난 보기는 건너뜀

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
SCRIPT = PROJECT/'scripts/nsf_smoothed_b_v2.py'
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
OUTPUT = ROOT/'evaluations'/f'ffpp_c23_nsf_smoothed_b_v2_{MODEL}'
BASELINE = ROOT/'runs/ffpp_c23_4methods_baseline_v1'
ARCHIVE = ROOT/'manifests/beauty_smoothing_v2/cache'
CDF_ROOT = Path('/content/celebdf_v2_face_crops16_v1')
WORKERS = max(2, min(8, os.cpu_count()))
COMMON = f'--model {MODEL} --runs-root "{RUNS}" --evaluation-root "{REFERENCE}" --pair-nsf-root "{PAIR_NSF}" --output-root "{OUTPUT}" --baseline-runs-root "{BASELINE}" --cdf-root "{CDF_ROOT}"'
for seed in [42, 43, 44]:
    assert (PAIR_NSF/'calibration'/f'seed{seed}_bilateral.json').is_file(), f'pair+nsf seed{seed} 결과가 먼저 필요합니다: {PAIR_NSF}'
for part in ['mask','bilateral','weighted_median','fast_global','domain_transform']:
    assert (ARCHIVE/f'cdf_test_{part}.tar').is_file(), f'캐시 TAR 없음: cdf_test_{part}.tar'
SCORES_DONE = (OUTPUT/'scores/identity.json').is_file() and all((OUTPUT/f'scores/cdf_test_{v}.csv').is_file()
               for v in ['normal','bilateral','weighted_median','fast_global','domain_transform'])
print('Model:',MODEL,'| B scores cached:',SCORES_DONE,'| workers:',WORKERS,flush=True)

```

## Cell 3 — CDF test 준비 (점수 B가 캐시돼 있으면 건너뜀)

```python
if not SCORES_DONE:
    name, size = 'celebdf_v2_roi16_test_v1.tar', 557015040
    if not (CDF_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv').is_file():
        source = ROOT/'data/celebdf_v2_face_crops16_v1'/name
        assert source.is_file() and source.stat().st_size==size, f'TAR 확인: {source}'
        local = Path('/content')/name; print('Copying:',name,flush=True); shutil.copy2(source,local)
        subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
    print('CDF test ready', flush=True)
else:
    print('점수 B 캐시 있음: 데이터 준비 생략', flush=True)

```

## Cell 4 — 점수 B 계산 (보기마다 얼굴 분할 → 기존 점수기로 B, 끊기면 다시 실행하면 이어서)

```python
if not SCORES_DONE:
    !python -u "{SCRIPT}" scores {COMMON} --archive-dir "{ARCHIVE}" --batch-size 32 --workers {WORKERS}
    assert _exit_code==0, '점수 B 계산 실패'
else:
    print('점수 B 캐시 사용', flush=True)

```

## Cell 5 — 결합 평가 (seed 42/43/44, 몇 초)

```python
!python -u "{SCRIPT}" evaluate {COMMON} --seeds 42 43 44
assert _exit_code==0, '평가 실패'

```

## Cell 6 — 결과 표

```python
import json
import pandas as pd
from IPython.display import display, Markdown
display(Markdown('### 원본 조건 B 재현 확인'))
print(json.loads((OUTPUT/'scores/clean_check.json').read_text()))
h = pd.read_csv(OUTPUT/'smoothed_b_headline.csv')
display(Markdown(f'### {MODEL} — 3 seed 평균, 강 보정 최저 교차 AUC'))
display(h.round(4))
display(Markdown('### 점수 B 단독 (필터별): 보정으로 B가 얼마나 움직였나, 마스크 IoU'))
display(pd.read_csv(OUTPUT/'b_compare.csv').round(4))
print('Saved:', OUTPUT)

```
