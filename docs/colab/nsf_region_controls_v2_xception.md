# NSF_Region_Controls_xception

pair+nsf의 점수 B 영역만 바꿔 비교하는 대조 실험. 학습 없음, 가중치·설정은 pair+nsf 그대로(w=.2).
- 대조군: 이미지 전체 / 얼굴 전체(피부 포함, 귀·머리카락·배경 제외) / 랜덤 패치(눈·입과 같은 면적, 얼굴 안)
- 참고: 눈·입(제안) 결과는 기존 pair+nsf와 숫자가 정확히 같은지 자동 확인
- 점수 B는 seed와 무관해서 한 번만 계산(Drive 캐시). seed 43/44는 `SEED`만 바꾸고 실행하면 Cell 3·4는 자동으로 건너뜀
- 처음 실행 시간: 얼굴 분할 + 추론(CDF test 8,288장 × 원본·강 보정 4종 × 영역 3개) 대략 30~60분

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
SCRIPT = PROJECT/'scripts/nsf_region_controls_v2.py'
assert SCRIPT.is_file(), '새 코드를 GitHub에 push한 후 Cell 1을 다시 실행하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(), "GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
print('CPU cores:',os.cpu_count(),'| local free GB:',shutil.disk_usage('/content').free//2**30)

```

## Cell 2 — 설정 (seed만 바꿔서 재사용)

```python
MODEL = 'xception'
SEED = 42                      # 43, 44도 이 값만 바꿔서 Cell 2부터 다시 실행
ROOT = Path('/content/drive/MyDrive/instagram-deepfake-robustness')
RUNS = ROOT/'runs'/f'ffpp_c23_smoothing_v2_diagonal_{MODEL}'
REFERENCE = ROOT/'evaluations'/f'ffpp_c23_smoothing_v2_diagonal_{MODEL}'
PAIR_NSF = ROOT/'evaluations'/f'ffpp_c23_pair_nsf_v2_{MODEL}'
OUTPUT = ROOT/'evaluations'/f'ffpp_c23_nsf_region_controls_v2_{MODEL}'
BASELINE = ROOT/'runs/ffpp_c23_4methods_baseline_v1'
ARCHIVE = ROOT/'manifests/beauty_smoothing_v2/cache'
FF_ROOT = Path('/content/nsf_region_ffpp')          # FF++ 3,200장(원래 NSF 표본)만 추출
CDF_ROOT = Path('/content/celebdf_v2_face_crops16_v1')
WORKERS = max(2, min(8, os.cpu_count()))
COMMON = f'--model {MODEL} --runs-root "{RUNS}" --evaluation-root "{REFERENCE}" --pair-nsf-root "{PAIR_NSF}" --output-root "{OUTPUT}" --baseline-runs-root "{BASELINE}" --source-root "{FF_ROOT}" --cdf-root "{CDF_ROOT}"'
assert (PAIR_NSF/'calibration'/f'seed{SEED}_bilateral.json').is_file(), f'pair+nsf seed{SEED} 결과가 먼저 필요합니다: {PAIR_NSF}'
for part in ['mask','bilateral','weighted_median','fast_global','domain_transform']:
    assert (ARCHIVE/f'cdf_test_{part}.tar').is_file(), f'캐시 TAR 없음: cdf_test_{part}.tar'
SCORES_DONE = (OUTPUT/'scores/probes.json').is_file() and all((OUTPUT/f'scores/cdf_test_{v}.csv').is_file()
               for v in ['normal','bilateral','weighted_median','fast_global','domain_transform'])
print('Model:',MODEL,'| seed:',SEED,'| region scores cached:',SCORES_DONE,'| workers:',WORKERS,flush=True)

```

## Cell 3 — 데이터 준비 (점수 B가 캐시돼 있으면 건너뜀)

```python
if not SCORES_DONE:
    !python -u "{SCRIPT}" prepare {COMMON} --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1"
    assert _exit_code==0, 'FF++ 표본 준비 실패'
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

## Cell 4 — 점수 B 계산 (얼굴 분할 → 분류기 학습 → CDF 5개 보기 점수, 끊기면 다시 실행하면 이어서)

```python
if not SCORES_DONE:
    !python -u "{SCRIPT}" scores {COMMON} --archive-dir "{ARCHIVE}" --batch-size 32 --workers {WORKERS}
    assert _exit_code==0, '점수 B 계산 실패'
else:
    print('점수 B 캐시 사용', flush=True)

```

## Cell 5 — 결합 평가 (seed별, 몇 초)

```python
!python -u "{SCRIPT}" evaluate {COMMON} --seed {SEED}
assert _exit_code==0, '평가 실패'

```

## Cell 6 — 결과 표

```python
import pandas as pd
from IPython.display import display, Markdown
h = pd.read_csv(OUTPUT/'region_headline.csv')
display(Markdown(f'### {MODEL} — 끝난 seed 평균, 강 보정 최저 교차 AUC'))
display(h.round(4))
display(Markdown('### 점수 B 단독 (4개 필터 평균)'))
display(pd.read_csv(OUTPUT/'b_alone.csv').groupby('region',sort=False)[['ffpp_val_auc','clean_auc','worse_cross_auc','real_shift','fake_shift']].mean().round(4))
print('Saved:', OUTPUT)

```
