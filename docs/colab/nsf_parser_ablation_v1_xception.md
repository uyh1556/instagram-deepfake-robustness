# NSF_Parser_Ablation_xception

pair+NSF의 점수 B를 만드는 눈·입 마스크의 parser만 바꿔 비교. 학습 없음, w=.2·pair 예측·FF++ 표본은 pair+NSF 그대로.
- 비교 순서: SegFormer(원본) → SegFormer 마스크 축소 1% → 확대 1% → BiSeNet → MediaPipe Face Mesh (마지막 줄은 참고용 pair)
- 점수 B는 탐지기 입력과 같은 이미지에서 계산(보정 이미지면 그 이미지에서 다시 파싱한 마스크)
- SegFormer는 기존 점수기를 그대로 쓰고, 나머지는 같은 FF++ 표본에서 자기 마스크로 점수기를 새로 학습
- 사전 판정 기준(결과 전 고정): pair보다 좋은 run ≥ 10/12, 16칸 평균 최저 교차 AUC가 SegFormer와 .01 이내면 "유지"
- 먼저 `build_nsf_parser_masks_v1`로 마스크 tar 3개가 있어야 함. seed 42/43/44를 한 번에 평가
- 처음 실행 시간: CDF test 8,288장 × 5개 보기 × 변형 5개 추론, 대략 30~60분. 끊기면 Cell 1부터 다시 실행하면 끝난 보기는 건너뜀

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
SCRIPT = PROJECT/'scripts/nsf_parser_ablation_v1.py'
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
OUTPUT = ROOT/'evaluations'/f'ffpp_c23_nsf_parser_ablation_v1_{MODEL}'
BASELINE = ROOT/'runs/ffpp_c23_4methods_baseline_v1'
ARCHIVE = ROOT/'manifests/beauty_smoothing_v2/cache'
MASK_DIR = ROOT/'manifests/nsf_parser_ablation_v1/cache'
FF_ROOT = Path('/content/nsf_parser_ffpp')
CDF_ROOT = Path('/content/celebdf_v2_face_crops16_v1')
WORKERS = max(2, min(8, os.cpu_count()))
COMMON = f'--model {MODEL} --runs-root "{RUNS}" --evaluation-root "{REFERENCE}" --pair-nsf-root "{PAIR_NSF}" --output-root "{OUTPUT}" --baseline-runs-root "{BASELINE}" --source-root "{FF_ROOT}" --cdf-root "{CDF_ROOT}"'
for seed in [42, 43, 44]:
    assert (PAIR_NSF/'calibration'/f'seed{seed}_bilateral.json').is_file(), f'pair+nsf seed{seed} 결과가 먼저 필요합니다: {PAIR_NSF}'
for part in ['mask','bilateral','weighted_median','fast_global','domain_transform']:
    assert (ARCHIVE/f'cdf_test_{part}.tar').is_file(), f'캐시 TAR 없음: cdf_test_{part}.tar'
for f in ['segformer','bisenet','mediapipe']:
    assert (MASK_DIR/f'parser_masks_{f}.tar').is_file(), f'마스크 tar 없음: build_nsf_parser_masks_v1부터 실행 ({f})'
SCORES_DONE = (OUTPUT/'scores/probes.json').is_file() and all((OUTPUT/f'scores/cdf_test_{v}.csv').is_file()
               for v in ['normal','bilateral','weighted_median','fast_global','domain_transform'])
print('Model:',MODEL,'| B scores cached:',SCORES_DONE,'| workers:',WORKERS,flush=True)

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

## Cell 4 — 점수 B 계산 (마스크 복원 → 변형별 점수기 → CDF 5개 보기, 끊기면 다시 실행하면 이어서)

```python
if not SCORES_DONE:
    !python -u "{SCRIPT}" scores {COMMON} --archive-dir "{ARCHIVE}" --mask-dir "{MASK_DIR}" --batch-size 16 --workers {WORKERS}
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
display(Markdown('### SegFormer 원본 조건 B 재현 확인'))
print(json.loads((OUTPUT/'scores/clean_check.json').read_text()))
display(Markdown(f'### {MODEL} — 3 seed 평균, 강 보정 최저 교차 AUC (holds = 사전 기준 통과)'))
display(pd.read_csv(OUTPUT/'parser_headline.csv').round(4))
display(Markdown('### 마스크 비교 (SegFormer 대비, 검출 프레임)'))
display(pd.read_csv(OUTPUT/'mask_agreement.csv').round(3))
display(Markdown('### 점수 B 단독'))
display(pd.read_csv(OUTPUT/'b_alone.csv').groupby('variant', sort=False).mean(numeric_only=True).round(4))
print('Saved:', OUTPUT)

```
