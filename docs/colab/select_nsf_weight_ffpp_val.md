# Select_NSF_Weight_FFpp_Val (w를 FF++ 검증셋으로 다시 고르기, 세 백본 한 번에, 학습 없음)

제안 방법(동시 증강 + 눈·입 점수 결합)의 w = 0.2는 Celeb-DF 개발세트(테스트 외 영상, 라벨 사용)로 고른 값이다.
평가 도메인의 라벨을 쓰지 않아도 같은 값이 나오는지 보기 위해, **같은 규칙을 FF++ 검증 이미지에만** 적용한다. Celeb-DF 이미지는 전혀 쓰지 않는다.

- 전체 얼굴 점수 A: v2 대각선 `aug_pair@bilateral` seed42(Celeb-DF 개발세트 선정 때와 같은 run).
- 눈·입 점수 B: Baseline 특징 + FF++ train 2,000장 로지스틱 probe(`face_inner`), 이전과 같은 표본·설정.
- z: 이전과 같은 FF++ val 1,200장(균형, seed42) 원본 통계.
- 융합: (1−w)·z(A) + w·z(B), w ∈ {0, .1, …, .5}. **w = 0 = 동시 증강 단독**.
- 선정 집합: 얼굴이 검출된 **FF++ val 전체 이미지**(원본만; 규칙이 원본 AUC만 보므로 보정 이미지는 필요 없음). 원본 AUC를 전체와 조작 기법별로 저장한다.
- **w 선정 규칙(2026-11-10, 결과 전 확정)**: Celeb-DF 개발세트 때와 같다. w = 0부터 격자를 올라가며, 원본 AUC(소수 셋째 자리)가 w = 0보다 낮아지는 백본이 하나라도 생기는 첫 w 직전 값.
- **결과 해석(결과 전 확정)**: 학위논문은 이미 고정한 w = 0.2(Celeb-DF 개발세트)를 그대로 쓴다. 이 결과는 학습 도메인 기준 확인으로 보고한다. 다른 w가 나오면 두 결과를 함께 보고하고, 테스트 추가 평가 여부는 따로 정한다.
- 주의: FF++ val은 원본 AUC가 높아(천장 근처) 변화가 작게 나타날 수 있다. 그 경우도 결과 그대로 기록한다.
- 결과는 기존 `evaluations/nonskin_fusion_precheck` 폴더에 `nonskin_fusion_on_pair_bilateral_ffppval_*`로 저장되므로 새 공유 폴더를 만들지 않는다. 런타임 하나에서 세 백본을 차례로 돈다. 런타임: GPU, 약 30~60분(얼굴 파싱 포함).

## Cell 1 — 설치·코드 갱신

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, shutil, os

PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
assert (PROJECT/'scripts/select_nsf_weight_ffpp_val.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
print('CPU cores:',os.cpu_count(),'| local free GB:',shutil.disk_usage('/content').free//2**30)
```

## Cell 2 — 설정·경로

```python
MODELS=['xception','clip_b16','dinov2_b14']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
BASELINE=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
OUT=ROOT/'evaluations/nonskin_fusion_precheck'
RUNS_PARENT=ROOT/'runs'
TRAIN_FILTER='bilateral'
FF_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
assert OUT.is_dir(),f'기존 결과 폴더가 없습니다: {OUT}'
for m in MODELS:
    run=RUNS_PARENT/f'ffpp_c23_smoothing_v2_diagonal_{m}'/f'{m}_{TRAIN_FILTER}_aug_pair_ffpp_c23_smoothing_v2_diagonal_v1_seed42'
    assert (run/'completed.json').is_file(),f'완료된 동시 증강(aug_pair) run이 없습니다: {run}'
print('Models:',MODELS,'| pair train filter:',TRAIN_FILTER,'| out:',OUT)
```

## Cell 3 — 데이터 준비 (FF++ train/val만)

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1" --data-root "{FF_ROOT}" --trainval-only
assert _exit_code==0,'FF++ train/val 준비 실패'
```

## Cell 4 — 실행 (파싱 + 세 백본)

```python
!python -u "{PROJECT}/scripts/select_nsf_weight_ffpp_val.py" --runs-parent "{RUNS_PARENT}" --train-filter {TRAIN_FILTER} --source-root "{FF_ROOT}" --baseline-runs-root "{BASELINE}" --output-root "{OUT}" --models {' '.join(MODELS)} --train-images 2000 --val-images 1200 --select-images 0 --batch-size 64 --workers 2
assert _exit_code==0,'FF++ val w 선정 실패'
```

## Cell 5 — 결과

w 선정표(`selected`가 True인 행이 고른 w) + w별 원본 AUC(전체·조작 기법별). w = 0 행이 동시 증강 단독.

```python
import pandas as pd
stem=f'nonskin_fusion_on_pair_{TRAIN_FILTER}_ffppval'
display(pd.read_csv(OUT/f'{stem}_wselect_seed42.csv').round(3))
t=pd.read_csv(OUT/f'{stem}_seed42.csv')
cols=['model','w','normal_auc',*[c for c in t if c.startswith('auc_')],'nonskin_alone_auc','frames']
display(t[cols].round(4))
```
