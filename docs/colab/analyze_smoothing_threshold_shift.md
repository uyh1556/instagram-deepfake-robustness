# Smoothing_Threshold_Shift (CPU, 학습 없음)

이미 저장된 CDF test 예측(8,288장 × 4조건)만 읽어, 스무딩이 점수를 "가짜" 쪽으로 얼마나 미는지
세 백본(Xception / CLIP / DINO)에서 계산한다. **GPU 불필요**(런타임 유형: CPU). 몇 분이면 끝난다.

|지표|뜻|
|---|---|
|`skew`|가짜가 더 스무딩된 칸 AUC 평균 − 진짜가 더 스무딩된 칸 AUC 평균. 0이면 스무딩이 점수를 안 움직임|
|`strong_real_vs_normal_fake`|강 스무딩 진짜 vs 원본 가짜 AUC|
|`fpr_<조건>@0`|모델 자체 임계값(확률 0.5)에서 진짜를 가짜로 판정한 비율|
|`fpr_strong@normal5`|원본 진짜의 5%만 가짜로 판정되게 임계값을 맞췄을 때, 강 스무딩 진짜가 가짜로 판정되는 비율|
|`flip_real@normal5`|같은 진짜 이미지가 원본에서는 진짜, 강 스무딩 후에는 가짜로 뒤집힌 비율|
|`real_shift` / `fake_shift`|원본→강 스무딩 점수 이동(원본 진짜 점수 표준편차 단위)|

- 읽는 결과: Baseline·CE+Rank v1(`delta_transfer`, 세 백본), CE+Rank pooled·SDA 새 풀(pool v2, CLIP/DINO),
  CE+Rank 고정6ep·TS EMA·head 분리(교사 보호, CLIP/DINO). 파일이 없는 항목은 건너뛰고 표시한다.
- `@normal5` 임계값은 CDF test 진짜 라벨로 맞춘 **진단용**이다. 모델 선택·배포 임계값이 아니다.
- 결과: `evaluations/smoothing_threshold_shift_v1/summary.csv`. 기존 결과 파일은 읽기만 한다.

## Cell 1 — 코드 갱신

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys

PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
assert (PROJECT/'scripts/analyze_smoothing_threshold_shift.py').is_file(),'새 코드를 GitHub에 push하세요.'
```

## Cell 2 — 계산

```python
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
EVAL=ROOT/'evaluations'
OUTPUT=EVAL/'smoothing_threshold_shift_v1/summary.csv'
SOURCES={  # 전략 이름: 모델별 예측 접두사(뒤에 _<조건>.csv)
    'baseline':           lambda m: EVAL/'ffpp_c23_smoothing_delta_transfer_v1/seed42'/m/f'{m}_baseline',
    'ce_rank_v1':         lambda m: EVAL/'ffpp_c23_smoothing_delta_transfer_v1/seed42'/m/f'{m}_ce_rank',
    'ce_rank_pooled':     lambda m: EVAL/f'ffpp_c23_smoothing_rank_sda_pool_v2_{m}/seed42'/f'{m}_ce_rank_pooled',
    'sda_new_balanced':   lambda m: EVAL/f'ffpp_c23_smoothing_rank_sda_pool_v2_{m}/seed42'/f'{m}_sda_new_balanced',
    'ce_rank_fixed6':     lambda m: EVAL/f'ffpp_c23_smoothing_uda_teacher_pool_v2_{m}/seed42'/f'{m}_ce_rank_fixed',
    'ts_ema':             lambda m: EVAL/f'ffpp_c23_smoothing_uda_teacher_pool_v2_{m}/seed42'/f'{m}_ts_ema',
    'ts_decoupled_head':  lambda m: EVAL/f'ffpp_c23_smoothing_uda_teacher_pool_v2_{m}/seed42'/f'{m}_ts_decoupled_head',
}
entries=[]
for m in ['xception','clip_b16','dinov2_b14']:
    for strategy,prefix in SOURCES.items():
        p=prefix(m)
        missing=[c for c in ['normal','smooth_weak','smooth_medium','smooth_strong'] if not Path(f'{p}_{c}.csv').is_file()]
        if missing:
            print(f'건너뜀 {m} {strategy}: {p}_{missing[0]}.csv 없음')
        else:
            entries+=['--entry',f'{m},{strategy},{p}']
subprocess.run([sys.executable,str(PROJECT/'scripts/analyze_smoothing_threshold_shift.py'),*entries,'--output',str(OUTPUT)],check=True)
print('Saved:',OUTPUT)
```
