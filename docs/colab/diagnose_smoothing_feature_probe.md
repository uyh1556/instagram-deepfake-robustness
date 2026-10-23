# Smoothing_Feature_Probe (D1, GPU, 추론만)

**질문:** bilateral 짝으로 학습한 증강 모델(aug_single / aug_pair / aug_pair_rank)은 스무딩에 **둔감해진** 것인가, 아니면 스무딩을 **감지해서 점수를 보정**하는 법을 배운 것인가?
처음 보는 필터에서 나온 과보정(domain transform에서 미탐 방향 역전)이 후자를 시사한다. 이 진단으로 직접 확인한다.

- 데이터: **FF++ val** 프레임 1,200장(진짜·가짜 반반, 얼굴 검출 성공분). CDF와 모든 test는 쓰지 않는다.
- 모델: v1 source ablation seed42 체크포인트 5개(Baseline, clean_only, aug_single, aug_pair, aug_pair_rank) × 세 백본. 학습 없음.
- 뷰: 원본 + v1 스무딩 강(0.75) 4개 — 학습에 쓴 bilateral, 처음 보는 weighted median / fast global / domain transform(v1 보정값, 기하 마스크).
- 지표(분류 직전 특징 기준):
  - `probe_auc`: 특징만 보고 "스무딩됐나"를 맞히는 선형 분류기 AUC(영상 단위 5겹 교차검증). 0.5 = 스무딩 흔적 없음.
  - `transfer_auc`: **bilateral**로 학습한 그 분류기가 다른 필터에도 반응하는가. 높으면 bilateral 감지기가 그 필터에서도 켜진다.
  - `feature_shift`: 스무딩이 특징을 얼마나 움직이는가(특징 퍼짐 대비).
  - `margin_shift_real/fake`: 그 결과 점수가 얼마나 움직였는가.
- 해석: 보정을 배웠다면 증강 모델의 `probe_auc`가 Baseline보다 **높거나 비슷하게 높고**, bilateral 점수 이동은 0 근처, domain transform 점수 이동은 음수(진짜 쪽)다. 진짜 둔감해졌다면 `probe_auc`가 0.5 쪽으로 내려간다.
- 결과: Drive `evaluations/smoothing_feature_probe_v1/feature_probe.csv`.

## Cell 1 — 코드·FF++ val 준비·OpenCV contrib

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
assert (PROJECT/'scripts/diagnose_smoothing_feature_probe.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','opencv-contrib-python-headless==4.11.0.86'],check=True)
subprocess.run([sys.executable,'-c','import cv2,torch; assert hasattr(cv2,"ximgproc"),"contrib 미설치"; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(cv2.__version__, torch.cuda.get_device_name(0))'],check=True)

ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'
```

## Cell 2 — 진단 (세 백본, 약 30분)

```python
OUTPUT=ROOT/'evaluations/smoothing_feature_probe_v1'
for m in ['xception','clip_b16','dinov2_b14']:
    !python -u "{PROJECT}/scripts/diagnose_smoothing_feature_probe.py" --source-root "{SOURCE_ROOT}" --baseline-runs-root "{ROOT}/runs/ffpp_c23_4methods_baseline_v1" --runs-root "{ROOT}/runs/ffpp_c23_smoothing_source_ablation_{m}" --calibration "{ROOT}/manifests/beauty_filter_calibration_v1/calibration.json" --output-root "{OUTPUT}/{m}" --models {m} --images 1200 --batch-size 64 --workers 2
    assert _exit_code==0,f'{m} 진단 실패: 위 오류 확인'
```

## Cell 3 — 결과 표시

```python
import pandas as pd
from IPython.display import Markdown,display
t=pd.concat([pd.read_csv(OUTPUT/m/'feature_probe.csv') for m in ['xception','clip_b16','dinov2_b14']])
order=['baseline','clean_only','aug_single','aug_pair','aug_pair_rank']
for metric,title in [('probe_auc','스무딩 감지 AUC (0.5 = 흔적 없음)'),('transfer_auc','bilateral 감지기가 이 필터에도 켜지나'),
                     ('feature_shift','특징 이동량'),('margin_shift_real','진짜 점수 이동 (음수 = 진짜 쪽)'),('margin_shift_fake','가짜 점수 이동')]:
    p=t.pivot_table(index=['model','strategy'],columns='filter',values=metric).reindex(columns=['bilateral','weighted_median','fast_global','domain_transform'])
    p=p.reindex([(m,s) for m in ['xception','clip_b16','dinov2_b14'] for s in order])
    display(Markdown(f'### {title}')); display(p.round(3))
```
