# Nonskin_Fusion_On_Pair (w 재선정, 세 백본 한 번에, 학습 없음)

제안 방법(동시 증강 + 눈·입 점수 결합, pair+NSF)의 결합 비율 w = 0.2는 **교차 순위(`aug_pair_rank`) 모델 위에서** cdf_dev로 고른 값이다.
제안 방법의 전체 얼굴 점수는 동시 증강(`aug_pair`) 모델이므로, 같은 규칙으로 **동시 증강 모델 위에서 w를 다시 고른다**.

- 전체 얼굴 점수 A: v2 대각선 `aug_pair@bilateral` seed42(본 실험에서 이미 학습된 run), cdf_dev 2,000장 원본 + 4필터 × 약·중·강.
- 눈·입 점수 B: Baseline 특징 + FF++ train 2,000장 로지스틱 probe(`face_inner`), 교차 순위 때와 같은 표본·설정.
- **B는 검출기가 받는 그 이미지에서 계산한다(`--b-on-input`)**: 원본에는 원본의 B, 보정 이미지에는 그 보정 이미지에서 눈·입 마스크를 다시 파싱해 계산한 B. 교차 순위 때 w 선정은 원본 B를 모든 조건에 재사용했는데(차이 ≤ .002로 확인됨), 이번에는 실제 배치 방식대로 계산한다.
- 융합: (1−w)·z(A) + w·z(B), z는 FF++ val 원본 통계, w ∈ {0, .1, …, .5}. **w = 0 = 동시 증강 단독**.
- **w 선정 규칙(2026-11-10, 결과 전 확정, `--select-w`)**: w = 0에서 시작해 격자를 올라가며, `face_inner`의 원본 AUC(소수 셋째 자리 반올림)가 w = 0보다 낮아지는 백본이 하나라도 생기는 첫 w 직전의 값을 고른다. 이 규칙을 교차 순위 결과(`nonskin_fusion_on_rank_bilateral_seed42.csv`)에 적용하면 w = 0.2가 나온다(DINOv2가 w = 0.3에서 −0.004).
- **결과 해석(결과 전 확정)**:
  - 선정 w = 0.2 → 지금의 pair+NSF cdf_test 결과를 그대로 쓴다(w가 자기 전체 얼굴 모델에서 확인됨).
  - 선정 w ≠ 0.2 → 그 w로 pair+NSF cdf_test를 다시 계산해 논문에 쓴다(별도 실행). w = 0.2 결과는 참고로만 남긴다.
  - 어떤 w > 0도 규칙을 통과하지 못하면 그 사실을 그대로 기록하고 결정은 사용자에게 맡긴다.
- CDF 공식 test 미사용. 런타임: GPU. 결과는 기존 `evaluations/nonskin_fusion_precheck` 폴더에 새 이름(`nonskin_fusion_on_pair_bilateral_binput_*`)으로 저장되므로 새 공유 폴더를 만들지 않는다. 런타임 하나에서 세 백본을 차례로 돈다(병렬 실행 불필요).
- 얼굴 파싱 마스크(원본 5,200장 + 보정 24,000장)는 런타임이 새로 시작되면 다시 만든다(약 20~30분). 같은 런타임에서 다시 실행하면 재사용한다.

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
assert '--b-on-input' in (PROJECT/'scripts/diagnose_nonskin_fusion.py').read_text(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
print('CPU cores:',os.cpu_count(),'| local free GB:',shutil.disk_usage('/content').free//2**30)
```

## Cell 2 — 설정·경로

```python
MODELS=['xception','clip_b16','dinov2_b14']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
ARCHIVE=ROOT/'manifests/beauty_smoothing_v2/cache'
BASELINE=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
OUT=ROOT/'evaluations/nonskin_fusion_precheck'
RUNS_PARENT=ROOT/'runs'
TRAIN_FILTER='bilateral'
FF_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
CDF_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
MEMBERSHIP=next((ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv' for m in ['clip_b16','dinov2_b14','xception']
                 if (ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv').is_file()),None)
assert MEMBERSHIP is not None,'pool v2 membership_labelled.csv를 찾지 못했습니다'
assert OUT.is_dir(),f'기존 결과 폴더가 없습니다: {OUT}'
for t in ['ffpp_trainval_mask.tar',*[f'cdf_dev_{p}.tar' for p in ['mask','bilateral','weighted_median','fast_global','domain_transform']]]:
    assert (ARCHIVE/t).is_file(),f'캐시 노트북을 먼저 실행하세요: {t}'
for m in MODELS:
    run=RUNS_PARENT/f'ffpp_c23_smoothing_v2_diagonal_{m}'/f'{m}_{TRAIN_FILTER}_aug_pair_ffpp_c23_smoothing_v2_diagonal_v1_seed42'
    assert (run/'completed.json').is_file(),f'완료된 동시 증강(aug_pair) run이 없습니다: {run}'
print('Models:',MODELS,'| pair train filter:',TRAIN_FILTER,'| out:',OUT)
```

## Cell 3 — 데이터 준비 (FF++ train/val, CDF 비테스트)

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1" --data-root "{FF_ROOT}" --trainval-only
assert _exit_code==0,'FF++ train/val 준비 실패'

def extract_cdf(name,size):
    if (CDF_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv').is_file(): return
    source=ROOT/'data/celebdf_v2_face_crops16_v1'/name
    assert source.is_file() and source.stat().st_size==size,f'TAR 확인: {source}'
    assert shutil.disk_usage('/content').free>2*size+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name; print('Copying:',name,flush=True); shutil.copy2(source,local)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
extract_cdf('celebdf_v2_roi16_non_test_real_v1.tar',826142720)
extract_cdf('celebdf_v2_roi16_non_test_fake_v1.tar',5679564800)
```

## Cell 4 — 실행 (파싱 + 세 백본, 약 1~1.5시간)

```python
!python -u "{PROJECT}/scripts/diagnose_nonskin_fusion.py" --full-model pair --variants face_inner --b-on-input --select-w --runs-parent "{RUNS_PARENT}" --train-filter {TRAIN_FILTER} --source-root "{FF_ROOT}" --cdf-root "{CDF_ROOT}" --membership "{MEMBERSHIP}" --baseline-runs-root "{BASELINE}" --archive-dir "{ARCHIVE}" --output-root "{OUT}" --models {' '.join(MODELS)} --train-images 2000 --val-images 1200 --cdf-images 2000 --batch-size 64 --workers 2
assert _exit_code==0,'동시 증강 위 융합 확인 실패'
```

## Cell 5 — 결과

w 선정표(`selected`가 True인 행이 고른 w) + w별 원본 AUC·필터별 최저 교차 AUC. w = 0 행이 동시 증강 단독.

```python
import pandas as pd
stem=f'nonskin_fusion_on_pair_{TRAIN_FILTER}_binput'
display(pd.read_csv(OUT/f'{stem}_wselect_seed42.csv').round(3))
t=pd.read_csv(OUT/f'{stem}_seed42.csv')
filters=['bilateral','weighted_median','fast_global','domain_transform']
w=t.pivot_table(index=['model','w'],columns='filter',values='worse_cross_auc',sort=False)[filters]
w.insert(0,'clean_auc',t.groupby(['model','w'],sort=False).normal_auc.first())
display(w.round(3))
```
