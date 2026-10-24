# Nonskin_Fusion_On_Rank (세 백본 한 번에, 학습 없음)

Baseline 위 피부 밖 융합은 Xception(face_inner)·CLIP에서 통과했다. 하지만 그 효과는 Baseline의 큰 스무딩 쏠림을 **희석**한 것일 수 있다.
여기서는 **이미 스무딩 증강을 한 `aug_pair_rank` 모델 위에 얹어도 추가 이득이 남는가**를 본다. 남아야 rank + 피부 가림 보조 손실 학습 파일럿의 근거가 된다.

- 전체 얼굴 점수: v2 대각선 `aug_pair_rank@bilateral` seed42(본 실험·파일럿에서 이미 학습된 run), cdf_dev 2,000장 원본 + 4필터 × 약·중·강.
- 피부 밖 점수: 지난번과 동일(Baseline 특징 + FF++ train 2,000장 로지스틱 probe; `noskin`, `face_inner`).
- 융합: (1−w)·z(rank 점수) + w·z(피부 밖), z는 FF++ val 원본 통계, w ∈ {0, .1, …, .5}. **w=0 = rank 단독**.
- **사전 판정 기준(2026-10-25, 결과 전 확정):**
  - 주 변형은 `face_inner`(배경 지름길 배제). `noskin`은 참고로만 보고한다.
  - 통과 = 어떤 w>0에서 원본 AUC가 rank 단독 대비 −0.01 이내이면서, 4개 필터 중 3개 이상에서 나쁜 쪽 교차 AUC가 rank 단독 대비 +0.02 이상.
  - w를 cdf_dev에서 고르는 상한 검사: face_inner가 불합격인 백본은 이 방향을 접고, 통과한 백본만 학습 파일럿 대상.
  - bilateral 학습 run 기준이므로 bilateral은 학습 필터, 나머지 3개(WMF·FGS·DT)는 처음 보는 필터다.
- CDF 공식 test 미사용. 런타임: GPU. 얼굴 파싱 마스크는 런타임이 새로 시작되면 다시 만든다(수 분).

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
assert (PROJECT/'scripts/diagnose_nonskin_fusion.py').is_file(),'새 코드를 GitHub에 push하세요.'
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
for t in ['ffpp_trainval_mask.tar',*[f'cdf_dev_{p}.tar' for p in ['mask','bilateral','weighted_median','fast_global','domain_transform']]]:
    assert (ARCHIVE/t).is_file(),f'캐시 노트북을 먼저 실행하세요: {t}'
for m in MODELS:
    run=RUNS_PARENT/f'ffpp_c23_smoothing_v2_diagonal_{m}'/f'{m}_{TRAIN_FILTER}_aug_pair_rank_ffpp_c23_smoothing_v2_diagonal_v1_seed42'
    assert (run/'completed.json').is_file(),f'완료된 rank run이 없습니다: {run}'
print('Models:',MODELS,'| rank train filter:',TRAIN_FILTER,'| out:',OUT)
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

## Cell 4 — 실행 (파싱 + 세 백본, 약 40~70분)

```python
!python -u "{PROJECT}/scripts/diagnose_nonskin_fusion.py" --full-model rank --runs-parent "{RUNS_PARENT}" --train-filter {TRAIN_FILTER} --source-root "{FF_ROOT}" --cdf-root "{CDF_ROOT}" --membership "{MEMBERSHIP}" --baseline-runs-root "{BASELINE}" --archive-dir "{ARCHIVE}" --output-root "{OUT}" --models {' '.join(MODELS)} --train-images 2000 --val-images 1200 --cdf-images 2000 --batch-size 64 --workers 2
assert _exit_code==0,'rank 위 융합 확인 실패'
```

## Cell 5 — 결과

판정표(`passed`, 주 변형 face_inner) + w별 원본 AUC·필터별 나쁜 쪽 교차 AUC. w=0 행이 rank 단독.

```python
import pandas as pd
stem=f'nonskin_fusion_on_rank_{TRAIN_FILTER}'
display(pd.read_csv(OUT/f'{stem}_verdict_seed42.csv').round(3))
t=pd.read_csv(OUT/f'{stem}_seed42.csv')
filters=['bilateral','weighted_median','fast_global','domain_transform']
w=t.pivot_table(index=['model','variant','w'],columns='filter',values='worse_cross_auc',sort=False)[filters]
w.insert(0,'clean_auc',t.groupby(['model','variant','w'],sort=False).normal_auc.first())
display(w.round(3))
```
