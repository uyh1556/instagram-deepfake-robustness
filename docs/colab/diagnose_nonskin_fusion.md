# Nonskin_Fusion_Precheck (세 백본 한 번에, 학습 없음)

피부 가림 사전 확인에서 **피부 밖 단서만으로 피부를 대체할 수는 없다**(cdf_dev probe −.14~−.16)고 나왔다. 여기서는 다른 질문을 본다:
**피부 밖 점수가 전체 얼굴 점수를 보조할 수 있는가.** v2 스무딩은 피부 마스크 안만 바꾸므로, 피부 없이 낸 점수는 어떤 v2 필터에도(처음 보는 DT 포함) 변하지 않는다.

- 피부 밖 입력 두 가지(각각 Baseline 특징 + FF++ train 2,000장으로 맞춘 로지스틱 probe):
  - `noskin`: 피부(+코) 마스크만 회색. 머리카락·배경 포함
  - `face_inner`: 눈·눈썹·입술·입 안만 남기고 나머지 전부 회색(같은 SegFormer 파싱, 프레임당 한 번, `/content/nonskin_fusion_work`에 저장)
- 전체 얼굴 점수: Baseline으로 cdf_dev 2,000장의 원본 + 4개 필터 × 약·중·강.
- 융합: (1−w)·z(전체 얼굴) + w·z(피부 밖), z는 FF++ val 원본 통계. w ∈ {0, .1, .2, .3, .4, .5}.
- **사전 판정 기준(2026-10-25, 결과 전 확정):** 어떤 w>0에서 원본 AUC가 w=0 대비 −0.01 이내이면서, 4개 필터 중 3개 이상에서 나쁜 쪽 교차 AUC가 +0.02 이상 오르면 통과.
  w를 cdf_dev 위에서 고르므로 **상한(oracle) 검사**다: 불합격이면 이 방향은 접고, 통과하면 학습 파일럿을 할 근거만 생긴다.
- `face_inner`에서도 통과해야 "얼굴 안 합성 단서가 보조한다"고 말할 수 있다(`noskin`만 통과하면 배경 지름길일 수 있음).
- CDF 공식 test 미사용. 런타임: GPU. 캐시: FF++ 마스크 + cdf_dev 5개 tar.

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
FF_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
CDF_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
MEMBERSHIP=next((ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv' for m in ['clip_b16','dinov2_b14','xception']
                 if (ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv').is_file()),None)
assert MEMBERSHIP is not None,'pool v2 membership_labelled.csv를 찾지 못했습니다'
for t in ['ffpp_trainval_mask.tar',*[f'cdf_dev_{p}.tar' for p in ['mask','bilateral','weighted_median','fast_global','domain_transform']]]:
    assert (ARCHIVE/t).is_file(),f'캐시 노트북을 먼저 실행하세요: {t}'
print('Models:',MODELS,'| out:',OUT)
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

## Cell 4 — 실행 (파싱 + 세 백본, 약 30~60분)

```python
!python -u "{PROJECT}/scripts/diagnose_nonskin_fusion.py" --source-root "{FF_ROOT}" --cdf-root "{CDF_ROOT}" --membership "{MEMBERSHIP}" --baseline-runs-root "{BASELINE}" --archive-dir "{ARCHIVE}" --output-root "{OUT}" --models {' '.join(MODELS)} --train-images 2000 --val-images 1200 --cdf-images 2000 --batch-size 64 --workers 2
assert _exit_code==0,'융합 확인 실패'
```

## Cell 5 — 결과

판정표(`passed`) + w별 원본 AUC·필터별 나쁜 쪽 교차 AUC. w=0 행이 전체 얼굴 Baseline 그대로.

```python
import pandas as pd
display(pd.read_csv(OUT/'nonskin_fusion_verdict_seed42.csv').round(3))
t=pd.read_csv(OUT/'nonskin_fusion_seed42.csv')
filters=['bilateral','weighted_median','fast_global','domain_transform']
w=t.pivot_table(index=['model','variant','w'],columns='filter',values='worse_cross_auc',sort=False)[filters]
w.insert(0,'clean_auc',t.groupby(['model','variant','w'],sort=False).normal_auc.first())
w.insert(1,'nonskin_alone',t.groupby(['model','variant','w'],sort=False).nonskin_alone_auc.first())
display(w.round(3))
```
