# Skin_Occlusion_Precheck (세 백본 한 번에, 학습 없음)

피부 가림 보조 손실을 시도하기 전 사전 확인: **v2 스무딩이 걸리는 피부(+코) 마스크를 회색으로 가렸을 때 진짜/가짜 단서가 얼마나 남는가.**
눈·눈썹·입술·입 안은 가리지 않는다(네 FF++ 조작 모두의 조작 영역 안에 있음).

- 모델: FF++ Baseline seed42 (Xception / CLIP / DINO), 학습 없음.
- 입력 두 가지: 원본, 피부 가림(v2 마스크의 부드러운 경계 그대로 회색 128로 채움).
- 읽는 값 두 가지:
  - `head`: Baseline 분류기 그대로(가린 얼굴을 본 적 없으므로 하한)
  - `probe`: 분류기 입력 특징에 로지스틱 회귀를 FF++ train 2,000장(같은 입력 종류)으로 맞추고 FF++ val 1,200장·cdf_dev 2,000장에서 채점
- AUC는 전체와 FF++ 조작 방법별(진짜 vs 그 방법). **CDF 공식 test는 쓰지 않는다.**
- 판정(미리 정한 기준): `probe`의 cdf_dev AUC가 원본 대비 **−0.05 이내**면 피부 가림 보조 손실 파일럿 진행, 크게 떨어지면 이 방향은 접는다.
- 런타임: GPU(T4도 가능). 캐시는 마스크 tar 두 개만 푼다.

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
assert (PROJECT/'scripts/diagnose_skin_occlusion.py').is_file(),'새 코드를 GitHub에 push하세요.'
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
OUT=ROOT/'evaluations/skin_occlusion_precheck'
FF_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
CDF_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
MEMBERSHIP=next((ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv' for m in ['clip_b16','dinov2_b14','xception']
                 if (ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv').is_file()),None)
assert MEMBERSHIP is not None,'pool v2 membership_labelled.csv를 찾지 못했습니다'
for t in ['ffpp_trainval_mask.tar','cdf_dev_mask.tar']:
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

## Cell 4 — 실행 (세 백본 순서대로, 약 20~40분)

```python
!python -u "{PROJECT}/scripts/diagnose_skin_occlusion.py" --source-root "{FF_ROOT}" --cdf-root "{CDF_ROOT}" --membership "{MEMBERSHIP}" --baseline-runs-root "{BASELINE}" --archive-dir "{ARCHIVE}" --output-root "{OUT}" --models {' '.join(MODELS)} --train-images 2000 --val-images 1200 --cdf-images 2000 --batch-size 64 --workers 2
assert _exit_code==0,'사전 확인 실패'
```

## Cell 5 — 결과 표

`drop` = 피부 가림 − 원본. probe의 cdf_dev `all` 행이 판정 기준(−0.05 이내면 진행).

```python
import pandas as pd
t=pd.read_csv(OUT/'skin_occlusion_seed42.csv')
w=t.pivot_table(index=['model','dataset','subset','scorer'],columns='view',values='auc',sort=False)[['original','skin_occluded']]
w['drop']=w.skin_occluded-w.original
display(w.round(3))
print('Mean skin-mask area:',t.groupby('dataset').skin_area.first().round(3).to_dict())
```
