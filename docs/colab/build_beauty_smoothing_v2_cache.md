# Beauty_Smoothing_V2_Cache (GPU 런타임, 한 번만)

v2 스무딩을 미리 계산해 Drive tar로 남긴다. 이후 평가·학습은 이 tar만 풀어 쓰고 다시 계산하지 않는다.

- 먼저 `calibrate_beauty_smoothing_v2` 노트북으로 `manifests/beauty_smoothing_v2/calibration.json`을 만들어야 한다.
- 저장 단위: (데이터셋, 부분)마다 tar 하나. 부분 = `mask`(피부 마스크, GPU) + 필터 4개(섞기 전 100% 결과, CPU 병렬). 위치: Drive `manifests/beauty_smoothing_v2/cache/<데이터셋>_<부분>.tar`.
  약·중·강(25/50/75%)은 쓸 때 섞으므로 저장하지 않는다. 얼굴 검출 실패(fallback) 프레임은 원래대로 두므로 저장하지 않는다.
- 데이터셋:
  - `cdf_test` — CDF 공식 test 8,288장 (Baseline 평가에 필요, 필터 4개 합쳐 약 2GB)
  - `ffpp_test` — FF++ test (같은 도메인 평가: CDF 도메인 차이 없이 스무딩 영향만)
  - `cdf_dev` — CDF pool v2의 라벨 있는 **비테스트** 프레임 (새 학습법 개발·비교용. 공식 test 영상과 겹치면 멈춤). 공식 test는 방법을 확정한 뒤 한 번만 본다.
  - `ffpp_trainval` — FF++ train+val (대각선 학습용). 필터 하나에 약 9GB, 4개면 30~40GB. bilateral·weighted median이 느려 몇 시간 걸린다.
- 끊겨도 다시 실행하면 이미 만든 프레임·부분은 건너뛰고, Drive에 tar가 있으면 계산 대신 푼다.
- 추천 순서: 이 노트북 하나에서 `DATASETS=['cdf_test','ffpp_test','cdf_dev']` → Baseline 평가 시작 (`cdf_dev`는 비테스트 가짜 TAR 5.7GB 복사가 추가된다). 동시에 다른 GPU 런타임에서 `DATASETS=['ffpp_trainval']`.
  CPU 코어가 많은 런타임(예: A100/L4 고RAM)일수록 필터 계산이 빠르다.
  `ffpp_trainval`을 나눠 돌리고 싶으면 노트북마다 `PARTS`를 다르게 준다(부분마다 tar가 따로라 겹치지 않는다).

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
assert (PROJECT/'scripts/build_beauty_smoothing_v2_cache.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','opencv-contrib-python-headless==4.11.0.86','transformers','safetensors'],check=True)
subprocess.run([sys.executable,'-c','import cv2,torch,transformers; assert hasattr(cv2,"ximgproc"),"contrib 미설치"; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(cv2.__version__, transformers.__version__, torch.cuda.get_device_name(0))'],check=True)
print('CPU cores:',os.cpu_count(),'| local free GB:',shutil.disk_usage('/content').free//2**30)
```

## Cell 2 — 데이터셋·경로

```python
DATASETS=['cdf_test','ffpp_test','cdf_dev']   # 또는 ['ffpp_trainval']
PARTS=['mask','bilateral','weighted_median','fast_global','domain_transform']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
V2=ROOT/'manifests/beauty_smoothing_v2'
CALIBRATION=V2/'calibration.json'
ARCHIVE=V2/'cache'
assert CALIBRATION.is_file(),'calibrate_beauty_smoothing_v2 노트북을 먼저 실행하세요.'
assert ARCHIVE.is_dir(),'calibrate 노트북 Cell 2가 cache 폴더를 만듭니다.'
CDF_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
FF_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
DATA_ROOT={'cdf_test':CDF_ROOT,'ffpp_test':FF_ROOT,'ffpp_trainval':FF_ROOT,'cdf_dev':CDF_ROOT}
MEMBERSHIP=next((ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv' for m in ['clip_b16','dinov2_b14','xception']
                 if (ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv').is_file()),None)

def extract_cdf(name,size):
    if (CDF_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv').is_file(): return
    source=ROOT/'data/celebdf_v2_face_crops16_v1'/name
    assert source.is_file() and source.stat().st_size==size,f'TAR 확인: {source}'
    assert shutil.disk_usage('/content').free>2*size+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name; print('Copying:',name,flush=True); shutil.copy2(source,local)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()

def prepare(dataset):
    if dataset in ('cdf_test','cdf_dev'):
        extract_cdf('celebdf_v2_roi16_test_v1.tar',557015040)   # cdf_dev also needs it for the overlap check
        if dataset=='cdf_dev':
            assert MEMBERSHIP is not None,'pool v2 membership_labelled.csv를 찾지 못했습니다'
            extract_cdf('celebdf_v2_roi16_non_test_real_v1.tar',826142720)
            extract_cdf('celebdf_v2_roi16_non_test_fake_v1.tar',5679564800)
    else:
        flag='--test-only' if dataset=='ffpp_test' else '--trainval-only'
        subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/prepare_ffpp_c23_training_data.py'),'--drive-data',str(ROOT/'data/ffpp_c23_4methods_roi16_v1'),
                        '--data-root',str(FF_ROOT),flag],check=True)
print('Datasets:',DATASETS,'| parts:',PARTS,'| tars ->',ARCHIVE)
```

## Cell 3 — 캐시 만들기 (끊기면 다시 실행: 이어서 계산)

```python
for dataset in DATASETS:
    prepare(dataset)
    extra=f'--membership "{MEMBERSHIP}"' if dataset=='cdf_dev' else ''
    !python -u "{PROJECT}/scripts/build_beauty_smoothing_v2_cache.py" --dataset {dataset} --data-root "{DATA_ROOT[dataset]}" --calibration "{CALIBRATION}" --archive-dir "{ARCHIVE}" --parts {' '.join(PARTS)} --workers {os.cpu_count()} {extra}
    assert _exit_code==0,f'{dataset} 캐시 실패: 위 오류 확인 후 다시 실행하면 이어서 계산'
!ls -la "{ARCHIVE}"
```
