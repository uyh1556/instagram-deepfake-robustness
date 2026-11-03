# Build_NSF_Parser_Masks (한 번만, 백본 공용)

parser ablation에 쓸 눈·입 마스크를 parser 3종으로 미리 만들어 Drive tar로 저장. 세 백본 노트북이 이 tar를 같이 씀.
- 대상: FF++ 점수기 표본 3,200장(원본) + CDF test 8,288장 × 원본·강 보정 4종(마스크는 각 보정 이미지에서 다시 파싱)
- `segformer`(현재 parser) → `bisenet` → `mediapipe` 순서로 만듦. erode/dilate는 segformer 마스크에서 계산하므로 따로 안 만듦
- 결과: `manifests/nsf_parser_ablation_v1/cache/parser_masks_<parser>.tar`. 이미 있는 tar는 건너뜀(끊기면 Cell 1부터 다시)
- GPU 필요(T4로 충분). 대략 parser당 15~30분, mediapipe는 CPU라 코어가 많을수록 빠름

## Cell 1 — 코드·parser 설치

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
# 비교 parser 패키지: --no-deps로 wheel만 설치(numpy·opencv 버전을 건드리지 않게). 가중치는 스크립트가 고정 URL·해시로 받음
subprocess.run([sys.executable,'-m','pip','install','-q','filterpy','absl-py~=2.3','flatbuffers~=25.9'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','--only-binary',':all:','facexlib==0.3.0','mediapipe==1.1.1'],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(), "GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
print('CPU cores:',os.cpu_count(),'| local free GB:',shutil.disk_usage('/content').free//2**30)

```

## Cell 2 — 설정

```python
ROOT = Path('/content/drive/MyDrive/instagram-deepfake-robustness')
# 프레임 목록·FF++ 표본 해시 확인에만 xception 폴더를 씀(마스크는 백본과 무관)
RUNS = ROOT/'runs/ffpp_c23_smoothing_v2_diagonal_xception'
REFERENCE = ROOT/'evaluations/ffpp_c23_smoothing_v2_diagonal_xception'
ARCHIVE = ROOT/'manifests/beauty_smoothing_v2/cache'
MASK_DIR = ROOT/'manifests/nsf_parser_ablation_v1/cache'
FF_ROOT = Path('/content/nsf_parser_ffpp')
CDF_ROOT = Path('/content/celebdf_v2_face_crops16_v1')
WORKERS = max(2, min(8, os.cpu_count()))
COMMON = f'--model xception --runs-root "{RUNS}" --evaluation-root "{REFERENCE}" --source-root "{FF_ROOT}" --cdf-root "{CDF_ROOT}"'
for part in ['mask','bilateral','weighted_median','fast_global','domain_transform']:
    assert (ARCHIVE/f'cdf_test_{part}.tar').is_file(), f'캐시 TAR 없음: cdf_test_{part}.tar'
for f in ['segformer','bisenet','mediapipe']:
    print(f, 'done:', (MASK_DIR/f'parser_masks_{f}.tar').is_file(), flush=True)

```

## Cell 3 — 데이터 준비 (FF++ 표본 3,200장 + CDF test)

```python
!python -u "{SCRIPT}" prepare {COMMON} --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1"
assert _exit_code==0, 'FF++ 표본 준비 실패'
name, size = 'celebdf_v2_roi16_test_v1.tar', 557015040
if not (CDF_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv').is_file():
    source = ROOT/'data/celebdf_v2_face_crops16_v1'/name
    assert source.is_file() and source.stat().st_size==size, f'TAR 확인: {source}'
    local = Path('/content')/name; print('Copying:',name,flush=True); shutil.copy2(source,local)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
print('CDF test ready', flush=True)

```

## Cell 4 — 마스크 생성 (segformer → bisenet → mediapipe)

```python
for FAMILY in ['segformer', 'bisenet', 'mediapipe']:
    !python -u "{SCRIPT}" masks --family {FAMILY} {COMMON} --archive-dir "{ARCHIVE}" --mask-dir "{MASK_DIR}" --workers {WORKERS}
    assert _exit_code==0, f'{FAMILY} 마스크 생성 실패'

```

## Cell 5 — 마스크 미리보기 (CDF 원본 6장: 원본 / SegFormer / BiSeNet / MediaPipe)

```python
import numpy as np, pandas as pd
from PIL import Image
from IPython.display import display
WORK = Path('/content/nsf_parser_masks')
area = pd.read_csv(WORK/'segformer/mask_area.csv', dtype={'crop_path': str})
paths = area[area.view.eq('normal') & area.roi_status.eq('detected')].crop_path.sample(6, random_state=0).tolist()
rows = []
for p in paths:
    with Image.open(CDF_ROOT/p) as im:
        img = np.asarray(im.convert('RGB').resize((192, 192)))
    tiles = [img]
    for f in ['segformer', 'bisenet', 'mediapipe']:
        with Image.open(WORK/f/'normal'/(p+'.png')) as im:
            m = np.asarray(im.resize((192, 192))) > 127
        t = img.copy(); t[m] = (0.5*t[m] + [127, 0, 0]).astype(np.uint8); tiles.append(t)
    rows.append(np.concatenate(tiles, 1))
display(Image.fromarray(np.concatenate(rows, 0)))
for f in ['segformer', 'bisenet', 'mediapipe']:
    a = pd.read_csv(WORK/f/'mask_area.csv', dtype={'crop_path': str})
    print(f, a.groupby('view', sort=False).area.agg(mean_area='mean', empty=lambda s: int((s == 0).sum())).to_dict('index'))

```
