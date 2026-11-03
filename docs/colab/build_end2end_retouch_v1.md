# Build_End2End_Retouch_GFPGAN (한 번만, 백본 공용)

CDF 공식 test 8,288장 중 얼굴 검출 프레임 전체를 **GFPGAN v1.4**(end-to-end, 마스크 없음)로 보정해 Drive tar 하나로 저장.
- 설정: arch clean, upscale 1(크롭 크기 그대로), 가운데 얼굴, 크롭에 다시 붙임. 프레임마다 고정 seed(GFPGAN 노이즈 주입 재현용)
- fallback 프레임은 v2와 같이 보정 안 함. GFPGAN이 얼굴을 못 찾은 프레임은 원본 그대로(개수 출력)
- 결과: `manifests/end2end_retouch_v1/cache/cdf_test_gfpgan.tar` + `preview_gfpgan.jpg`(위: 원본, 아래: 보정)
- 이 tar가 기준이고 다시 만들지 않음. 끊기면 Cell 1부터 다시 실행하면 이어서 함
- Cell 4로 몇 장 먼저 확인한 뒤 Cell 5(전체)를 실행
- 끝나면 백본별 평가 노트북 3개(`end2end_retouch_v1_<model>`)를 동시에 실행

## Cell 1 — 코드·GFPGAN 설치

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
SCRIPT = PROJECT/'scripts/end2end_retouch_v1.py'
assert SCRIPT.is_file(), '새 코드를 GitHub에 push한 후 Cell 1을 다시 실행하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
# GFPGAN 1.3.8 (v1.4 가중치) + facexlib: wheel만 설치(--no-deps: opencv-python이 opencv-contrib(cv2.ximgproc)을 덮지 않게).
# basicsr 1.4.2는 소스 배포만 있고 Python 3.13에서 설치가 깨져서, 스크립트가 PyPI 소스(해시 확인)에서 패키지 폴더만 꺼내 씀
subprocess.run([sys.executable,'-m','pip','install','-q','addict','future','lmdb','yapf','filterpy'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','--only-binary',':all:','facexlib==0.3.0','gfpgan==1.3.8'],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(), "GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
print('CPU cores:',os.cpu_count(),'| local free GB:',shutil.disk_usage('/content').free//2**30)

```

## Cell 2 — 설정

```python
ROOT = Path('/content/drive/MyDrive/instagram-deepfake-robustness')
ARCHIVE = ROOT/'manifests/end2end_retouch_v1/cache'
CDF_ROOT = Path('/content/celebdf_v2_face_crops16_v1')
print('Archive:', ARCHIVE, '| done:', (ARCHIVE/'cdf_test_gfpgan.tar').is_file(), flush=True)

```

## Cell 3 — CDF test 준비

```python
name, size = 'celebdf_v2_roi16_test_v1.tar', 557015040
if not (CDF_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv').is_file():
    source = ROOT/'data/celebdf_v2_face_crops16_v1'/name
    assert source.is_file() and source.stat().st_size==size, f'TAR 확인: {source}'
    local = Path('/content')/name; print('Copying:',name,flush=True); shutil.copy2(source,local)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
print('CDF test ready', flush=True)

```

## Cell 4 — 샘플 미리보기 (전체 실행 전, 진짜 3장 + 가짜 3장, 1~2분). 열: 원본 | GFPGAN | 차이×4

```python
SAMPLE_SEED = 0   # 다른 샘플을 보려면 숫자만 바꿔 다시 실행
!python -u "{SCRIPT}" sample --cdf-root "{CDF_ROOT}" --count 6 --sample-seed {SAMPLE_SEED} --sample-out /content/gfpgan_sample.jpg
assert _exit_code==0, '샘플 실패: GFPGAN 설치/GPU 확인'
from IPython.display import Image as Show, display
display(Show(filename='/content/gfpgan_sample.jpg'))

```

## Cell 5 — GFPGAN 보정 + tar 저장 (T4 기준 대략 15~30분)

```python
!python -u "{SCRIPT}" build --archive-dir "{ARCHIVE}" --cdf-root "{CDF_ROOT}"
assert _exit_code==0, 'GFPGAN 보정 실패'

```

## Cell 6 — 전체 결과 미리보기 (위: 원본, 아래: GFPGAN)

```python
from IPython.display import Image as Show, display
display(Show(filename=str(ARCHIVE/'preview_gfpgan.jpg')))

```
