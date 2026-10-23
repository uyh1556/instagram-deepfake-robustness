# Beauty_Smoothing_V2_Calibration (GPU 권장, 학습 없음)

새 스무딩 프로토콜 v2의 강도를 **FF++ train**으로 확정한다. CDF는 쓰지 않는다. 이후 캐시·평가·학습은 모두 여기서 나온 `calibration.json`을 쓴다.

- 마스크: 얼굴 파싱 피부 마스크 (SegFormer `jonathandinu/face-parsing`, 고정 revision, 피부+코, 경계 1% 침식·1.2% 페더링). 눈·눈썹·입술·머리카락·배경은 스무딩하지 않는다.
- 필터 4개, 경계 보존 설정(색 시그마 고정): bilateral 35 · weighted median 35 · fast global 8 · domain transform 40. 강도는 공간 범위로만 정한다.
- 기준(앵커): 눈으로 확인한 bilateral(공간 0.08225)의 피부 변화량 중앙값. 나머지 세 필터는 같은 변화량이 나오도록 공간 범위를 다시 맞춘다.
- 약·중·강 = 마스크 안 25/50/75% 섞기.
- 결과: Drive `manifests/beauty_smoothing_v2/` — `calibration.json`, `calibration_frames.csv`, `preview_strong.png`. 캐시 tar용 `cache/` 폴더도 여기서 미리 만든다(동시 실행 시 같은 이름 폴더가 두 개 생기는 것 방지).
- 런타임: GPU면 파싱이 1분 안쪽. CPU도 되지만 300장 파싱에 10분 이상.

## Cell 1 — 코드·FF++ train/val 준비·OpenCV contrib

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
assert (PROJECT/'scripts/calibrate_beauty_smoothing_v2.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','opencv-contrib-python-headless==4.11.0.86','transformers','safetensors'],check=True)
subprocess.run([sys.executable,'-c','import cv2,transformers; assert hasattr(cv2,"ximgproc"),"contrib 미설치"; print(cv2.__version__, transformers.__version__)'],check=True)

ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
OUTPUT=ROOT/'manifests/beauty_smoothing_v2'
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'
```

## Cell 2 — 강도 확정

```python
assert not (OUTPUT/'calibration.json').is_file(),'이미 확정됨: 바꾸려면 새 프로토콜 이름이 필요합니다 (기존 캐시와 섞이면 안 됨).'
!python -u "{PROJECT}/scripts/calibrate_beauty_smoothing_v2.py" --source-root "{SOURCE_ROOT}" --output-root "{OUTPUT}" --images 300 --seed 42
assert _exit_code==0,'보정 실패: 위 오류 확인'
(OUTPUT/'cache').mkdir(exist_ok=True)
import json
from IPython.display import Image as Show, display
c=json.loads((OUTPUT/'calibration.json').read_text())
print('목표 피부 변화량(앵커 bilateral):',round(c['target_value'],3))
for n,r in c['filters'].items(): print(f"{n:17s} 공간 {r['value']:.5f}  변화량 중앙값 {r['median']:.3f}  (눈으로 정한 값 {c['approved'][n]})")
display(Show(str(OUTPUT/'preview_strong.png')))
print('열 순서: 원본 | bilateral | weighted median | fast global | domain transform (모두 강 0.75, 파싱 마스크)')
```
