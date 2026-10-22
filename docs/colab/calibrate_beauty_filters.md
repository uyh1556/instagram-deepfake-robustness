# Beauty_Filter_Calibration (CPU, 학습 없음)

학습에 쓰지 않은 피부 스무딩 필터 3개의 세기를 학습용 bilateral에 맞춘다. 피부 스무딩 필터 세 계열에서 하나씩 골랐다
(StyleRetoucher, arXiv 2312.14389, 2.1절 분류).

|필터|계열|OpenCV contrib|맞추는 파라미터|
|---|---|---|---|
|`weighted_median`|국소 엣지 보존 (Zhang et al., CVPR 2014)|`ximgproc.weightedMedianFilter`|반경 (ROI 한 변 대비 비율)|
|`fast_global`|전역 에너지 최적화 (Min et al., TIP 2014)|`ximgproc.fastGlobalSmootherFilter`|평활 범위 (ROI 한 변 대비 비율)|
|`domain_transform`|구조 보존·guided 계열 (Gastal & Oliveira, SIGGRAPH 2011)|`ximgproc.dtFilter`|색 시그마 (공간 시그마는 bilateral과 같은 0.02)|

- 기준: 피부 마스크 안 **고주파 에너지 감소율**(1 − 스무딩 후 고주파 / 원본 고주파)의 중앙값이 bilateral 완전 스무딩과 같아지도록.
- 데이터: **FF++ train** 프레임 300장(진짜·가짜 반반, 얼굴 검출 성공분). CDF는 쓰지 않는다.
- 약·중·강은 bilateral과 똑같이 같은 피부 마스크 안에서 0.25/0.5/0.75 섞기. 이후 평가는 우선 **강(0.75)만** 쓴다.
- 결과: Drive `manifests/beauty_filter_calibration_v1/` — `calibration.json`(필터별 값·중앙값·사분위), `preview_strong.png`(원본·bilateral·세 필터 강 스무딩 비교 4장), `calibration_frames.csv`.
- 런타임: **CPU**면 충분하다. weighted median이 가장 느리다(수십 분 예상, 실측 아님).

## Cell 1 — 코드·FF++ 준비·OpenCV contrib 설치

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
assert (PROJECT/'scripts/calibrate_beauty_filters.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
# Same OpenCV version as the pinned headless build, plus contrib (ximgproc); bilateral stays identical.
subprocess.run([sys.executable,'-m','pip','install','-q','opencv-contrib-python-headless==4.11.0.86'],check=True)
subprocess.run([sys.executable,'-c','import cv2; assert hasattr(cv2,"ximgproc"),"contrib 미설치"; print(cv2.__version__)'],check=True)

ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
OUTPUT=ROOT/'manifests/beauty_filter_calibration_v1'
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'
```

## Cell 2 — 강도 맞추기

```python
!python -u "{PROJECT}/scripts/calibrate_beauty_filters.py" --source-root "{SOURCE_ROOT}" --output-root "{OUTPUT}" --images 300 --seed 42
assert _exit_code==0,'보정 실패: 위 오류 확인'
from IPython.display import Image as Show, display
display(Show(str(OUTPUT/'preview_strong.png')))
print('열 순서: 원본 | bilateral | weighted median | fast global | domain transform (모두 강 0.75)')
```
