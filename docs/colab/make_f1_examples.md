# F1_Examples (발표 F1: 진짜 오탐 / 가짜 미탐 예시, GPU 불필요)

이미 저장된 seed42 Baseline 예측(`evaluations/ffpp_c23_smoothing_v2_diagonal_<모델>/seed42/cdf_test/*_baseline_*.csv`)과 v2 cdf_test 캐시만 쓴다. 모델은 돌리지 않는다.
- 진짜 후보: 세 백본 모두 원본에서 가짜 확률 < 0.5, bilateral 강 보정본에서 > 0.5 (오탐)
- 가짜 후보: Xception·DINO가 원본 > 0.5, DT 강 보정본 < 0.5 (미탐)
- 영상당 한 장, 뒤집힘이 뚜렷한 순서. 후보마다 미리보기 PNG를 저장하고, Cell 4에서 고른 번호로 최종 F1을 만든다.
- 결과: Drive `evaluations/proposal_f1/`. **예시이므로 슬라이드에 전체 뒤집힘 비율도 함께 보여줄 것.**

## Cell 1 — 코드·폰트

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, shutil, sys
PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
subprocess.run(['apt-get','-qq','-y','install','fonts-nanum'],check=True)   # 그림 속 한글
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
ARCHIVE=ROOT/'manifests/beauty_smoothing_v2/cache'
CDF_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
OUT=ROOT/'evaluations/proposal_f1'
```

## Cell 2 — CDF test 이미지 준비

```python
if not (CDF_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv').is_file():
    src=ROOT/'data/celebdf_v2_face_crops16_v1/celebdf_v2_roi16_test_v1.tar'
    assert src.is_file() and src.stat().st_size==557015040,f'TAR 확인: {src}'
    shutil.copy2(src,'/content/cdf_test.tar'); subprocess.run(['tar','-xf','/content/cdf_test.tar','-C','/content'],check=True)
print('ok')
```

## Cell 3 — 후보 고르기 + 미리보기

```python
!cd "{PROJECT}/scripts" && python -u make_f1_examples.py --eval-root "{ROOT}/evaluations" --cdf-root "{CDF_ROOT}" --archive-dir "{ARCHIVE}" --output-root "{OUT}"
assert _exit_code==0
import pandas as pd
from IPython.display import Image as Show, display
for case in ['real','fake']:
    c=pd.read_csv(OUT/f'F1_candidates_{case}.csv'); print(case,len(c)); display(c.filter(like='p_').round(2).assign(video=c.video_id))
    for i in range(len(c)): display(Show(str(OUT/f'F1_candidate_{case}_{i}.png'),width=330))
```

## Cell 4 — 고른 번호로 최종 F1

```python
REAL_PICK=0   # Cell 3의 진짜 후보 번호
FAKE_PICK=0   # Cell 3의 가짜 후보 번호
!cd "{PROJECT}/scripts" && python -u make_f1_examples.py --eval-root "{ROOT}/evaluations" --cdf-root "{CDF_ROOT}" --archive-dir "{ARCHIVE}" --output-root "{OUT}" --real-pick {REAL_PICK} --fake-pick {FAKE_PICK}
display(Show(str(OUT/'F1_examples.png')))
```
