# ROI Skin Smoothing DA Pilot — Seed42

Instagram 공식 알고리즘이 아닌 **피부 영역 근사 스무딩** 실험.
기존 frozen ROI 1.5배 내부의 타원형 마스크에서 눈·입 주변을 제외하고,
bilateralFilter 결과와 원본을 25/50/75% 혼합한다. 정확한 피부 segmentation이나
랜드마크 기반 보정은 아니므로 자세/얼굴 위치에 따라 마스크가 어긋날 수 있다.
얼굴 검출 실패 ROI는 보정하지 않는다. 피부 보정과 일반 blur를 동일시하지 않는다.

기존 source-aware split과 ROI checkpoint 유지. Train 3,000 / Val 900 / Test 1,500
(각 original/Deepfakes/Face2Face별 1,000/300/500). Val은 원본만 사용.
기존 모델 + 원본 추가학습 + supervised smoothing augmentation + DANN 비교.
추가학습은 기존 ROI best에서 각각 3 epochs, LR1e-5. DANN target은 train 이미지의
보정 view이며 해당 view의 class label은 사용하지 않는다. 3강도 모두 학습에 포함된다.
새 조작 기법/새 스무딩 구현 일반화를 증명하는 실험은 아니다.
기존 색감 파일럿과 별도 cache/runs/evaluations 사용. 데이터 TAR나 manifest 재업로드 불필요.

## Cell 1 — Drive·코드·설치

실행 전에 새 코드를 GitHub에 push해야 한다. ZIP 불필요. 무거운 라이브러리는
별도 Python 프로세스에서 실행하므로 노트북 kernel 재시작을 요구하지 않는다.

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, json, shutil
PROJECT = Path('/content/instagram_deepfake_robustness')
REPO = 'https://github.com/uyh1556/instagram-deepfake-robustness.git'
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone',REPO,str(PROJECT)],check=True)
assert (PROJECT/'scripts/smoothing_roi_pilot.py').is_file(), '새 코드 먼저 push하세요'
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(PROJECT/'requirements-smoothing.txt')],check=True)
subprocess.run([sys.executable,'-c',"import torch,cv2; from PIL import Image; assert torch.cuda.is_available(), 'GPU 런타임으로 변경'; print(torch.cuda.get_device_name(0), 'OpenCV:',cv2.__version__)"],check=True)
```

## Cell 2 — 고정 경로

```python
ROOT = Path('/content/drive/MyDrive/instagram-deepfake-robustness')
DRIVE_DATA = ROOT/'data/ffpp_c0_v1'
DATA_ROOT = Path('/content/ffpp_c0_v1')
MANIFEST = PROJECT/'manifests/ffpp_c0_face_roi_v1/manifest.csv'
CHECKPOINT = Path('/content/roi_c0_initial_best.pt')
TRAIN_PER_METHOD, VAL_PER_METHOD, TEST_PER_METHOD = 1000, 300, 500
TAG = f't{TRAIN_PER_METHOD}_v{VAL_PER_METHOD}_e{TEST_PER_METHOD}_seed42'
CACHE_ROOT = Path('/content/smoothing_roi_pilot_v1')/TAG
RUNS_ROOT = ROOT/'runs/smoothing_roi_pilot_v1'/TAG
EVALUATION_ROOT = ROOT/'evaluations/smoothing_roi_pilot_v1'/TAG
COMMON = f'--data-root "{DATA_ROOT}" --manifest "{MANIFEST}" --checkpoint "{CHECKPOINT}" --cache-root "{CACHE_ROOT}" --seed 42'
print('Results:', EVALUATION_ROOT)
```

## Cell 3 — 기존 데이터·ROI 모델

```python
assets = json.loads((PROJECT/'configs/assets.json').read_text())
temp_root = Path('/content/_c0_archives')
temp_root.mkdir(exist_ok=True)
for info in assets['dataset']['archives']:
    archive = DRIVE_DATA/info['name']
    assert archive.is_file(), archive
    assert archive.stat().st_size == info['size_bytes'], f'업로드 미완료: {archive}'
    marker = Path('/content')/f".{info['name']}.extracted"
    if marker.exists():
        print('Already extracted:',info['name'],flush=True)
        continue
    local = temp_root/info['name']
    if not local.exists() or local.stat().st_size != info['size_bytes']:
        print('Copying:',info['name'],flush=True)
        shutil.copy2(archive,local)
    print('Extracting:',info['name'],flush=True)
    result = subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        print(result.stderr,flush=True)
        raise RuntimeError(f"압축 해제 실패: {info['name']}")
    marker.touch()
    local.unlink()  # local temporary copy only; Drive originals preserved
original_best = ROOT/'checkpoints/ffpp_c0_xception_face_roi_seed42/best.pt'
assert original_best.is_file(), original_best
if not CHECKPOINT.exists():
    shutil.copy2(original_best,CHECKPOINT)
print('Source data and checkpoint ready',flush=True)
```

## Cell 4 — 보정 cache

RGB → frozen native ROI → 마스크 스무딩 → 기존 bicubic299 → PNG cache.
원본은 기존 ROI 전처리 그대로. JPEG·noise 추가 없음. 약 30초마다 진행 출력.
재실행하면 저장된 PNG를 재사용한다. 실제 적용 조건/코드 hash/OpenCV 버전 저장.

```python
!python -u "{PROJECT}/scripts/smoothing_roi_pilot.py" prepare {COMMON} \
  --train-per-method {TRAIN_PER_METHOD} --val-per-method {VAL_PER_METHOD} \
  --test-per-method {TEST_PER_METHOD} --workers 2
assert _exit_code == 0, '준비 실패: 위 오류 확인'
PREVIEW = EVALUATION_ROOT/'train_smoothing_preview.png'
subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/smoothing_roi_pilot.py'),
    'preview','--cache-root',str(CACHE_ROOT),'--output',str(PREVIEW)],check=True)
from IPython.display import Image as DisplayImage, display
display(DisplayImage(filename=str(PREVIEW)))
```

## Cell 5 — Baseline 원본·약·중·강 평가

같은 test 1,500장 × 4조건. AUC뿐 아니라 0.5 threshold의 FPR/FNR도 확인한다.
결과를 보고 강도를 바꾸면 이후 결과는 개발용이다. PNG가 눈에 띄게 뭉개지면
현실적인 보정으로 주장하지 않는다. 성능 저하가 없더라도 과도하게 강도를 높여
유리한 조건을 찾아내지 않는다.

```python
!python -u "{PROJECT}/scripts/smoothing_roi_pilot.py" evaluate {COMMON} \
  --strategies baseline --output-root "{EVALUATION_ROOT}" --batch-size 32 --workers 2
assert _exit_code == 0, 'Baseline 평가 실패'
```

## Cell 6 — 원본 추가학습·augmentation·DANN

source_ft는 원본 추가학습 대조군. augmentation은 원본/스무딩 두 view의 분류 loss 평균.
DANN은 원본 분류 loss + 0.1×domain loss, GRL 점진 증가.
동일 train source와 epoch 수/2-view forward를 맞추지만 target class 정답 사용 여부는 다르다.
3강도 중 하나를 매 이미지·epoch마다 무작위 선택하며 real/fake와 무관하게 적용한다.
원본 val AUC 최고 checkpoint 저장. 완료 skip, 미완료 last.pt에서 epoch 단위 재개.

```python
!python -u "{PROJECT}/scripts/smoothing_roi_pilot.py" train {COMMON} \
  --strategies source_ft augmentation dann --runs-root "{RUNS_ROOT}" \
  --epochs 3 --batch-size 16 --workers 2 --learning-rate 1e-5 \
  --weight-decay 1e-4 --domain-weight 0.1
assert _exit_code == 0, '학습 실패: 위 오류 확인'
```

## Cell 7 — 전후 동일 test 평가

4모델 × 원본/약/중/강 = 16조건. 완료 baseline 평가는 재사용한다.

```python
!python -u "{PROJECT}/scripts/smoothing_roi_pilot.py" evaluate {COMMON} \
  --strategies baseline source_ft augmentation dann --runs-root "{RUNS_ROOT}" \
  --output-root "{EVALUATION_ROOT}" --batch-size 32 --workers 2
assert _exit_code == 0, '평가 실패: 위 오류 확인'
```

## Cell 8 — 결과

```python
import pandas as pd
table = pd.read_csv(EVALUATION_ROOT/'condition_summary.csv')
display(table[['strategy','condition','roc_auc','accuracy','real_fpr','fake_fnr','Deepfakes_auc','Face2Face_auc']].round(6))
display(pd.read_csv(EVALUATION_ROOT/'model_summary.csv').round(6))
print('분석 파일: condition_summary.csv, model_summary.csv, baseline_deltas.csv')
print(EVALUATION_ROOT)
```

`filtered_*`는 기존 공통 집계 열 이름이며 이번에는 **스무딩 3강도 평균**이다.
WildDeepfake·실제 Instagram·unseen 보정 구현은 포함하지 않는다.
