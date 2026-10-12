# ROI 필터·DA 파일럿

고정 필터: Nashville, Lark, Clarendon, Ludwig, Crema (후보 21개에서 seed42로 결과와 무관하게 추출).

기존 ROI best → 원본/5개 필터 평가 → source-only 추가학습·필터 증강·DANN → 동일 테스트 재평가.

기본 subset: train 3,000 / val 900 / test 1,500. 기존 split 안에서만 추출하며 각 기법에 같은 수를 사용한다. 학습은 모두 기존 best.pt에서 시작해 3 epochs, LR 1e-5. Best 선택은 원본 validation AUC만 사용한다.

5개 필터 모두 adaptation에 포함되므로 이 파일럿은 unseen-filter 일반화를 검증하지 않는다. DANN target은 train 이미지의 필터 버전으로, 정답을 숨긴 모의 적응 설정이다. 증강 모델은 그 정답을 사용한다. 비교는 동일 epoch·source 이미지·2-view 계산을 맞추되 사용되는 정답 정보는 다름을 구분한다.

필터는 full-frame RGB에 적용 → 기존 고정 ROI crop → 기존 bicubic299 → Normalize(0.5). 필터 적용 ROI를 lossless PNG로 미리 cache해 반복 학습을 빠르게 한다. 추가 JPEG·noise·blur는 없다.

TAR 3개의 Drive 업로드 완료 후 새 GPU 런타임에서 시작한다. 모든 무거운 라이브러리는 별도 Python 프로세스에서 실행하므로 노트북 kernel의 기존 Pillow 로딩 여부를 검사하거나 강제로 재시작하지 않는다.

## Cell 1 — Drive·저장소·의존성

코드 ZIP 업로드는 필요 없음. Python 패키지는 새 저장소에서 설치한다.

```python
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import subprocess, sys, json, shutil

PROJECT = Path("/content/instagram_deepfake_robustness")
REPO = "https://github.com/uyh1556/instagram-deepfake-robustness.git"
if PROJECT.exists():
    remote = subprocess.check_output(["git","-C",str(PROJECT),"remote","get-url","origin"],text=True).strip()
    assert remote.rstrip("/") == REPO, f"Different repository at {PROJECT}: {remote}"
    subprocess.run(["git","-C",str(PROJECT),"pull","--ff-only"],check=True)
else:
    subprocess.run(["git","clone",REPO,str(PROJECT)],check=True)
subprocess.run([sys.executable,"-m","pip","install","-q","-r",str(PROJECT/"requirements-pilot.txt")],check=True)
subprocess.run([sys.executable,"-c",
    "import torch,pilgram2; from PIL import Image; "
    "assert torch.cuda.is_available(), 'GPU 런타임으로 변경하세요'; "
    "print('GPU:',torch.cuda.get_device_name(0)); print('Pillow:',Image.__version__)"],check=True)
```

## Cell 2 — 경로·파일럿 규모

전체 데이터로 실행하려면 세 PER_METHOD를 모두 0으로 변경한다. CACHE와 결과 이름이 자동으로 달라져 파일럿 결과와 섞이지 않는다. 기본값은 빠른 파일럿용이다.

```python
ROOT = Path("/content/drive/MyDrive/instagram-deepfake-robustness")
DRIVE_DATA = ROOT/"data/ffpp_c0_v1"
DATA_ROOT = Path("/content/ffpp_c0_v1")
MANIFEST = PROJECT/"manifests/ffpp_c0_face_roi_v1/manifest.csv"
CHECKPOINT = Path("/content/roi_c0_initial_best.pt")

TRAIN_PER_METHOD = 1000
VAL_PER_METHOD = 300
TEST_PER_METHOD = 500
TAG = f"t{TRAIN_PER_METHOD}_v{VAL_PER_METHOD}_e{TEST_PER_METHOD}_seed42"
CACHE_ROOT = Path("/content/pilgram2_roi_pilot_v1")/TAG
RUNS_ROOT = ROOT/"runs/pilgram2_roi_pilot_v1"/TAG
EVALUATION_ROOT = ROOT/"evaluations/pilgram2_roi_pilot_v1"/TAG
COMMON = f'--data-root "{DATA_ROOT}" --manifest "{MANIFEST}" --checkpoint "{CHECKPOINT}" --cache-root "{CACHE_ROOT}" --seed 42'
print("Results:",EVALUATION_ROOT)
```

## Cell 3 — c0 TAR 압축 해제·기존 ROI 체크포인트

Drive의 TAR를 하나씩 로컬에 복사한 뒤 해제한다. 완료 marker가 있는 archive는 건너뛴다. 실패 시 TAR stderr를 보여주며 marker를 만들지 않는다.

```python
assets = json.loads((PROJECT/"configs/assets.json").read_text())
temp_root = Path("/content/_c0_archives")
temp_root.mkdir(exist_ok=True)
for info in assets["dataset"]["archives"]:
    archive = DRIVE_DATA/info["name"]
    assert archive.is_file(), f"TAR 업로드를 완료하세요: {archive}"
    assert archive.stat().st_size == info["size_bytes"], f"업로드 미완료 또는 파일 크기 불일치: {archive}"
    marker = Path("/content")/f".{info['name']}.extracted"
    if marker.exists():
        print("Already extracted:",info["name"],flush=True)
        continue
    local = temp_root/info["name"]
    if not local.exists():
        print("Copying:",info["name"],flush=True)
        shutil.copy2(archive,local)
    assert local.stat().st_size == info["size_bytes"], local
    print("Extracting:",info["name"],flush=True)
    result = subprocess.run(["tar","-xf",str(local),"-C","/content"],capture_output=True,text=True)
    if result.returncode:
        print(result.stderr,flush=True)
        raise RuntimeError(f"압축 해제 실패: {info['name']}")
    marker.touch()
    local.unlink()  # local temporary copy only; Drive TAR is preserved
original_best = ROOT/"checkpoints/ffpp_c0_xception_face_roi_seed42/best.pt"
assert original_best.is_file(),original_best
if not CHECKPOINT.exists():
    shutil.copy2(original_best,CHECKPOINT)
print("Source data and ROI checkpoint ready",flush=True)
```

## Cell 4 — 고정 subset·필터 ROI cache 준비

처음 한 번은 필터 적용과 cache 저장에 시간이 걸린다. 진행률은 약 30초마다 표시되며, 중단 후 같은 셀을 다시 실행하면 이미 저장된 PNG는 건너뛴다. 기존 split은 유지한다.

```python
!python -u "{PROJECT}/scripts/pilgram2_roi_pilot.py" prepare {COMMON} \
  --train-per-method {TRAIN_PER_METHOD} --val-per-method {VAL_PER_METHOD} \
  --test-per-method {TEST_PER_METHOD} --workers 2
assert _exit_code == 0, "준비 실패: 위 오류 확인 후 Cell 4 재실행"
```

## Cell 5 — 기존 ROI best.pt 평가

학습 전에 원본+필터 5개를 같은 test 이미지로 평가한다. 결과를 보고 필터·학습 설정을 바꾸면 그 이후 결과는 개발용으로 해석해야 한다.

```python
!python -u "{PROJECT}/scripts/pilgram2_roi_pilot.py" evaluate {COMMON} \
  --strategies baseline --output-root "{EVALUATION_ROOT}" \
  --batch-size 32 --workers 2
assert _exit_code == 0, "Baseline 평가 실패"
```

## Cell 6 — 추가 학습

source_ft: 원본으로 같은 횟수 추가 학습. augmentation: 원본/필터 두 view를 정답으로 학습. dann: 원본 판별 loss + 원본/필터 domain-adversarial loss(가중치0.1, GRL 점진 증가). DANN은 필터 view의 real/fake 정답을 loss·모델 선택에 사용하지 않는다.

3개 모델을 순서대로 학습한다. 원본 validation만으로 best 선택, 각 모델 동일 3 epochs. epoch당 요약+30초 간격 진행 출력. 완료 run은 skip, 미완료 run은 last.pt의 완료 epoch 다음부터 재개한다. epoch 중간에 끊기면 해당 epoch는 다시 실행한다.

```python
!python -u "{PROJECT}/scripts/pilgram2_roi_pilot.py" train {COMMON} \
  --strategies source_ft augmentation dann --runs-root "{RUNS_ROOT}" \
  --epochs 3 --batch-size 16 --workers 2 \
  --learning-rate 1e-5 --weight-decay 1e-4 --domain-weight 0.1
assert _exit_code == 0, "학습 실패: 위 오류 확인 후 Cell 6 재실행"
```

## Cell 7 — 전후 동일 조건 평가

baseline+source_ft+augmentation+dann의 원본/필터 5개, 총 24조건 평가. Cell 5의 baseline 평가가 완료되었다면 같은 결과를 재사용한다. AUC·Real FPR·Fake FNR·기법별 AUC 및 paired 예측을 저장한다.

```python
!python -u "{PROJECT}/scripts/pilgram2_roi_pilot.py" evaluate {COMMON} \
  --strategies baseline source_ft augmentation dann --runs-root "{RUNS_ROOT}" \
  --output-root "{EVALUATION_ROOT}" --batch-size 32 --workers 2
assert _exit_code == 0, "평가 실패: 위 오류 확인 후 Cell 7 재실행"
```

## Cell 8 — 결과 확인

AUC는 클수록 좋고, Real FPR/Fake FNR은 작을수록 좋다. 비율은 0~1로 저장한다. 분석할 때 아래 CSV 3개를 전달하면 된다. WildDeepfake·unseen 필터는 이번 파일럿에 포함하지 않는다.

```python
import pandas as pd
display(pd.read_csv(EVALUATION_ROOT/"model_summary.csv").round(4))
display(pd.read_csv(EVALUATION_ROOT/"condition_summary.csv")[[
    "strategy","condition","roc_auc","real_fpr","fake_fnr","Deepfakes_auc","Face2Face_auc"
]].round(4))
print("분석 파일: model_summary.csv, condition_summary.csv, baseline_deltas.csv")
print(EVALUATION_ROOT)
```
