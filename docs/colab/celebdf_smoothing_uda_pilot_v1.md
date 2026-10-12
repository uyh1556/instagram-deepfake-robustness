# FFpp_to_CelebDFv2_Smoothing_UDA_Pilot_Seed42

기존 FF++ c0 ROI checkpoint -> Celeb-DF V2 원본/스무딩3강도 평가 -> 추가학습 -> 같은 test 재평가.
Instagram 공식 구현이 아닌 **피부 영역 근사 스무딩**이다.
원래 해상도의 frozen1.5x ROI crop -> 스무딩 -> 기존 bicubic299 -> Normalize(0.5).
PNG 재크롭/JPEG/noise 추가 없음. 검출 실패는 기존 letterbox fallback을 그대로 유지하며 보정하지 않는다.

| 역할 | 이미지 |
|---|---:|
| FF++ source train | 3,000 (Real1,000 + Deepfakes1,000 + Face2Face1,000) |
| FF++ original validation | 900 |
| Celeb-DF unlabelled target | 1,200 |
| Celeb-DF official test | 8,288 / 518영상 |

공식 test에 알려진 유명인 ID56개가 포함되어 있으므로, target은 해당 ID와 겹치지 않는
id59/id60/id61에만 한정된다. 생성 영상은 파일명에 나온 두 ID 모두 검사한다.
1200장 전체를 사용하며 class label 기준으로 샘플링하지 않는다. adaptation manifest에서
target label을 제거한다. target/test 영상 비중복 및 알려진 celebrity ID 비중복을 확인했다.
YouTube-real의 인물 정보는 확인되지 않았으므로 완전한 생체 identity 검증이라고 주장하지 않는다.
**target이3개 ID로 제한된 좁은 pilot이며 본 실험 일반화 근거로 과장하지 않는다.**

| 비교 | FF++ class loss | 추가 정보 |
|---|---|---|
| baseline | 학습 없음 | 기존 checkpoint |
| source_ft | 원본 FF++ | 추가 학습 대조군 |
| source_aug | 원본+스무딩 FF++ | target을 보지 않는 source augmentation |
| uda_normal | 원본 FF++ | 정답 없는 Celeb-DF 원본, domain loss |
| uda_smoothing | 원본 FF++ | 정답 없는 Celeb-DF 스무딩, domain loss |

모두 기존 best에서 독립적으로3epochs, LR1e-5. DANN domain weight0.1, GRL 점진 증가.
같은 source 3000장의 epoch 수와2-view forward를 맞춘다. 각 step에 target은 독립 무작위
샘플링한다. Best 선택은 **FF++ 원본 validation AUC만** 사용한다.
Target/test 정답으로 checkpoint/강도/하이퍼파라미터를 선택하지 않는다.
Target으로 가짜·진짜를 학습하는 supervised Celeb-DF fine-tuning은 포함하지 않는다.
All3 smoothing strengths appear in adaptation; no unseen-smoothing claim.

## Cell1 — Drive·저장소·설치

새 코드를 먼저 GitHub에 push하세요. 무거운 라이브러리는 별도 Python에서 실행하여
노트북 kernel Pillow 충돌을 피한다. ZIP 업로드 불필요.

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, json, shutil
PROJECT=Path('/content/instagram_deepfake_robustness')
REPO='https://github.com/uyh1556/instagram-deepfake-robustness.git'
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone',REPO,str(PROJECT)],check=True)
assert (PROJECT/'scripts/celebdf_smoothing_uda_pilot.py').is_file(), '먼저 새 코드 push'
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(PROJECT/'requirements-smoothing.txt')],check=True)
subprocess.run([sys.executable,'-c',"import torch,cv2; assert torch.cuda.is_available(), 'GPU 런타임으로 변경'; print(torch.cuda.get_device_name(0),cv2.__version__)"],check=True)
```

## Cell2 — 경로

TAR 하나에 source+target+test가 모두 들어 있다. 기존 대형 FF++/Celeb TAR를 별도 해제하지 않는다.
Drive에 pilot TAR와 pilot_inventory.json을 같은 폴더에 올린다.

```python
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
DRIVE_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
DATA_ROOT=Path('/content/celebdf_v2_smoothing_uda_pilot_v1')
CHECKPOINT=Path('/content/roi_c0_initial_best.pt')
CACHE_ROOT=Path('/content/celebdf_smoothing_uda_cache_v1')
RUNS_ROOT=ROOT/'runs/celebdf_v2_smoothing_uda_pilot_v1/seed42'
EVALUATION_ROOT=ROOT/'evaluations/celebdf_v2_smoothing_uda_pilot_v1/seed42'
COMMON=f'--data-root "{DATA_ROOT}" --checkpoint "{CHECKPOINT}" --cache-root "{CACHE_ROOT}" --runs-root "{RUNS_ROOT}" --output-root "{EVALUATION_ROOT}" --seed 42'
print('Results:',EVALUATION_ROOT)
```

## Cell3 — 작은 TAR·기존 모델

빠른 준비: 파일 존재·크기·압축 해제 성공 여부만 확인한다. TAR 전체 SHA256 검사는 생략한다.

```python
inventory=json.loads((DRIVE_DATA/'pilot_inventory.json').read_text())
archive=DRIVE_DATA/inventory['name']
assert archive.is_file() and archive.stat().st_size==inventory['bytes'], f'파일럿 TAR 업로드 확인: {archive}'
marker=Path('/content/.celebdf_smoothing_uda_pilot_v1.extracted')
if not marker.exists():
    local=Path('/content')/archive.name
    if not local.exists() or local.stat().st_size!=inventory['bytes']:
        print('Copying pilot TAR...',flush=True)
        shutil.copy2(archive,local)
    print('Extracting pilot TAR...',flush=True)
    result=subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        print(result.stderr,flush=True)
        raise RuntimeError('TAR 압축 해제 실패')
    assert (DATA_ROOT/'manifest.csv').is_file() and (DATA_ROOT/'protocol.json').is_file()
    marker.touch()
    local.unlink()  # local temporary archive only; Drive preserved
original_best=ROOT/'checkpoints/ffpp_c0_xception_face_roi_seed42/best.pt'
assert original_best.is_file(),original_best
if not CHECKPOINT.exists(): shutil.copy2(original_best,CHECKPOINT)
print(json.loads((DATA_ROOT/'protocol.json').read_text())['counts'])
```

## Cell4 — 원본·스무딩 cache

원본 해상도 crop에서 동일 mask/bilateral을 적용한 다음299로 resize한다.
스무딩 혼합비율25/50/75%, sigmaColor35, sigmaSpace ROI 크기의2%.
얼굴 검출 fallback은 변경하지 않는다. source validation은 원본만 저장한다.
첫 준비는 시간이 걸리며30초마다 출력, 재실행 시 PNG 재사용.

```python
!python -u "{PROJECT}/scripts/celebdf_smoothing_uda_pilot.py" prepare {COMMON} --workers 2
assert _exit_code==0, 'Cache 준비 실패'
```

## Cell5 — 기존 FF++ 모델 baseline 평가

학습 전에 같은 Celeb test8288장을 원본/약/중/강4조건에서 평가한다.
원본부터 성능이 낮으면 데이터셋 차이와 보정의 추가 영향을 구분한다.

```python
!python -u "{PROJECT}/scripts/celebdf_smoothing_uda_pilot.py" evaluate {COMMON} \
  --strategies baseline --batch-size 64 --workers 2
assert _exit_code==0, 'Baseline 평가 실패'
```

## Cell6 — 추가 학습4조건

각 source_ft/source_aug/uda_normal/uda_smoothing은 FF++ best에서 새로 시작한다.
완료 skip, 미완료 last.pt의 완료 epoch 뒤부터 재개. 출력은30초 간격+epoch 요약.
UDA에서는 source real/fake label만 CE에 사용하고 target class label은 읽지 않는다.

```python
!python -u "{PROJECT}/scripts/celebdf_smoothing_uda_pilot.py" train {COMMON} \
  --strategies source_ft source_aug uda_normal uda_smoothing \
  --epochs 3 --batch-size 16 --workers 2 --learning-rate 1e-5 \
  --weight-decay 1e-4 --domain-weight 0.1
assert _exit_code==0, '학습 실패'
```

## Cell7 — 전후 동일 test 평가

5모델×4조건=20조건. 완료된 baseline 결과 재사용.
Frame AUC/FPR/FNR와 **영상 평균 점수의 AUC/FPR/FNR** 모두 저장한다.

```python
!python -u "{PROJECT}/scripts/celebdf_smoothing_uda_pilot.py" evaluate {COMMON} \
  --strategies baseline source_ft source_aug uda_normal uda_smoothing \
  --batch-size 64 --workers 2
assert _exit_code==0, '평가 실패'
```

## Cell8 — 결과 확인

```python
import pandas as pd
summary=pd.read_csv(EVALUATION_ROOT/'condition_summary.csv')
display(summary[['strategy','condition','roc_auc','video_roc_auc','real_fpr','fake_fnr','video_real_fpr','video_fake_fnr']].round(6))
display(summary.pivot(index='strategy',columns='condition',values='video_roc_auc').round(6))
print('분석 파일: condition_summary.csv, baseline_deltas.csv')
print(EVALUATION_ROOT)
```

핵심 비교: uda_smoothing vs uda_normal, 두 UDA vs source_ft/source_aug.
미학습 target class label 활용 없이 얼마나 회복되는지 확인한다. 새로운 학습법의 novelty나
실제 Instagram 재현을 이 pilot 하나로 주장하지 않는다.
