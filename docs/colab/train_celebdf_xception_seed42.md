# CelebDFv2_Xception_IdentitySplit_Seed42

정식 원본 baseline: ImageNet 사전학습 Xception 전체 fine-tuning.
기존 FF++ 탐지 checkpoint, smoothing, JPEG/noise augmentation, UDA는 사용하지 않는다.
**새 코드 push 후** 실행. 아래 Cell2의 MODEL만 clip_b16으로 바꾸어도 같은 엔진을 쓴다.

## 분할과 학습 조건

- V2 기존 native frozen1.5x face crop 사용, fallback 유지. 재검출/재크롭 없음.
- 전체 유명인59개 filename ID를 seed42로 약80/10/10, Train47 / Val6 / Test6으로 분리.
- fake 파일의 두 ID가 서로 다른 분할이면 해당 영상 제외; source/driver 방향은 추정하지 않음.
- YouTube-real은 인물 ID 미확인이므로 첫 통제 실험에서 제외.
- **공식 Test와 별도 자체 분할**: 기존 공식 Test 이미지 일부도 새 Train에 포함될 수 있음.
  공식 Test 성능으로 보고하거나 이전 official-test 파일럿과 직접 비교하지 않음.
- 학습 pool Real7441 / Fake54832; 매 epoch Real7441 + 무작위 Fake7441 =14882장.
  Fake subset은 epoch마다 변경. Val1680장(105영상), Test1824장(114영상) 전체 사용.
  80/10/10은 인물 수 기준이며 이미지 수 비율과는 다름. Test 인물6명의 한계를 명시함.
- 모든 모델 같은 split, epoch sampling, 좌우 반전0.5. 원본 validation **video AUC**로 best 선택.
- 기존 test를 이미 살펴본 연구이므로 새 분할도 완전히 미사용된 독립 최종 평가라고 주장하지 않음.
- 15epochs / batch16 / AdamW / patience4 / LR1e-4 / weight decay1e-4.
- RGB native crop → bicubic299 → ToTensor → Normalize(mean=.5,std=.5).
- seed는 학습 seed이고 **split seed는42로 고정**. 모델별 경로 분리, 완료 skip / last.pt 자동재개.
- 출력은60초 간격 및 epoch 요약. 재개는 마지막 완료 epoch부터이며 중단 epoch는 다시 실행.

## Cell1 — Drive·코드·설치

노트북 kernel에서 PIL/torch를 import하지 않고 학습을 별도 Python으로 실행한다.
설치 뒤 session restart를 요구하지 않는다. GPU 런타임 권장 A100.

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
assert (PROJECT/'scripts/train_celebdf_baseline.py').is_file(), '새 코드부터 push하세요'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-celebdf-training.txt')],check=True)
subprocess.run([sys.executable,'-c',"import torch,timm,clip; from PIL import Image; assert torch.cuda.is_available(), 'GPU 런타임 선택'; print(torch.cuda.get_device_name(0), 'Pillow', Image.__version__)"],check=True)
```

## Cell2 — 변수

```python
MODEL='xception'
SEED=42
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
DRIVE_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
DATA_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
MANIFEST_ROOT=ROOT/'manifests/celebdf_v2_identity_disjoint_v1'
RUNS_ROOT=ROOT/'runs/celebdf_v2_supervised_baseline_v1'
COMMON=['--model',MODEL,'--data-root',str(DATA_ROOT),'--manifest-root',str(MANIFEST_ROOT),
        '--output-root',str(RUNS_ROOT),'--seed',str(SEED)]
print('Model:',MODEL,'| Runs:',RUNS_ROOT)
```

## Cell3 — 기존 전체 TAR 세 개 해제

새 TAR를 만들거나 올릴 필요 없음. **pilot TAR가 아닌 기존 전체 TAR3개와 inventory.json** 사용.
한 번에 하나만 로컬로 복사해 해제하고 임시 복사본만 삭제. Drive 원본 보존.
TAR 전체 hash 반복 검사는 생략, 크기·해제 오류만 확인하며 stderr를 표시함.

```python
names=['celebdf_v2_roi16_test_v1.tar','celebdf_v2_roi16_non_test_real_v1.tar','celebdf_v2_roi16_non_test_fake_v1.tar']
inventory=json.loads((DRIVE_DATA/'inventory.json').read_text())
sizes={x['name']:x['bytes'] for x in inventory['archives']}
if not DATA_ROOT.exists() and shutil.disk_usage('/content').free<16*2**30:
    raise RuntimeError('로컬 여유 공간16GiB 이상 필요: 새 런타임 권장')
for name in names:
    archive=DRIVE_DATA/name
    assert archive.is_file() and archive.stat().st_size==sizes[name], f'업로드 확인: {archive}'
    stem=name.removesuffix('.tar')
    manifest=DATA_ROOT/'manifests'/f'{stem}.csv'
    marker=Path('/content')/f'.supervised_{stem}.done'
    if marker.exists() and manifest.is_file():
        print('Already extracted:',name,flush=True)
        continue
    local=Path('/content')/name
    if not local.is_file() or local.stat().st_size!=sizes[name]:
        print('Copying:',name,flush=True)
        shutil.copy2(archive,local)
    print('Extracting:',name,flush=True)
    result=subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        print(result.stderr,flush=True)
        raise RuntimeError(f'{name}: 압축 해제 실패 — 위 오류 확인')
    assert manifest.is_file(),manifest
    marker.touch()
    local.unlink()
```

## Cell4 — 새 인물 분리 manifest 고정

공식 role은 provided_split 열에 보존하되 새 실험의 분할을 적용한다.
기존 pilot/FF++ manifest는 변경하지 않는다. 두 코랩이 같은 manifest를 생성하면 그대로 재사용한다.

```python
subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/create_celebdf_identity_split.py'),
    '--data-root',str(DATA_ROOT),'--output-root',str(MANIFEST_ROOT),
    '--split-seed','42','--allow-custom-repartition'],check=True)
```

## Cell5 — 학습

validation은 원본만 사용. **test를 학습 중 평가하지 않는다.**
중단 후 같은 셀 재실행: last.pt에서 epoch 단위 재개, completed는 건너뜀.

```python
subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/train_celebdf_baseline.py'),'train',*COMMON,
    '--epochs','15','--patience','4','--batch-size','16','--workers','2',
    '--log-interval-seconds','60'],check=True)
```

## Cell6 — 학습 후 원본 Test 평가

best.pt로 자체 Test 전체 평가. frame/video AUC·F1·Accuracy·Real FPR·Fake FNR,
test_predictions.csv 저장. video 점수는 해당 영상 frame 확률 평균.
임계값0.5 고정, test로 threshold 조정 없음. 스무딩/DA는 다음 실험으로 분리한다.

```python
subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/train_celebdf_baseline.py'),'test',*COMMON,
    '--batch-size','64','--workers','2','--log-interval-seconds','60'],check=True)
run=RUNS_ROOT/f'{MODEL}_celebdf_identity_v1_seed{SEED}'
result=json.loads((run/'test_metrics.json').read_text())
for level in ['frame','video']:
    m=result[level]
    print(f"{level}: AUC={m['auc']:.4f}, F1={m['f1']:.4f}, Real FPR={m['real_fpr']:.2%}, Fake FNR={m['fake_fnr']:.2%}")
print('Saved:',run)
```
