# CelebDFv2_Public_Detectors_Aug_vs_UDA_Custom_Pilot

RECCE/UCF/SELFI 공개 checkpoint에서 추가 학습. 기존 공식 테스트 결과와 별개인 custom 80/10/10 인물 분할.
FF++ c23 source train 3000(Real1500/Deepfakes750/Face2Face750), val600.
UDA target: 47명 학습 풀에서 정답을 사용하지 않고 무작위3000장, 영상당 최대4장.
Test: 분리된6명 전체1824장. Celeb validation6명은 사용하지 않음.
모델 선택은 clean FF++ validation frame AUC. 기존 crop/모델별 정규화 유지, 추가 JPEG 없음.
스무딩은 기존 masked bilateral 25/50/75%이며 실제 Instagram 알고리즘은 아님.
동일3epochs, LR1e-5, BN running statistics 고정. RECCE/UCF 원논문의 보조 손실 재현이 아닌 분류 fine-tuning.
Source FT / Source augmentation / UDA normal / UDA smoothing + 재학습 없는 baseline 비교.
모든 조건은 같은 source 순회 수를 사용하되, UDA는 별도의 비라벨 target과 추가 domain loss를 사용함.
테스트 인물6명 및 이전 평가 노출 한계가 있으므로 최종 독립 평가라고 주장하지 않음.

새 TAR를 Drive `instagram-deepfake-robustness/data/public_c23_adaptation_custom_pilot_v2/`에 업로드하고 코드 push 후 실행.

## Cell 1 — 설치

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess,sys,shutil,json
PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-public-detector-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 경로

모델 하나씩 돌리려면 MODELS만 변경. 같은 TAR와 평가 데이터를 사용함.

```python
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
DRIVE_DATA=ROOT/'data/public_c23_adaptation_custom_pilot_v2'
PILOT_ROOT=Path('/content/public_c23_adaptation_custom_pilot_v2')
ASSET_ROOT=Path('/content/public_detector_assets_v1')
RUNS_ROOT=ROOT/'runs/public_c23_adaptation_custom_pilot_v2'
OUTPUT_ROOT=ROOT/'evaluations/public_c23_adaptation_custom_pilot_v2/seed42'
MODELS='recce ucf selfi'
STRATEGIES='source_ft source_aug uda_normal uda_smoothing'
COMMON=f'--pilot-root "{PILOT_ROOT}" --test-root "{PILOT_ROOT}" --asset-root "{ASSET_ROOT}" --runs-root "{RUNS_ROOT}" --output-root "{OUTPUT_ROOT}" --models {MODELS} --strategies {STRATEGIES} --epochs 3 --batch-size 16 --workers 2 --seed 42'
```

## Cell 3 — 공개 모델 준비

```python
ASSET_ROOT.mkdir(parents=True,exist_ok=True)
for model in ['recce','ucf','selfi']:
    source=ROOT/'checkpoints/public_detectors_v1'/f'{model}_public.pth'
    dest=ASSET_ROOT/source.name
    if source.is_file() and not dest.exists():shutil.copy2(source,dest)
!python -u "{PROJECT}/scripts/evaluate_public_detectors_smoothing.py" setup --asset-root "{ASSET_ROOT}"
assert _exit_code==0,'공개 모델 준비 실패'
```

## Cell 4 — 데이터 TAR 하나 해제

```python
archive=DRIVE_DATA/'public_detector_c23_custom_source_target_test_seed42_v2.tar'
if not (PILOT_ROOT/'protocol.json').is_file():
    assert archive.is_file(),archive
    local=Path('/content/public_custom_pilot.tar')
    print('Copying pilot TAR',flush=True)
    shutil.copy2(archive,local)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True)
    local.unlink()
cfg=json.loads((PILOT_ROOT/'protocol.json').read_text())
print('Training pools:',cfg['counts'],'Test:',cfg['test_images'],flush=True)
```

## Cell 5 — 추가 학습

완료 run은 건너뛰고 미완료 run은 last.pt에서 재개. batch 진행률·경과시간·ETA 표시.

```python
!python -u "{PROJECT}/scripts/public_c23_adaptation_pilot.py" train {COMMON}
assert _exit_code==0,'학습 실패: 위 오류 확인'
```

## Cell 6 — 원본 + 스무딩 3단계 평가

3모델 × baseline 포함5전략 ×4조건 ×1824장. 완료 조건 재사용.

```python
!python -u "{PROJECT}/scripts/public_c23_adaptation_pilot.py" evaluate {COMMON}
assert _exit_code==0,'평가 실패: 위 오류 확인'
```

## Cell 7 — 이미지 단위 결과

```python
import pandas as pd
table=pd.read_csv(OUTPUT_ROOT/'frame_summary.csv')
for col in ['accuracy','f1','real_fpr','fake_fnr']:
    table[col]=table[col].map(lambda x:f'{x:.2%}')
display(table)
print('Saved:',OUTPUT_ROOT)
```
