# CelebDFv2_RECCE_UCF_SELFI_Smoothing_Evaluation

**재학습/UDA 없음.** 공개 탐지 가중치를 그대로 사용해 원본/스무딩 3단계 평가.
RECCE·UCF는DeepfakeBench FF++ 탐지 checkpoint, SELFI는저자공개 B16+ArcFace checkpoint.
세파일 실제다운로드·strict loading·CPU추론확인. 공식 Celeb-DF V2 Test518영상/8288frames 사용.
총3모델×4조건×8288=99456 frame predictions. 기존 자체114영상test와구분.

## 실험에서지킬조건

- 공식test manifest그대로, train/validation사용안함. 세모델 같은frame/조건으로paired비교.
- 원본native MTCNN1.5x crop -> 이전과같은maskedbilateral blend25/50/75% -> 모델별preprocessing.
- 검출실패fallback289장은보정없이유지. 원본도재크롭/추가JPEG하지않음.
- RECCE/UCF: OpenCV bicubic256, mean/std0.5. SELFI: OpenCV bicubic224, 공개YAML ImageNet mean/std.
- SELFI의공개checkpoint에는학습config가없음. README는FF++c23, YAMLtemplate은FF-NT를표시하므로
  정확한학습subset은미확인. 이설정차이/출처는protocol.json에기록하며임의로CLIP정규화로바꾸지않음.
- RECCE는DeepfakeBench의2-logit재현버전이다. 보정논문의원래RECCE299/1-logit와동일checkpoint아님.
- 기존nativeMTCNNcrop은공개모델의RetinaFace/aligncrop과다를수있음. paired sensitivity테스트이지논문수치재현아님.
- SELFI는arXivpreprint, 논문존재·저자계정·checkpoint확인; 널리확립된표준이라고표현하지않음.
- threshold0.5고정, 원본및스무딩AUC/FPR/FNR모두보고. 성능저하보장없으며강도선별안함.
- primary video AUC와FPR/FNR, frame결과도저장. pairedvideo1000bootstrap CI는인물clusterCI아님.
- 기존Celeb-DF직접학습baseline은다른학습/평가조건이므로전체원점수를직접순위비교하지않음.

## Cell1 — Drive·저장소·설치

새코드먼저push. kernel에서PIL/torchimport안함. 라이브러리는별도Python실행.

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess,sys,json,shutil
PROJECT=Path('/content/instagram_deepfake_robustness')
REPO='https://github.com/uyh1556/instagram-deepfake-robustness.git'
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone',REPO,str(PROJECT)],check=True)
assert (PROJECT/'scripts/evaluate_public_detectors_smoothing.py').is_file(),'새코드push필요'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-public-detector-eval.txt')],check=True)
subprocess.run([sys.executable,'-c',"import torch;assert torch.cuda.is_available(),'GPU런타임선택';print(torch.cuda.get_device_name(0))"],check=True)
```

## Cell2 — 변수

```python
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
DRIVE_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
DATA_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
ASSET_ROOT=Path('/content/public_detector_assets_v1')
DRIVE_WEIGHTS=ROOT/'checkpoints/public_detectors_v1'
CACHE_ROOT=Path('/content/public_celebdf_official_smoothing_cache_v1')
OUTPUT_ROOT=ROOT/'evaluations/public_recce_ucf_selfi_smoothing_v1/seed42'
COMMON=f'--asset-root "{ASSET_ROOT}" --data-root "{DATA_ROOT}" --cache-root "{CACHE_ROOT}" --output-root "{OUTPUT_ROOT}" --seed 42'
print('Results:',OUTPUT_ROOT)
```

## Cell3 — 공개가중치준비

첫실행에공개URL에서자동다운로드: RECCE191MB/UCF188MB/SELFI609MB.
코드commit·전체checkpointSHA를한번검증함. 실패시랜덤모델로대체하지않음.
ArcFace포함전체SELFI파일이므로별도얼굴인식가중치업로드불필요.
완료파일은Drive에복사해다음runtime에서재사용.

```python
ASSET_ROOT.mkdir(parents=True,exist_ok=True)
for model in ['recce','ucf','selfi']:
    source=DRIVE_WEIGHTS/f'{model}_public.pth'
    dest=ASSET_ROOT/source.name
    if source.is_file() and not dest.exists():
        print('Copying existing weights:',model,flush=True)
        shutil.copy2(source,dest)
!python -u "{PROJECT}/scripts/evaluate_public_detectors_smoothing.py" setup --asset-root "{ASSET_ROOT}"
assert _exit_code==0,'공개가중치/소스준비실패: 위오류확인'
DRIVE_WEIGHTS.mkdir(parents=True,exist_ok=True)
for model in ['recce','ucf','selfi']:
    source=ASSET_ROOT/f'{model}_public.pth'
    dest=DRIVE_WEIGHTS/source.name
    if not dest.exists():shutil.copy2(source,dest)
```

## Cell4 — 공식Test TAR하나해제

train TAR불필요. 이미전체데이터를해제한같은런타임이면그대로재사용.

```python
archive=DRIVE_DATA/'celebdf_v2_roi16_test_v1.tar'
manifest=DATA_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv'
if not manifest.is_file():
    assert archive.is_file(),archive
    inventory=json.loads((DRIVE_DATA/'inventory.json').read_text())
    expected=next(x['bytes'] for x in inventory['archives'] if x['name']==archive.name)
    assert archive.stat().st_size==expected,'Test TAR업로드미완료'
    local=Path('/content/public_eval_test.tar')
    if not local.exists() or local.stat().st_size!=expected:
        print('Copying official test TAR',flush=True)
        shutil.copy2(archive,local)
    result=subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        print(result.stderr)
        raise RuntimeError('TAR해제실패')
    assert manifest.is_file(),manifest
    local.unlink()
print('Official test manifest:',manifest)
```

## Cell5 — 같은스무딩이미지준비

nativecrop에적용한PNGcache를세모델이공유. 진행률표시, 원본보존.

```python
!python -u "{PROJECT}/scripts/evaluate_public_detectors_smoothing.py" prepare {COMMON} --workers 2
assert _exit_code==0,'스무딩준비실패: 위오류확인'
```

## Cell6 — 세모델평가

!python -u로직접실행. progress bar에batch진행률·경과시간·예상남은시간표시.
현재모델/조건별frame·video 결과저장. 완료조건재사용, 중단시동일셀재실행.

```python
!python -u "{PROJECT}/scripts/evaluate_public_detectors_smoothing.py" evaluate {COMMON} \
  --models recce ucf selfi --batch-size 16 --workers 2 --bootstrap-repeats 1000
assert _exit_code==0,'평가실패: 위오류확인'
```

## Cell7 — 결과표

```python
import pandas as pd
summary=pd.read_csv(OUTPUT_ROOT/'condition_summary.csv')
table=summary[['model','condition','frame_auc','video_auc','video_accuracy','video_real_fpr','video_fake_fnr']].copy()
for c in ['video_accuracy','video_real_fpr','video_fake_fnr']:
    table[c]=table[c].map(lambda x:f'{x:.2%}')
display(table)
display(pd.read_csv(OUTPUT_ROOT/'paired_deltas.csv')[['model','condition','delta_video_auc','delta_video_real_fpr','delta_video_fake_fnr']])
print('Saved:',OUTPUT_ROOT)
```
