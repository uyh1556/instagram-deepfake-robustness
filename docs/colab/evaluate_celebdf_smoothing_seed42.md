# CelebDFv2_Xception_CLIP_Smoothing_Evaluation_Seed42

재학습·UDA 없음. 방금 학습한 두 best.pt를 고정하고 같은 Test1824장/114영상에
원본 + 스무딩25/50/75%를 적용한다. 총2모델×4조건×1824=14592 frame predictions.

## 고정 평가 설계

- 기존80/10/10 manifest를 **그대로 읽음**, 새 분할 생성 안 함.
- native1.5x frozen crop → 기존 기하 마스크 bilateral smoothing → 모델별 전처리.
- sigmaColor35 / sigmaSpace ROI 짧은변의2%, 혼합비율만25/50/75%로 변경.
- Real/Fake에 같은 규칙 적용. 얼굴 검출 실패 fallback은 보정하지 않고 유지·집계.
- Instagram 공식 알고리즘이 아님. 가짜 artifact도 지워질 수 있는 synthetic perturbation.
- Xception299/Normalize0.5, CLIP224/공식normalize 그대로. JPEG/noise/재검출 추가 없음.
- Primary: 영상 평균 점수 AUC·Real FPR·Fake FNR, threshold0.5 고정.
- Secondary: frame metrics, detected-only frame metrics, 예측 점수 변화, 정탐→오탐 변화.
- paired video bootstrap1000회로 변화량 CI 제공. 같은6명에 조건부인 CI이며 인물 일반화 CI 아님.
- 원본 결과도 다시 평가하여 기존test_metrics.json과 비교. 최대지표차이0.002를 넘으면 중지.
- 전체조건을 보고하고 결과를 보고 강도나checkpoint/threshold를 선택하지 않음.
- 새seed42 best는 자체split용이며 공식test/다른CelebDF모델 결과와 직접 비교 금지.

## Cell1 — 설치

새 코드 GitHub push 후 실행. 기존 학습코랩 뒤 같은 runtime에서도 가능.
PIL/torch는 notebook kernel에서 import하지 않음. 별도Python이 설치된 버전을 사용함.

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
assert (PROJECT/'scripts/evaluate_celebdf_smoothing.py').is_file(),'새 코드 push 필요'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-celebdf-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c',"import torch,clip,cv2; assert torch.cuda.is_available(),'GPU 런타임 선택'; print(torch.cuda.get_device_name(0))"],check=True)
```

## Cell2 — 경로·두 checkpoint 로컬 복사

manifest는 Drive에 있는 기존 확정본 그대로 사용. checkpoint만 로컬로 복사해서 읽는다.

```python
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
DRIVE_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
DATA_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
MANIFEST_ROOT=ROOT/'manifests/celebdf_v2_identity_disjoint_v1'
DRIVE_RUNS=ROOT/'runs/celebdf_v2_supervised_baseline_v1'
LOCAL_RUNS=Path('/content/celebdf_supervised_eval_checkpoints')
CACHE_ROOT=Path('/content/celebdf_identity_smoothing_cache_v1')
OUTPUT_ROOT=ROOT/'evaluations/celebdf_identity_smoothing_v1/seed42'
for model in ['xception','clip_b16']:
    name=f'{model}_celebdf_identity_v1_seed42'
    source=DRIVE_RUNS/name
    dest=LOCAL_RUNS/name
    dest.mkdir(parents=True,exist_ok=True)
    for filename in ['best.pt','test_metrics.json']:
        path=source/filename
        assert path.is_file(),path
        local=dest/filename
        if not local.exists() or local.stat().st_size!=path.stat().st_size:
            print('Copying:',model,filename,flush=True)
            shutil.copy2(path,local)
COMMON=['--data-root',str(DATA_ROOT),'--manifest-root',str(MANIFEST_ROOT),
        '--runs-root',str(LOCAL_RUNS),'--cache-root',str(CACHE_ROOT),'--output-root',str(OUTPUT_ROOT),'--seed','42']
print('Results:',OUTPUT_ROOT)
```

## Cell3 — Test만 추출

이미 학습코랩에서 전체이미지를 풀었다면 그대로 재사용.
새런타임이면 기존TAR3개에서 현재test와preview용train2장만 추출.
공식test TAR 하나만 풀면 새test 중 일부가 누락됨. Train 전체·Validation은 가져오지 않음.

```python
subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/extract_celebdf_evaluation_images.py'),
    '--drive-data',str(DRIVE_DATA),'--data-root',str(DATA_ROOT),'--manifest-root',str(MANIFEST_ROOT)],check=True)
```

## Cell4 — Native smoothing cache·Train preview

두 모델에서 같은 native 변형 이미지를 사용하도록 한번만 생성.
PNG lossless cache, 원본 보존. 진행률 표시. Train2장 preview는 테스트 강도 선택용이 아님.

```python
subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/evaluate_celebdf_smoothing.py'),
    'prepare',*COMMON,'--workers','2'],check=True)
preview=CACHE_ROOT/'train_preview.png'
if preview.is_file():
    from IPython.display import display,Image as DisplayImage
    display(DisplayImage(filename=str(preview)))
```

## Cell5 — 두 모델·원본+3단계 평가

진행률·경과시간·예상남은시간이 한 줄에서 갱신됨. 완료조건만 재사용하므로 중단 후 재실행 가능.
첫번째원본조건이 baseline과 재현되는지 확인 후 스무딩변화를 해석.

```python
subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/evaluate_celebdf_smoothing.py'),
    'evaluate',*COMMON,'--models','xception','clip_b16','--batch-size','64','--workers','2',
    '--bootstrap-repeats','1000'],check=True)
```

## Cell6 — 요약표

paired_deltas.csv의 FPR/FNR 변화량은0~1비율이므로100을 곱하면 percentage point.
표의primary는video, frame 결과도같은CSV에 포함됨. AUC 변화가없어도FPR/FNR 변화 확인.

```python
import pandas as pd
summary=pd.read_csv(OUTPUT_ROOT/'condition_summary.csv')
cols=['model','condition','video_auc','video_accuracy','video_real_fpr','video_fake_fnr']
table=summary[cols].copy()
for c in ['video_accuracy','video_real_fpr','video_fake_fnr']:
    table[c]=table[c].map(lambda x:f'{x:.2%}')
table['video_auc']=table.video_auc.map(lambda x:f'{x:.4f}')
display(table)
delta=pd.read_csv(OUTPUT_ROOT/'paired_deltas.csv')
display(delta[['model','condition','delta_video_auc','delta_video_real_fpr','delta_video_fake_fnr']])
print('Saved:',OUTPUT_ROOT)
```
