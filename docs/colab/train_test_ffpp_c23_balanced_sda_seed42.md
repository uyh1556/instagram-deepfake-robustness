# FFPP_c23_Class_Balanced_SDA_Seed42

Xception·CLIP B/16·DINOv2의 **SDA+스무딩에 타깃 클래스 가중치만 추가**하는 비교다.
같은 타깃 5,000장·샘플링·학습 조건을 사용하여, 진짜가 많은 타깃 분포가 미탐에 영향을 주는지 확인한다.

- 추가 학습: 세 모델 각각 1개, `sda_smoothing_balanced`. 기존 FF++ baseline best.pt에서 시작한다.
- 손실: **source CE + 클래스 가중치를 적용한 target CE**. FF++ source 손실에는 가중치를 추가하지 않는다.
- 타깃은 영상 균등 샘플링을 유지한다. 그 샘플러에서 클래스가 선택될 확률을 p라고 하면 가중치는 `1/(2p)`이다.
  진짜·가짜가 기대 손실에 각각 절반씩 기여하고, 평균 가중치는 1이 되도록 한다.
- 배치별 가중치 합으로 나누지 않고 `mean(가중치 × 개별 CE)`를 사용한다. 한 클래스만 있는 배치에서도 가중치가 사라지지 않는다.
- 데이터 목록·분할·스무딩·전처리·초기 가중치·검증 기준·seed는 기존 SDA와 동일하다.
- Source 6,000장/epoch, clean FF++ validation 1,200장, target pool 5,000장/epoch당 6,000회 노출.
- 최대 10epoch, patience 3, encoder/head LR 1e-5, batch 16, seed 42. BN 통계 고정 및 기존 fine-tuning 범위 유지.
- 타깃만 스무딩 25/50/75% 무작위 적용. validation은 FF++ 원본이며 macro4 frame AUC로 best를 선택한다.
- 공식 Celeb-DF test 8,288장 × 원본·스무딩 3단계, **총 12조건** 평가. threshold 0.5, frame 지표, FP32.
- 마지막 셀에서 저장된 기존 SDA+스무딩 예측을 읽어 AUC·FPR·FNR과 TPR@FPR≤5%를 비교한다.
  TPR@FPR≤5%는 테스트 ROC의 기술 지표이며 배포 임계값을 테스트에서 결정하지 않는다.
- 새 TAR 업로드는 필요 없다. 이 코드를 GitHub에 push한 뒤 실행한다. 기존 SDA 코드와 결과는 보존한다.

## Cell 1 — Drive·코드·설치

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess,sys,json,shutil
PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
assert (PROJECT/'scripts/ffpp_c23_balanced_supervised_da.py').is_file(),'새 코드 push 필요'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 변수

세 모델을 한 번에 실행한다. 나눠 실행하려면 MODELS만 변경한다.
UDA_MANIFEST_DIR은 기존 공통 목록인 xception_clip_b16을 유지한다.

```python
MODELS=['xception','clip_b16','dinov2_b14']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
UDA_MANIFEST_DIR=ROOT/'manifests/ffpp_c23_smoothing_adaptation_v1/seed42/xception_clip_b16'
RUNS_ROOT=ROOT/'runs/ffpp_c23_balanced_supervised_da_v1'
GROUP='_'.join(MODELS)
MANIFEST_ROOT=ROOT/'manifests/ffpp_c23_balanced_supervised_da_v1/seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations/ffpp_c23_balanced_supervised_da_v1/seed42'/GROUP
REFERENCE_EVALUATIONS=ROOT/'evaluations/ffpp_c23_supervised_da_v1/seed42'
CACHE_ROOT=Path('/content/celebdf_official_smoothing_cache_v1')
MODEL_ARGS=' '.join(MODELS)
COMMON=f'--models {MODEL_ARGS} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --uda-manifest-dir "{UDA_MANIFEST_DIR}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --output-root "{EVALUATION_ROOT}" --cache-root "{CACHE_ROOT}" --batch-size 16 --workers 2'
for name in ['protocol.json','target_unlabelled.csv','source_validation.csv']:
    assert (UDA_MANIFEST_DIR/name).is_file(),f'기존 UDA 목록 경로 확인: {UDA_MANIFEST_DIR/name}'
print('Models:',MODELS,'| new training: class-balanced SDA smoothing')
```

## Cell 3 — FF++ Train/Val 준비

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{FF_DATA}" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'
```

## Cell 4 — Celeb-DF·초기 baseline 준비

기존 런타임의 추출 완료 표시가 있으면 재추출을 건너뛴다.

```python
archives={
 'celebdf_v2_roi16_test_v1.tar':557015040,
 'celebdf_v2_roi16_non_test_real_v1.tar':826142720,
 'celebdf_v2_roi16_non_test_fake_v1.tar':5679564800,
}
for name,size in archives.items():
    source=CDF_DATA/name
    assert source.is_file(),source
    manifest=TARGET_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv'
    marker=Path('/content')/f'.sda_balance_{name}.done'
    available=any((Path('/content')/f'.{prefix}_{name}.done').exists() for prefix in ['adapt','sda','sda_balance'])
    if name=='celebdf_v2_roi16_test_v1.tar':
        available=available or Path('/content/.celebdf_official_roi16_test_v1.done').exists()
    if available and manifest.is_file():
        print('Already extracted:',name,flush=True)
        continue
    assert source.stat().st_size==size,f'업로드 미완료 또는 다른 TAR: {source}'
    assert shutil.disk_usage('/content').free>2*size+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name
    print('Copying:',name,flush=True)
    shutil.copy2(source,local)
    assert local.stat().st_size==size,'로컬 복사 미완료'
    print('Extracting:',name,flush=True)
    result=subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        print(result.stderr)
        raise RuntimeError('Celeb-DF 압축 해제 실패')
    assert manifest.is_file(),manifest
    marker.touch()
    local.unlink()
for model in MODELS:
    name=f'{model}_ffpp_c23_roi16_baseline_v1_seed42'
    source=BASELINE_DRIVE_RUNS/name
    dest=BASELINE_RUNS/name
    for filename in ['best.pt','config.json','completed.json']:
        assert (source/filename).is_file(),f'Baseline 완료 확인: {source/filename}'
    dest.mkdir(parents=True,exist_ok=True)
    for filename in ['best.pt','config.json','completed.json']:
        shutil.copy2(source/filename,dest/filename)
    print('Initial baseline ready:',model,flush=True)
```

## Cell 5 — 클래스 보정 SDA+스무딩 학습

가중치와 진행률·ETA·검증 요약을 표시한다. 완료 run은 건너뛰고 last.pt에서 재개한다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_balanced_supervised_da.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --encoder-lr 1e-5 --head-lr 1e-5
assert _exit_code==0,'클래스 보정 SDA 학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — 원본·스무딩 3단계 평가

새 세 체크포인트만 평가한다. 기존 SDA·UDA·baseline은 재추론하지 않는다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_balanced_supervised_da.py" evaluate {COMMON} --batch-size 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 기존 SDA+스무딩과 비교

기존 결과의 xception_clip_b16·dinov2_b14 등 모델별 폴더를 자동으로 찾는다.
기존 파일이 없으면 새 결과만 출력하고 안내한다. 재학습·재평가는 하지 않는다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_balanced_supervised_da.py" compare {COMMON} --reference-output-root "{REFERENCE_EVALUATIONS}"
assert _exit_code==0,'비교 실패: 위 오류 확인'
import csv
from IPython.display import Markdown,display
rows=list(csv.DictReader((EVALUATION_ROOT/'class_balance_comparison.csv').open()))
for model in MODELS:
    table=[f'### {model}', '', '|학습|조건|AUC|Real FPR|Fake FNR|TPR@FPR≤5%|',
           '|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['model']==model:
            label='기존 SDA+스무딩' if r['training']=='SDA smoothing original' else '클래스 보정 SDA+스무딩'
            table.append(f"|{label}|{r['condition']}|{float(r['auc']):.4f}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['tpr_at_fpr_5pct']):.2%}|")
    display(Markdown('\n'.join(table)))
print('Saved:',EVALUATION_ROOT)
```
