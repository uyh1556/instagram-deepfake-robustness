# Smoothing_Soft_Label_UDA_Seed42

완료한 Teacher–Student UDA를 기준으로 **soft 정답·CDF 가중치 0.5 / 0.25** 두 조건만 새로 학습하고 평가한다. 기본 모델은 CLIP B/16과 Xception이다. 모델별 코랩을 나누려면 Cell 2의 `MODELS`를 하나만 남긴다. 먼저 새 코드를 GitHub에 push한다.

|조건|CDF 학습 목표|최대 CDF 가중치|
|---|---|---:|
|기존 Hard UDA — 저장 결과 재사용|교사 argmax를 진짜/가짜로 확정|0.5|
|Soft UDA 0.5 — 새 학습|교사의 진짜·가짜 확률 그대로|0.5|
|Soft UDA 0.25 — 새 학습|교사의 진짜·가짜 확률 그대로|0.25|

- 두 조건 모두 **기존 FF++ baseline best.pt부터 독립 시작**한다. 완료한 hard UDA를 이어 학습하지 않는다. FF++ 원본+무작위 스무딩의 실제 정답 CE와 기존 인코더 학습 범위를 재사용한다.
- CDF 교사 확신 >=0.9인 표본에만 `-sum(teacher_probability * log(student_probability))`를 적용한다. Temperature=1, 확률 sharpening 없음. 선택된 손실 합을 **전체 CDF 배치 크기**로 나누며 2epoch 동안 가중치를 선형 증가시킨다. soft CE는 교사 확률의 엔트로피를 포함하므로, 예측이 일치해도 0이 아닐 수 있다. hard/soft의 loss 숫자 자체를 성능으로 비교하지 않는다.
- 기존 hard 실험의 고정 validation·비라벨 CDF 목록과 새 목록을 비교한다. 같은 seed42·FF++ 6,000장/epoch·CDF 고정 pool 5,000장·영상균등 6,000노출/epoch·원본/약/중/강 관측 각25%·반대 좌우반전이다. **CDF 실제 정답으로 5:5를 만들지 않는다. 추가 스무딩은 없다.** 원본 영상·ROI·split·전처리·fallback은 기존 그대로다.
- 최대10epoch/patience3, LR1e-5, EMA0.999, BN 통계 고정. **FF++ validation 1,200장×4조건 평균 BER 최소**로 학생 best.pt를 선택한다. epoch0 후보이며 AUC guard는 없다. 추론은 학생 하나로 한다.
- 기존 source/target bilateral 및 validation 캐시를 재사용한다. 학습·평가는 직접 `!python -u` 실행하여 진행률이 표시된다. 첫 캐시 생성 시간은 필요하다. 기존 엔진 파일은 수정하지 않는다.
- CDF 공식 test 8,288장×원본/약/중/강에서 FP32 frame FPR/FNR/BER/raw-margin AUC를 평가한다. 기존 Baseline·Matched Aug·Hard UDA는 저장된 예측과 평가 조건을 확인하여 재사용한다. 이들이 누락되면 새 추론 전에 경로를 안내하고 중단한다. 기존 모델을 재학습하거나 hard 모델을 재추론하지 않는다.
- 결과는 새 `ffpp_c23_smoothing_mean_teacher_soft_v1` 폴더에 저장한다. 완료 조건은 skip, 중단 시 마지막 완료 epoch의 학생·교사·optimizer를 복원한다. epoch 도중 중단하면 그 epoch부터 다시 진행한다. 이 CDF test는 반복 확인한 개발 평가이며, 인물 중복은 허용하되 동일 test 영상은 적응에서 제외한다. filename 기반 parent 제외는 완전한 source/driver 검증을 의미하지 않는다.

## Cell 1 — 설치·갱신

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, json, shutil

PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
assert (PROJECT/'scripts/ffpp_c23_smoothing_mean_teacher_soft.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['clip_b16','xception']  # 각각 돌리려면 ['clip_b16'] 또는 ['xception']
STRATEGIES=['soft_w05','soft_w025']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_mean_teacher_soft_v1'
GROUP='_'.join(MODELS)
RUNS_ROOT=ROOT/'runs'/EXPERIMENT
MANIFEST_ROOT=ROOT/'manifests'/EXPERIMENT/'seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations'/EXPERIMENT/'seed42'/GROUP
HARD_MANIFEST_ROOT=ROOT/'manifests/ffpp_c23_smoothing_mean_teacher_v1/seed42'
HARD_EVALUATIONS=ROOT/'evaluations/ffpp_c23_smoothing_mean_teacher_v1/seed42'
VALIDATION_CACHE=Path('/content/smoothing_mean_teacher_source_validation_v1')
TRAIN_CACHE=Path('/content/smoothing_pivot_bilateral_cache_v1')
TARGET_TRAIN_CACHE=Path('/content/smoothing_mean_teacher_target_cache_v1')
CDF_CACHE=Path('/content/celebdf_official_smoothing_cache_v1')
CDF_PIXEL_CACHE=Path('/content/smoothing_ranking_cdf_bilateral_cache_v1')
MODEL_ARGS=' '.join(MODELS)
STRATEGY_ARGS=' '.join(STRATEGIES)
COMMON=f'--models {MODEL_ARGS} --strategies {STRATEGY_ARGS} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --output-root "{EVALUATION_ROOT}" --validation-cache "{VALIDATION_CACHE}" --train-cache "{TRAIN_CACHE}" --target-train-cache "{TARGET_TRAIN_CACHE}" --cache-root "{CDF_CACHE}" --evaluation-pixel-cache "{CDF_PIXEL_CACHE}" --hard-manifest-root "{HARD_MANIFEST_ROOT}" --hard-evaluations "{HARD_EVALUATIONS}" --workers 2'
print('Models:',MODELS,'| New strategies:',STRATEGIES,flush=True)
print('Runs:',RUNS_ROOT,flush=True)
```

## Cell 3 — FF++ 및 초기 baseline 준비

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{FF_DATA}" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'

for model in MODELS:
    name=f'{model}_ffpp_c23_roi16_baseline_v1_seed42'
    source=BASELINE_DRIVE_RUNS/name
    dest=BASELINE_RUNS/name
    dest.mkdir(parents=True,exist_ok=True)
    for filename in ['best.pt','config.json','completed.json']:
        src,dst=source/filename,dest/filename
        assert src.is_file(),src
        if dst.is_file() and dst.stat().st_size==src.stat().st_size:
            continue
        print('Copying:',model,filename,flush=True)
        temp=dst.with_suffix(dst.suffix+'.copying')
        shutil.copy2(src,temp)
        assert temp.stat().st_size==src.stat().st_size,'복사 미완료'
        temp.replace(dst)
assert HARD_MANIFEST_ROOT.is_dir(),f'기존 hard UDA manifest 경로 확인: {HARD_MANIFEST_ROOT}'
assert HARD_EVALUATIONS.is_dir(),f'기존 hard UDA 평가 경로 확인: {HARD_EVALUATIONS}'
```

## Cell 4 — 기존 CDF TAR 준비

```python
archives={
    'celebdf_v2_roi16_test_v1.tar':557015040,
    'celebdf_v2_roi16_non_test_real_v1.tar':826142720,
    'celebdf_v2_roi16_non_test_fake_v1.tar':5679564800,
}
for name,size in archives.items():
    manifest=TARGET_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv'
    markers=[Path('/content')/f'.{p}_{name}.done' for p in ['adapt','sda','sda_balance','mixed_uda','consistency_c','response','pivot','ranking','mean_teacher']]
    if name=='celebdf_v2_roi16_test_v1.tar':
        markers += [Path('/content/.celebdf_official_roi16_test_v1.done'),Path('/content/.preservation_b_celebdf_official_test.done')]
    if manifest.is_file() and any(p.exists() for p in markers):
        print('Already extracted:',name,flush=True)
        continue
    source=CDF_DATA/name
    assert source.is_file() and source.stat().st_size==size,f'TAR 업로드 확인: {source}'
    assert shutil.disk_usage('/content').free>2*size+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name
    print('Copying:',name,flush=True)
    shutil.copy2(source,local)
    assert local.stat().st_size==size,'복사 미완료'
    print('Extracting:',name,flush=True)
    result=subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-5000:])
    assert manifest.is_file(),manifest
    (Path('/content')/f'.mean_teacher_{name}.done').touch()
    local.unlink()
```

## Cell 5 — Soft 0.5·0.25 학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_mean_teacher_soft.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --pseudo-threshold 0.9 --ramp-epochs 2 --ema-decay 0.999
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — 새 두 조건 평가·기존 hard 결과 병합

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_mean_teacher_soft.py" evaluate {COMMON} --batch-size 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표시

```python
import csv
from IPython.display import Markdown,display
labels={'baseline':'Baseline','augmentation':'Matched Aug','mean_teacher':'Hard UDA 0.5','soft_w05':'Soft UDA 0.5','soft_w025':'Soft UDA 0.25'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison_with_hard.csv').open()))
means=list(csv.DictReader((EVALUATION_ROOT/'condition_means_with_hard.csv').open()))
for model in MODELS:
    table=[f'### {model} — 원본·약·중·강 평균','','|방법|FPR|FNR|BER ↓|AUC ↑|','|---|---:|---:|---:|---:|']
    for r in means:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    table=['### 조건별 결과','','|방법|조건|FPR|FNR|BER|AUC|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{conditions[r['condition']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    for strategy in STRATEGIES:
        run=RUNS_ROOT/f'{model}_{strategy}_{EXPERIMENT}_seed42'
        done=json.loads((run/'completed.json').read_text())
        print(model,labels[strategy],'selected epoch:',done['selected_epoch'],'/ trained:',done['epochs_completed'])
print('Baseline·Matched Aug·Hard UDA는 기존 저장 결과입니다. 새 추론은 Soft 두 조건만 수행합니다.')
print('선택 기준: FF++ 4조건 평균 BER / 추론 모델: 학생 best.pt / CDF 실제 정답은 학습·선택에 사용하지 않음')
print('결과:',EVALUATION_ROOT)
```
