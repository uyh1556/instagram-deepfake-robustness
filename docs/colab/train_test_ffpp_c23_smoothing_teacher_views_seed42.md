# Smoothing_Teacher_View_UDA_Seed42

DINO에서 **CDF 원본 적응**과 **교사 원본 → 학생 스무딩 학습** 두 조건만 추가한다. 학습부터 CDF 평가·기존 결과 비교까지 포함한다. 먼저 새 코드를 GitHub에 push한다.

|조건|교사 CDF 입력|학생 CDF 입력|
|---|---|---|
|기존 일반 UDA — 저장 결과 재사용|원본·약·중·강 중 무작위|교사와 같은 스무딩 상태|
|새 `normal_only`|원본|원본|
|새 `clean_to_mixed`|원본|원본·약·중·강 중 무작위|

- 여기서 **원본은 우리가 스무딩을 추가하기 전 CDF 크롭**이다. 기존 압축·보정이 전혀 없다는 뜻은 아니다. 새 실험은 **적응 시 비라벨 원본 CDF를 확보할 수 있는 설정**이며, 이미 보정된 이미지만 주어지는 설정과 구분한다.
- 기본 `MODELS=['dinov2_b14']`. 나중에 CLIP 또는 Xception을 돌리려면 Cell 2의 MODELS만 바꾼다. 해당 모델의 기존 일반 UDA·학생 추가 스무딩 UDA·Matched Aug 평가가 필요하다.
- 두 조건 모두 **기존 FF++ baseline best.pt부터 독립 시작**한다. 이전 UDA/Aug 모델을 이어 학습하지 않는다. FF++ 원본+무작위 스무딩의 실제 정답 CE, 기존 인코더 학습 범위, 전처리, native ROI, fallback, BN 통계 고정을 유지한다.
- 고정 FF++ 6,000장/epoch와 검증 1,200장, 비라벨 CDF pool 5,000장을 재사용한다. CDF 6,000회 영상균등 노출/epoch이며 두 조건과 기존 UDA의 **이미지 경로·노출 순서·좌우반전 계획이 같다**. CDF 실제 정답을 표본선택·손실·검증에 사용하지 않는다.
- 교사 확신 >=0.9인 **hard 임시 정답**, 전체 CDF 배치 분모, 최대 가중치0.5·2epoch ramp, EMA0.999, LR1e-5를 유지한다. 클래스 가중치나 안정성 선별은 추가하지 않는다. 교사·학생은 반대 좌우반전을 사용한다.
- 최대10epoch/patience3. **FF++ 검증 1,200장×원본·약·중·강 평균 BER 최소**로 학생 best.pt 선택. epoch0 후보, AUC guard 없음. 추론에는 학생만 사용한다.
- 기존 EMA 학습 엔진·soft 실험 파일은 수정하지 않는다. source/target bilateral·검증 캐시를 재사용하고, 직접 `!python -u` 실행으로 학습·평가 진행률을 표시한다. 완료 run은 skip, 중단하면 마지막 완료 epoch의 학생·교사·optimizer에서 재개한다.
- 노출 CSV의 `condition`은 **실제 학생 입력 조건**, `teacher_condition`은 항상 normal이다. `scheduled_condition`은 기존 UDA의 무작위 조건 추첨을 보존한 비교용 값이다. 조건마다 별도 CSV를 저장한다.
- CDF 공식 test **8,288장×4조건**, FP32 frame FPR/FNR/BER/raw-margin AUC. 새 두 모델만 추론하며 기존 Baseline·Matched Aug·일반 UDA·학생 추가 스무딩 UDA는 저장된 결과를 읽는다. 누락·불일치는 새 추론 전에 안내한다.
- 결과를 **원본 성능 / 4조건 평균 / 강−원본 FPR·FNR·BER 변화 / 혼합 AUC**로 함께 표시한다. 혼합 AUC는 네 조건의 점수를 합쳐 계산하며 조건별 AUC 평균과 다르다. 강−원본 변화가 작더라도 성능 자체가 좋은지는 따로 확인해야 한다.
- CDF test는 반복 확인한 개발 평가다. 테스트 기반 임계값·epoch 선택은 하지 않는다. 인물 중복은 허용하지만 같은 테스트 영상·프레임은 적응에서 제외한다. filename 기반 parent 제외는 완전한 source/driver 검증이 아니다.
- 새 출력은 `ffpp_c23_smoothing_teacher_views_v1`. 기존 데이터·manifest·학습 결과는 유지한다. 새 TAR·zip 업로드는 필요 없다.

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_teacher_views.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['dinov2_b14']  # 다른 모델: ['clip_b16'] 또는 ['xception']
STRATEGIES=['normal_only','clean_to_mixed']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_teacher_views_v1'
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
assert HARD_MANIFEST_ROOT.is_dir(),f'기존 UDA manifest 경로 확인: {HARD_MANIFEST_ROOT}'
assert HARD_EVALUATIONS.is_dir(),f'기존 UDA 평가 경로 확인: {HARD_EVALUATIONS}'
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

## Cell 5 — 원본 적응 / 교사 원본 → 학생 혼합 학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_teacher_views.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --pseudo-threshold 0.9 --ramp-epochs 2 --ema-decay 0.999
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — 새 두 조건 평가·기존 결과 비교

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_teacher_views.py" evaluate {COMMON} --batch-size 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표시

```python
import csv
from IPython.display import Markdown,display

labels={
    'baseline':'Baseline','augmentation':'Matched Aug',
    'mean_teacher':'기존 일반 UDA','mean_teacher_smoothing':'기존 학생 추가 스무딩 UDA',
    'normal_only':'새 원본 → 원본','clean_to_mixed':'새 원본 → 원본·약·중·강',
}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
means=list(csv.DictReader((EVALUATION_ROOT/'condition_means_with_reference.csv').open()))
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison_with_reference.csv').open()))
summary=list(csv.DictReader((EVALUATION_ROOT/'domain_and_smoothing.csv').open()))
for model in MODELS:
    table=[f'### {model} — 원본·약·중·강 평균','','|방법|FPR|FNR|BER ↓|AUC ↑|','|---|---:|---:|---:|---:|']
    for r in means:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    table=['### 원본 성능과 스무딩 변화 분리','','|방법|원본 BER|원본 AUC|강−원본 FPR (%p)|강−원본 FNR (%p)|혼합 AUC|','|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{float(r['normal_balanced_error_rate']):.2%}|{float(r['normal_auc']):.4f}|{100*float(r['strong_minus_normal_real_fpr']):+.2f}|{100*float(r['strong_minus_normal_fake_fnr']):+.2f}|{float(r['pooled_auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    table=['### 조건별 결과','','|방법|조건|FPR|FNR|BER|AUC|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{conditions[r['condition']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    for strategy in STRATEGIES:
        run=RUNS_ROOT/f'{model}_{strategy}_{EXPERIMENT}_seed42'
        done=json.loads((run/'completed.json').read_text())
        print(model,labels[strategy],'선택 epoch:',done['selected_epoch'],'/ 학습:',done['epochs_completed'])
print('기존 네 조건은 저장된 결과입니다. 새 두 조건만 학습·추론합니다.')
print('CDF 진위 정답은 학습·선택에 사용하지 않습니다. 적응 시 스무딩 전 원본 CDF를 확보한 설정입니다.')
print('검증: FF++ 4조건 평균 BER / 추론: 학생 best.pt / 결과:',EVALUATION_ROOT)
```

