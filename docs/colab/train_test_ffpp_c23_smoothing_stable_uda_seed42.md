# Smoothing_Agreement_UDA_Xception_Seed42

기본 Xception으로 **판단 일치 UDA**와 **무작위 제외 UDA**를 새로 학습하고 CDF 원본·약·중·강까지 평가한다. 먼저 새 코드를 GitHub에 push하고 이 노트북을 Colab에서 연다. Cell 2의 MODELS만 바꾸면 CLIP/DINO도 실행할 수 있다.

|조건|임시 정답 사용 규칙|
|---|---|
|기존 Baseline·Matched Aug·Hard UDA|완료된 동일 조건의 저장 예측 재사용|
|stable_agreement|관측 view의 신뢰도 ≥0.9이고 원본·약·중·강의 교사 argmax가 모두 같을 때 사용|
|random_agreement_count|자기 교사의 신뢰도 ≥0.9 후보 중, 판단 일치 조건과 같은 수를 예측 클래스별로 무작위 선택|

두 조건은 원래 FF++ baseline에서 독립적으로 시작한다. 학습 완료한 UDA를 이어 학습하지 않는다. 학생과 EMA 교사는 각각 별도로 갱신한다. **무작위 대조군의 수 맞춤은 각 run 내부의 교사·배치·예측 클래스 기준이다. 독립적인 두 교사가 달라지므로 두 run 전체의 채택 수까지 정확히 같지는 않다.** 후보·일치·실사용 수를 매 배치 CSV와 epoch history에 기록한다. 이는 채택률을 맞춘 무작위 선택에 대한 비교이며 완벽한 cross-run 학습량 통제는 아니다.

- FF++ source 6,000장/epoch(실제3,000·가짜기법별750), 원본+무작위 스무딩 정답 CE 유지. CDF 고정 pool5,000장에서 정답 없이 영상균등6,000노출/epoch, 원본/약/중/강 각25% 관측. 교사의 임시 정답은 관측 view에서 만들고, 학생은 같은 관측 view의 반대 좌우반전을 본다.
- 교사는 한 이미지의 네 강도를 같은 좌우반전으로 보고 판단 일치만 검사한다. 신뢰도0.9는 관측 view에만 적용하며, 네 view 전부의 신뢰도0.9를 요구하지 않는다. 스무딩으로 확률이 이동하더라도 진짜/가짜 판단이 같으면 통과할 수 있다.
- 현재처럼 **스무딩 전 학습용 CDF crop을 보유한 상황**을 가정한다. 모델에는 스무딩 강도나 정답을 전달하지 않는다. ROI fallback은 네 view 모두 동일 원본이며, 따라서 안정성 검사만으로 추가 제외하지 않는다. 테스트에는 원본 짝·강도 정보·선별기가 필요 없고 학생 한 번만 추론한다.
- 타깃 손실: 관측 view 교사의 hard 정답에 대한 CE × 선택 mask. **선택된 합을 전체 타깃 배치 크기로 나눈다.** 가중치0.5, 2epoch ramp, EMA0.999. 선택되지 않은 노출은 타깃 손실0이며 dataset을 줄이지 않는다. 일관된 오답은 통과할 수 있으므로 일치가 정확성의 증거는 아니다.
- 인코더 및 원래 classifier fine-tuning 범위·전처리·native ROI·원본 데이터와 split은 기존과 같다. Xception 전체, CLIP/DINO 마지막2블록 등 기존 범위. LR1e-5/AdamW/BN 통계고정. 최대10epoch/patience3, **FF++ validation1,200장×4조건 평균 BER 최소**로 학생 best.pt를 선택한다. epoch0 후보, AUC guard 없음. CDF test로 모델이나 임계값을 선택하지 않는다.
- 교사 네 view를 보므로 기존 UDA보다 연산량이 증가한다. 학생은 타깃 한 view만 본다. native bilateral은 한 이미지당 한 번만 계산/읽고 네 강도에 재사용한다. 기존 source·target·validation 캐시를 재사용한다. teacher-batch-size32로 교사 추론을 나누며 학습/평가는 직접 진행률 표시한다. 처음 캐시 생성 시간은 남는다.
- 공식 CDF test8,288장×4조건, FP32 frame FPR/FNR/BER/raw-margin AUC. 핵심 목표는 고정 임계값0.5에서 FPR·FNR 동시 감소와 AUC 유지다. 평균과 조건별 결과를 함께 보여준다. 수치상 AUC 유지 flag는 통계적 동등성 검정이 아니다.
- 기존 코드·checkpoint·결과를 보존하고 새 ffpp_c23_smoothing_stable_uda_v1 폴더에 저장한다. 완료 run은 skip, 중단 시 마지막 완료 epoch의 학생·EMA·optimizer를 재개한다. epoch 도중 중단하면 그 epoch를 처음부터 재실행한다.
- 기존 Baseline·Matched Aug·Hard UDA 평가가 없거나 프로토콜이 다르면 새 평가 추론 전에 중단한다. test는 반복 확인한 개발 평가이며, 인물 중복 허용/같은 test 영상 적응 제외 정책과 filename 기반 parent 제외의 한계는 기존과 같다. 새 TAR나 ZIP은 필요 없다.

## Cell 1 — 설치·코드 갱신

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_stable_uda.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['xception']  # 이후 ['clip_b16'] 또는 ['dinov2_b14']로 변경 가능
STRATEGIES=['stable_agreement','random_agreement_count']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_stable_uda_v1'
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
COMMON=f'--models {MODEL_ARGS} --strategies {STRATEGY_ARGS} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --output-root "{EVALUATION_ROOT}" --validation-cache "{VALIDATION_CACHE}" --train-cache "{TRAIN_CACHE}" --target-train-cache "{TARGET_TRAIN_CACHE}" --cache-root "{CDF_CACHE}" --evaluation-pixel-cache "{CDF_PIXEL_CACHE}" --hard-manifest-root "{HARD_MANIFEST_ROOT}" --hard-evaluations "{HARD_EVALUATIONS}" --teacher-batch-size 32 --workers 2'
print('Models:',MODELS,'| New strategies:',STRATEGIES,flush=True)
print('Runs:',RUNS_ROOT,flush=True)
```

## Cell 3 — FF++·초기 baseline 준비

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

## Cell 4 — CDF 기존 TAR 준비

```python
archives={
    'celebdf_v2_roi16_test_v1.tar':557015040,
    'celebdf_v2_roi16_non_test_real_v1.tar':826142720,
    'celebdf_v2_roi16_non_test_fake_v1.tar':5679564800,
}
for name,size in archives.items():
    manifest=TARGET_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv'
    markers=[Path('/content')/f'.{p}_{name}.done' for p in ['adapt','sda','sda_balance','mixed_uda','consistency_c','response','pivot','ranking','mean_teacher','stable_uda']]
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
    (Path('/content')/f'.stable_uda_{name}.done').touch()
    local.unlink()
```

## Cell 5 — 판단 일치·무작위 제외 학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_stable_uda.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --pseudo-threshold 0.9 --ramp-epochs 2 --ema-decay 0.999
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — 두 조건 평가·기존 결과 병합

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_stable_uda.py" evaluate {COMMON} --batch-size 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과·채택 수 표시

```python
import csv
from IPython.display import Markdown,display
labels={'baseline':'Baseline','augmentation':'Matched Aug','mean_teacher':'기존 UDA','stable_agreement':'판단 일치 UDA','random_agreement_count':'무작위 제외 UDA'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison_with_reference.csv').open()))
means=list(csv.DictReader((EVALUATION_ROOT/'condition_means_with_reference.csv').open()))
deltas=list(csv.DictReader((EVALUATION_ROOT/'deltas_vs_reference.csv').open()))
for model in MODELS:
    table=[f'### {model} — 원본·약·중·강 평균','','|방법|FPR ↓|FNR ↓|BER ↓|AUC ↑|','|---|---:|---:|---:|---:|']
    for r in means:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    table=['### 조건별 결과','','|방법|조건|FPR|FNR|BER|AUC|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{conditions[r['condition']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    table=['### 평균 오류 동시 감소 여부','','|방법|비교 기준|FPR·FNR 모두 감소|AUC 유지·상승|','|---|---|---|---|']
    for r in deltas:
        if r['model']==model and r['level']=='mean':
            table.append(f"|{labels[r['strategy']]}|{labels[r['reference']]}|{r['both_errors_lower']}|{r['auc_maintained_numeric']}|")
    display(Markdown('\n'.join(table)))
    for strategy in STRATEGIES:
        run=RUNS_ROOT/f'{model}_{strategy}_{EXPERIMENT}_seed42'
        done=json.loads((run/'completed.json').read_text())
        history=json.loads((run/'history.json').read_text())
        print(model,labels[strategy],'selected epoch:',done['selected_epoch'],'/ trained:',done['epochs_completed'])
        table=['','|Epoch|신뢰도 통과 실제/가짜|판단 일치 실제/가짜|실제 사용 실제/가짜|','|---|---:|---:|---:|']
        for h in history:
            g=h['selection_diagnostics']
            table.append(f"|{h['epoch']}|{g['candidate_real']}/{g['candidate_fake']}|{g['agreement_real']}/{g['agreement_fake']}|{g['selected_real']}/{g['selected_fake']}|")
        display(Markdown('\n'.join(table)))
print('채택 수의 실제/가짜는 교사의 예측입니다. CDF 정답을 학습에 사용하지 않습니다.')
print('무작위 대조군은 각 run 자신의 교사·배치·예측 클래스별 판단 일치 수를 맞춥니다. 두 run의 전체 채택 수가 같다는 뜻은 아닙니다.')
print('CDF 고정 임계값 0.5 / 학생 best.pt / 선택 기준: FF++ 4조건 평균 BER')
print('결과:',EVALUATION_ROOT)
```

