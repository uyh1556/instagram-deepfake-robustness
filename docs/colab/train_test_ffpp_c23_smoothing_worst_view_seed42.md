# Smoothing_Worst_View_TS_Seed42

FF++ 원본과 스무딩 중 **실제 정답 손실이 더 큰 쪽**을 이미지마다 선택하여 학습한다.
기본 Xception·seed42이며 Cell 2의 MODELS를 ['clip_b16'] 또는 ['dinov2_b14']로 바꿀 수 있다.
코드를 GitHub에 push한 뒤 코랩에서 실행한다.

|FF++ 손실|CDF 학습 없음|CDF 일반 Teacher–Student|
|---|---|---|
|원본·스무딩 손실 평균|기존 Matched Aug 재사용|기존 일반 TS 재사용|
|이미지별 두 손실의 최대값|Worst-view Aug 새 학습|Worst-view TS 새 학습|

- 두 새 조건은 원래 FF++ baseline에서 독립 시작한다. 기존 TS를 이어 학습하지 않는다.
- source 손실 = mean_i max(CE(원본_i, 실제정답_i), CE(무작위스무딩_i, 실제정답_i)).
  배치 전체에서 한 장을 고르는 것이 아니라 이미지마다 선택한다. 정확히 같은 손실이면 두 view에 gradient가 반씩 전달된다.
  진짜와 가짜를 같은 수로 표집하는 기존 source 규칙은 유지한다. 실제 정답 손실로 선택하므로 진짜에는 진짜, 가짜에는 가짜 쪽으로 학습한다.
- 이전 점수 차이 Huber/보정기/원본 점수 복원 항목은 없다. 원래 classifier 및 허용된 encoder 범위를 함께 학습한다.
  같은 원본+무작위 약/중/강 두 forward를 재사용한다. 세 스무딩 전체의 최대값을 보장하는 방식은 아니다.
  fallback도 기존 동일 이미지 paired 경로를 유지한다. 학습 중 dropout이 있으면 두 손실이 다를 수 있다.
- 최대값은 평균값보다 크거나 같으며 추가 배율 없이 사용한다. CDF 가중치0.5를 고정했으므로 source/target의 상대적인 기울기 영향까지 같다고 주장하지 않는다.
- Worst-view TS의 CDF는 기존과 동일: 고정 pool5000장, 정답 없이 영상균등6000노출/epoch, 원본/약/중/강 각25%,
  교사·학생 반대 flip, confidence≥0.9 hard label, targetweight0.5/2epoch ramp, EMA0.999.
  Worst-view Aug에는 CDF forward/임시 정답 손실이 없다. 공통 데이터 검증 및 기존 비교를 위해 같은 CDF manifest는 준비한다.
- source6000장/epoch, validation1200장×원본·약·중·강의 평균BER 최소, epoch0 후보, 최대10epoch/patience3,
  LR1e-5/AdamW/BN통계고정. 모델별 전처리·스무딩·학습 범위·정답·split을 유지한다.
- 기존 native source/target bilateral·validation 캐시를 재사용하고 직접 !python -u로 진행률을 표시한다.
  완료 조건은 skip, 중단 시 마지막 완료 epoch의 학생·교사·optimizer부터 재개한다.
- 평가: CDF 공식test8288장×4조건, FP32 frame FPR/FNR/BER/raw-margin AUC.
  기존 Baseline/Matched Aug/일반 TS의 동일-protocol 저장 예측을 확인하여 재사용한다.
  새 조건만 추론하고 평균·조건별·원본→강한 스무딩 변화·각각 기존 Aug/TS 대비 차이를 저장한다.
- 새 ffpp_c23_smoothing_worst_view_v1 경로를 사용한다. 기존 파일/엔진/데이터/이전 결과를 보존하며 새 TAR/ZIP은 필요 없다.
  이 CDF test는 반복 확인한 개발 평가다. 같은 test 영상·프레임 적응 제외와 기존 filename 기반 parent 제외의 한계는 유지한다.
  기존 teacher의 잘못된 CDF 임시 정답까지 해결하는 방법은 아니며 효과는 실험으로 판단한다.

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_worst_view.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['xception']  # ['clip_b16'] 또는 ['dinov2_b14']로 변경 가능
STRATEGIES=['worst_view_aug','worst_view_ts']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_worst_view_v1'
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

## Cell 5 — Worst-view Aug·TS 학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_worst_view.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --pseudo-threshold 0.9 --ramp-epochs 2 --ema-decay 0.999
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — 새 두 조건 평가·기존 결과 비교

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_worst_view.py" evaluate {COMMON} --batch-size 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표시

```python
import csv
from IPython.display import Markdown,display
labels={'baseline':'Baseline','augmentation':'기존 Aug','mean_teacher':'기존 TS','worst_view_aug':'Worst-view Aug','worst_view_ts':'Worst-view TS'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison_with_reference.csv').open()))
means=list(csv.DictReader((EVALUATION_ROOT/'condition_means_with_reference.csv').open()))
changes=list(csv.DictReader((EVALUATION_ROOT/'smoothing_changes.csv').open()))
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
    table=['### 원본 → 강한 스무딩 변화','','|방법|FPR 변화|FNR 변화|AUC 변화|','|---|---:|---:|---:|']
    for r in changes:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{100*float(r['strong_minus_normal_real_fpr']):+.2f}%p|{100*float(r['strong_minus_normal_fake_fnr']):+.2f}%p|{float(r['strong_minus_normal_auc']):+.4f}|")
    display(Markdown('\n'.join(table)))
    for strategy in STRATEGIES:
        run=RUNS_ROOT/f'{model}_{strategy}_{EXPERIMENT}_seed42'
        done=json.loads((run/'completed.json').read_text())
        history=json.loads((run/'history.json').read_text())
        print(labels[strategy],'선택 epoch:',done['selected_epoch'],'/ 학습 epoch:',done['epochs_completed'])
        for r in history:
            harder=r['smoothed_harder_by_class']
            print(f"epoch {r['epoch']}: source worst CE={r['source_ce']:.4f}, clean CE={r['source_clean_ce']:.4f}, smoothing CE={r['source_smoothed_ce']:.4f}, 스무딩이 더 어려운 비율 진짜={harder['real']:.1%}/가짜={harder['fake']:.1%}")
print('기존 Baseline·Aug·TS는 저장 결과를 재사용했습니다.')
print('변화 폭뿐 아니라 실제 FPR·FNR·BER·AUC를 함께 확인하세요.')
print('결과:',EVALUATION_ROOT)
```

