# Smoothing_TS_Paired_Score_Seed42

기존 일반 Teacher–Student UDA에 **FF++ 원본–스무딩 점수 차이 손실 하나**를 추가한다.
기본 Xception, Cell 2의 MODELS를 바꾸면 CLIP/DINO도 실행할 수 있다. 코드 push 후 코랩을 연다.

- 새 후보 한 개만 FF++ baseline에서 독립 시작한다. 완료한 TS를 이어 학습하지 않는다.
- 기존 Baseline·Matched Aug·일반 hard TS는 동일 평가 프로토콜의 저장 예측을 재사용한다.
- FF++는 실제 정답 원본+무작위 약/중/강 한 장을 함께 학습한다.
- m = 가짜 로짓 − 진짜 로짓. 추가 손실은 **0.1 × mean[검출성공 × SmoothL1(m스무딩−m원본, 0)]**, beta=1 로짓 단위다.
  두 현재 학생 점수 모두에 gradient를 전달한다. 과거 baseline 점수로 강제 복원하거나 한쪽 점수를 detach하지 않는다.
  전체 source 이미지 쌍 수로 나누며 fallback은 CE만 학습한다. 진짜·가짜에 같은 규칙을 적용한다.
- 가중치0.1은 이번 비교에서 사전 고정한 탐색 설정이며 최적값으로 주장하지 않는다. 제약이 과하면 AUC가 낮아질 수 있어 CE/추가 손실과 전체 지표를 함께 저장한다.
- CDF 고정 pool5,000장·영상균등6,000노출/epoch, 원본/약/중/강 각25%, 교사·학생 반대 flip,
  confidence≥0.9 hard 정답, target weight0.5/2epoch ramp, EMA0.999는 기존 일반 TS와 같다.
  실제 CDF 정답·GT5:5·네 조건 일치 선별·학생 추가 스무딩은 사용하지 않는다.
- source6,000장/epoch, FF++ validation1,200×4조건 평균BER 최소, epoch0 후보, 최대10epoch/patience3.
  원래 classifier와 기존 허용 인코더 범위(Xception 전체, CLIP/DINO 마지막2블록 등)를 학습한다.
  BN 통계 고정, LR1e-5. 추가 source/target forward는 없고 기존 source logits를 재사용한다.
- CDF 공식 test8,288장×4조건 FP32 FPR/FNR/BER/raw-margin AUC, 원본→강한 스무딩 차이 및 기존 TS 대비 차이를 저장한다.
  추론은 학생 best.pt 한 번이다. Test는 여러 번 확인한 개발 평가이며 최종 독립 검증은 별도로 필요하다.
- 완료 skip/마지막 완료 epoch 학생·교사·optimizer 재개, 기존 native pixel/validation 캐시 재사용.
  새 TAR/ZIP은 필요 없다. 기존 원본·분할·엔진·checkpoint는 보존하고 새 pair_v1 경로에 저장한다.
- 학습·평가 직접 실행으로 진행률을 표시한다. 검증을 추가 반복하지 않고 기존 TS와 같은 검증만 수행한다.

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_mean_teacher_pair.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['xception']  # ['clip_b16'] 또는 ['dinov2_b14']로 변경 가능
STRATEGIES=['mean_teacher_pair']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_mean_teacher_pair_v1'
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

## Cell 5 — TS + 원본–스무딩 점수 학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_mean_teacher_pair.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --pseudo-threshold 0.9 --ramp-epochs 2 --ema-decay 0.999 --pair-weight 0.1
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — 새 후보 평가·기존 TS 결과 병합

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_mean_teacher_pair.py" evaluate {COMMON} --batch-size 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표시

```python
import csv
from IPython.display import Markdown,display
labels={'baseline':'Baseline','augmentation':'Matched Aug','mean_teacher':'기존 TS','mean_teacher_pair':'TS + 원본–스무딩 점수 학습'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison_with_hard.csv').open()))
means=list(csv.DictReader((EVALUATION_ROOT/'condition_means_with_hard.csv').open()))
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
    run=RUNS_ROOT/f'{model}_mean_teacher_pair_{EXPERIMENT}_seed42'
    done=json.loads((run/'completed.json').read_text())
    history=json.loads((run/'history.json').read_text())
    print('선택 epoch:',done['selected_epoch'],'/ 학습 epoch:',done['epochs_completed'])
    for row in history:
        print(f"epoch {row['epoch']}: source CE={row['source_ce']:.4f}, target CE={row['target_ce']:.4f}, pair Huber={row['source_pair_huber']:.4f}, weighted pair={row['weighted_source_pair']:.4f}")
print('Baseline·Matched Aug·기존 TS는 저장 결과 재사용. 새 후보만 학습·평가.')
print('변화 폭만 작아져도 성공은 아닙니다. 실제 FPR/FNR·BER·AUC도 함께 확인하세요.')
print('결과:',EVALUATION_ROOT)
```

