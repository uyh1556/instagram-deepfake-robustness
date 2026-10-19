# Smoothing_Delta_Transfer_Seed42

비라벨 CDF에서 **스무딩으로 바뀌는 특징**을 측정하고, 정답이 있는 FF++ 특징에 작은 변화로 적용해 진짜·가짜를 구분하도록 학습한다.
기본 CLIP B/16. Cell 2의 MODELS를 바꾸어 Xception/DINO에도 실행한다. 새 코드를 GitHub에 push한 뒤 사용한다.

|새 학습 조건|내용|
|---|---|
|ce_rank|FF++ 원본+무작위 스무딩의 실제 정답 CE + 진짜/가짜 순위 손실|
|source_delta|같은 학습 + FF++에서 측정한 스무딩 특징 변화 전이|
|target_delta|같은 학습 + 비라벨 CDF에서 측정한 스무딩 특징 변화 전이|

- 세 조건은 각각 기존 FF++ baseline에서 독립 시작한다. 기존 TS나 Worst-view를 이어 학습하지 않는다.
- source: 기존 고정 split에서 epoch당 6,000장(진짜3,000/가짜 각750), 매 배치 1:1. 원본+무작위 약/중/강, 같은 flip.
  배치 균형은 순위 쌍 확보를 위한 변경이며 세 새 조건에 동일하다. 과거 TS의 배치 순서와 동일하다고 주장하지 않는다.
- donor: 원본/약/중/강 중 하나를 관측 이미지로 정하고 추가 약한 스무딩을 적용한다. 같은 flip을 적용한 두 이미지의 EMA 인코더 특징 차이를 사용한다.
  source donor는 같은 source epoch의 섞인 이미지, target donor는 기존 CDF 고정5,000장에서 영상균등6,000회 표집한다.
  CDF 정답·추정 진위·confidence는 사용하지 않으며 5:5 GT sampling도 없다. 얼굴 검출 실패 donor는 변화0이다.
- EMA0.999는 특징 측정용 인코더를 천천히 갱신한다. 분류 pseudo-label teacher가 아니다. donor 특징 차이는 역전파하지 않는다.
- 특징 변화는 확대하지 않고, recipient 특징 norm의 최대5%로 제한한다. 첫2epoch에 cap을 선형 증가시킨다.
  양의 변화만 원본·스무딩 source 특징에 더하며, 변형된 특징도 같은 FF++ 정답으로 분류한다.
  특징 이동 뒤에도 정답이 보존된다는 것은 **실험 가정**이다. 변화에 조작 단서가 섞일 수 있으며 새롭거나 우수한 방법으로 확정된 것은 아니다.
- 손실: 실제 정답 CE + 0.25 × mean softplus(진짜 margin − 가짜 margin).
  margin은 fake logit − real logit. 배치의 모든 진짜/가짜·view 조합을 비교한다. AUC surrogate이며 CDF AUC 상승을 보장하지 않는다.
  ce_rank는2view, 전이조건은기존2view+전이2feature view이다. 변화0이면 중복view의 평균이므로 같은 목적함수다.
- **선택 기준 변경:** 세 새 조건 모두 FF++ validation1,200장×4조건 평균 raw-margin AUC 최대. 완전히 같은 AUC이면 BER가 낮은 것.
  epoch0도 후보, 최대10/patience3, 유의한 개선 기준0.0001. BER/FPR/FNR/AUC를 모두 기록하고 별도 guard는 없다.
  target/test는 선택에 사용하지 않는다. 기존 Aug·TS는 BER 선택 결과이므로 참고 기준이며 손실만의 공정한 ablation은 새 세 조건끼리다.
- 원래 분류기와 기존 인코더 학습범위(Xception전체/CLIP·DINO마지막2블록 등)를 함께 업데이트한다.
  LR1e-5/AdamW/BN통계고정, native ROI·모델별resize/normalize·스무딩·split은 기존대로다.
- 전이조건은 source2view 외에 donor2view EMA FP32 추론이 추가되어 ce_rank보다 계산량이 크다.
  추가약한 스무딩은 observed 이미지마다 캐시한다. 첫 생성에는 시간이 걸리지만 진행률을 표시하고 다음epoch/조건은 재사용한다.
- 평가 포함: CDF 공식test8,288장×원본/약/중/강, FP32 frame FPR/FNR/BER/raw-margin AUC.
  기존 Baseline·Aug·일반 TS 저장 예측을 재사용하고 새3조건만 추론한다. 추론은 원래 모델1회이며 donor/EMA/보정짝이 필요 없다.
- 완료skip/완료epoch student·EMA·optimizer 재개, 같은출력폴더의 완료된 다른 새조건도 평가표에 포함한다.
  기존3개CDF TAR·FF++기법별trainval TAR를 재사용한다. ZIP/새TAR/장시간 전수픽셀검증 없음.
- 출력: comparison_with_reference.csv, condition_means_with_reference.csv, deltas_vs_controls.csv, smoothing_changes.csv.
  target_delta가 source_delta보다 좋아야 타깃 정보의 추가기여 근거가 생긴다. 효과는 여러seed 확인이 필요하다.
  CDF test는 반복 확인한 개발 평가이며 기존 filename-parent 제외의 한계가 있다. 공식test 영상/프레임은 적응에 제외한다.

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_delta_transfer.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['clip_b16']  # ['xception'] 또는 ['dinov2_b14']; 모델별 코랩으로 나누어 실행 가능
STRATEGIES=['ce_rank','source_delta','target_delta']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_delta_transfer_v1'
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
print('Models:',MODELS,'| Strategies:',STRATEGIES,flush=True)
print('Best: FF++ validation 4조건 평균 AUC 최대 / epoch0 포함',flush=True)
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
assert HARD_MANIFEST_ROOT.is_dir(),f'기존 TS manifest 경로 확인: {HARD_MANIFEST_ROOT}'
assert HARD_EVALUATIONS.is_dir(),f'기존 TS 평가 경로 확인: {HARD_EVALUATIONS}'
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
    markers=[Path('/content')/f'.{p}_{name}.done' for p in ['adapt','sda','sda_balance','mixed_uda','consistency_c','response','pivot','ranking','mean_teacher','delta_transfer']]
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
    (Path('/content')/f'.delta_transfer_{name}.done').touch()
    local.unlink()
```

## Cell 5 — 새 세 조건 학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_delta_transfer.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --ranking-weight 0.25 --delta-cap-ratio 0.05 --ramp-epochs 2 --ema-decay 0.999
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — 평가·기존 결과 비교

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_delta_transfer.py" evaluate {COMMON} --batch-size 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표시

```python
import csv
from IPython.display import Markdown,display
labels={'baseline':'Baseline','augmentation':'기존 Aug (BER 선택)','mean_teacher':'기존 TS (BER 선택)','ce_rank':'분류+순위','source_delta':'FF++ 변화 전이','target_delta':'CDF 변화 전이'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison_with_reference.csv').open()))
means=list(csv.DictReader((EVALUATION_ROOT/'condition_means_with_reference.csv').open()))
changes=list(csv.DictReader((EVALUATION_ROOT/'smoothing_changes.csv').open()))
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
    table=['### 원본 → 강한 스무딩 변화','','|방법|FPR 변화|FNR 변화|AUC 변화|','|---|---:|---:|---:|']
    for r in changes:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{100*float(r['strong_minus_normal_real_fpr']):+.2f}%p|{100*float(r['strong_minus_normal_fake_fnr']):+.2f}%p|{float(r['strong_minus_normal_auc']):+.4f}|")
    display(Markdown('\n'.join(table)))
    for strategy in STRATEGIES:
        run=RUNS_ROOT/f'{model}_{strategy}_{EXPERIMENT}_seed42'
        done=json.loads((run/'completed.json').read_text())
        print(labels[strategy],'선택 epoch:',done['selected_epoch'],'/ 학습 epoch:',done['epochs_completed'],
              '| FF++ val AUC:',round(done['best_mean_auc'],4),'BER:',f"{done['selected_mean_ber']:.2%}")
print('새 세 조건: 같은 AUC 선택 기준. 기존 Aug·TS: BER 선택이므로 참고 비교.')
print('특징 변화가 실제 조작 단서를 보존한다는 가정은 결과로 검증해야 합니다.')
print('결과:',EVALUATION_ROOT)
```

