# DINO_CE_Rank_TS_SDA_Seed42

기존 **CE+Rank / CE+Rank+TS / CE+Rank+CDF 실제 정답**을 같은 기준에서 비교한다.
기본 DINOv2 B/14. Cell 2의 MODELS만 바꾸면 Xception·CLIP도 지원한다.
새 코드를 GitHub에 push한 뒤 Cell 1부터 실행한다.

|비교 조건|학습 내용|이번 실행|
|---|---|---|
|CE+Rank|FF++ 실제 정답 CE + 진짜/가짜 순위 손실|기존 완료 모델·저장 예측 재사용|
|CE+Rank+TS|같은 source 손실 + 비라벨 CDF의 EMA 교사 임시 정답|기존 FF++ baseline에서 새 추가학습|
|CE+Rank+SDA|같은 source 손실 + 같은 CDF의 실제 정답|기존 FF++ baseline에서 새 추가학습; 진단용|

- 원래 baseline의 인코더 허용 범위와 분류기를 함께 학습한다. 기존 CE+Rank 모델을 이어 학습하지 않는다.
- FF++ epoch당 6,000장(진짜3,000/가짜4기법 각750), 매 배치 진짜:가짜 1:1. 원본+랜덤 약/중/강, 같은 flip.
- source 손실은 CE + 0.25 × mean softplus(real margin − fake margin). 기존 CE+Rank의 배치·손실·학습률을 재사용한다.
- CDF는 기존 비테스트 고정 5,000장. 영상균등 복원추출 6,000회/epoch, 원본/약/중/강 각25%, 반대 flip의 학생 입력.
  두 새 조건은 같은 이미지·변형 계획을 사용한다. 실제 정답으로 5:5 샘플링하거나 기존 pool을 변경하지 않는다.
- TS: EMA0.999 교사가 확신0.9 이상인 hard 임시 정답만 사용한다. SDA는 같은 노출 이미지의 실제 정답을 모두 사용한다.
  둘 다 target CE를 전체 target 배치로 나누고, 최대가중치0.5까지 첫2epoch 선형 증가한다.
  **SDA는 정답 품질뿐 아니라 사용되는 정답의 양도 다르다. UDA 결과나 엄밀한 성능 상한으로 해석하지 않는다.**
- FF++ validation 1,200장×원본/약/중/강의 평균 raw-margin AUC 최대. AUC가 정확히 같으면 BER 최소.
  epoch0도 후보, 최대10epoch/patience3/min_delta0.0001. CDF 정답·테스트는 모델 선택에 쓰지 않는다.
- 기존 CE+Rank가 같은 데이터·학습 설정인지 확인한다. 기존 실험과 하이퍼파라미터를 다르게 지정하면 비교를 중단한다.
- native ROI·스무딩·resize/normalize·split·BN통계고정·LR1e-5 유지. 기존 픽셀 캐시를 재사용한다.
- CDF 공식test 8,288장×4조건, frame FPR/FNR/BER/AUC까지 포함. CE+Rank 저장 예측을 재사용하고 새 두 조건만 평가한다.
  기존 baseline도 맞는 저장 예측이 있으면 재사용한다. 추론은 학생 모델 한 번이다.
- 진행률은 직접 `!python -u`로 표시한다. 완료 skip/마지막 완료epoch 재개를 지원한다.
- 기존 결과를 덮어쓰지 않는다. 출력은 `ffpp_c23_smoothing_rank_adaptation_v1`에 저장한다.
  공식test는 반복 확인한 개발 평가이며, 기존 비테스트 영상/filename-parent 제외의 한계는 유지된다.

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_rank_adaptation.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['dinov2_b14']  # ['clip_b16'] 또는 ['xception']; 모델별 코랩으로 나누어 실행 가능
STRATEGIES=['rank_ts','rank_sda']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_rank_adaptation_v1'
GROUP='_'.join(MODELS)
RUNS_ROOT=ROOT/'runs'/EXPERIMENT
MANIFEST_ROOT=ROOT/'manifests'/EXPERIMENT/'seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations'/EXPERIMENT/'seed42'/GROUP
HARD_MANIFEST_ROOT=ROOT/'manifests/ffpp_c23_smoothing_mean_teacher_v1/seed42'
CE_RANK_RUNS=ROOT/'runs/ffpp_c23_smoothing_delta_transfer_v1'
CE_RANK_EVALUATIONS=ROOT/'evaluations/ffpp_c23_smoothing_delta_transfer_v1/seed42'
VALIDATION_CACHE=Path('/content/smoothing_mean_teacher_source_validation_v1')
TRAIN_CACHE=Path('/content/smoothing_pivot_bilateral_cache_v1')
TARGET_TRAIN_CACHE=Path('/content/smoothing_mean_teacher_target_cache_v1')
CDF_CACHE=Path('/content/celebdf_official_smoothing_cache_v1')
CDF_PIXEL_CACHE=Path('/content/smoothing_ranking_cdf_bilateral_cache_v1')
MODEL_ARGS=' '.join(MODELS)
STRATEGY_ARGS=' '.join(STRATEGIES)
COMMON=f'--models {MODEL_ARGS} --strategies {STRATEGY_ARGS} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --output-root "{EVALUATION_ROOT}" --validation-cache "{VALIDATION_CACHE}" --train-cache "{TRAIN_CACHE}" --target-train-cache "{TARGET_TRAIN_CACHE}" --cache-root "{CDF_CACHE}" --evaluation-pixel-cache "{CDF_PIXEL_CACHE}" --hard-manifest-root "{HARD_MANIFEST_ROOT}" --ce-rank-runs-root "{CE_RANK_RUNS}" --ce-rank-evaluations "{CE_RANK_EVALUATIONS}" --workers 2'
print('Models:',MODELS,'| 새 추가학습:',STRATEGIES,flush=True)
print('기존 CE+Rank 재사용 / 최대10epoch·patience3 / FF++ 4조건 평균 AUC 선택',flush=True)
print('Results:',EVALUATION_ROOT,flush=True)
```

## Cell 3 — FF++·초기 baseline 준비

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{FF_DATA}" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'

for model in MODELS:
    name=f'{model}_ffpp_c23_roi16_baseline_v1_seed42'
    source,dest=BASELINE_DRIVE_RUNS/name,BASELINE_RUNS/name
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
for folder in [HARD_MANIFEST_ROOT,CE_RANK_RUNS,CE_RANK_EVALUATIONS]:
    assert folder.is_dir(),f'기존 결과 경로 확인: {folder}'
```

## Cell 4 — 기존 CDF TAR·비교 조건 준비

```python
archives={
    'celebdf_v2_roi16_test_v1.tar':557015040,
    'celebdf_v2_roi16_non_test_real_v1.tar':826142720,
    'celebdf_v2_roi16_non_test_fake_v1.tar':5679564800,
}
for name,size in archives.items():
    manifest=TARGET_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv'
    markers=[Path('/content')/f'.{p}_{name}.done' for p in ['adapt','sda','sda_balance','mixed_uda','consistency_c','response','pivot','ranking','mean_teacher','delta_transfer','rank_adaptation']]
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
    (Path('/content')/f'.rank_adaptation_{name}.done').touch()
    local.unlink()

!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_rank_adaptation.py" prepare {COMMON}
assert _exit_code==0,'비교 조건 확인 실패: 위 오류 확인'
```

## Cell 5 — 두 조건 새 추가학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_rank_adaptation.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --ranking-weight 0.25 --pseudo-threshold 0.9 --target-weight 0.5 --ramp-epochs 2 --ema-decay 0.999
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — 테스트·기존 CE+Rank 비교

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_rank_adaptation.py" evaluate {COMMON} --eval-batch-size 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

학습 batch16과 별도로 평가 batch32를 사용한다. FP32 frame 평가이다.

## Cell 7 — 결과 표시

```python
import csv
from IPython.display import Markdown,display
labels={'baseline':'기존 Baseline','ce_rank':'CE+Rank (기존)','rank_ts':'CE+Rank+TS','rank_sda':'CE+Rank+SDA (진단)'}
order=['baseline','ce_rank','rank_ts','rank_sda']
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison_with_ce_rank.csv').open()))
means=list(csv.DictReader((EVALUATION_ROOT/'condition_means_with_ce_rank.csv').open()))
changes=list(csv.DictReader((EVALUATION_ROOT/'smoothing_changes.csv').open()))
for model in MODELS:
    table=[f'### {model} — 원본·약·중·강 평균','','|방법|FPR ↓|FNR ↓|BER ↓|AUC ↑|','|---|---:|---:|---:|---:|']
    for strategy in order:
        for r in means:
            if r['model']==model and r['strategy']==strategy:
                table.append(f"|{labels[strategy]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    table=['### 조건별 결과','','|방법|조건|FPR|FNR|BER|AUC|','|---|---|---:|---:|---:|---:|']
    for strategy in order:
        for r in rows:
            if r['model']==model and r['strategy']==strategy:
                table.append(f"|{labels[strategy]}|{conditions[r['condition']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    table=['### 원본 → 강한 스무딩 변화','','|방법|FPR 변화|FNR 변화|AUC 변화|','|---|---:|---:|---:|']
    for r in changes:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{100*float(r['strong_minus_normal_real_fpr']):+.2f}%p|{100*float(r['strong_minus_normal_fake_fnr']):+.2f}%p|{float(r['strong_minus_normal_auc']):+.4f}|")
    display(Markdown('\n'.join(table)))
    for strategy in STRATEGIES:
        done=json.loads((RUNS_ROOT/f'{model}_{strategy}_{EXPERIMENT}_seed42'/'completed.json').read_text())
        print(labels[strategy],': selected epoch',done['selected_epoch'],'/ completed',done['epochs_completed'])
print('SDA는 실제 CDF 정답 사용 진단입니다. TS와 정답 사용량도 다르므로 차이를 정답 정확성 하나로 단정하지 않습니다.')
print('Results:',EVALUATION_ROOT)
```
