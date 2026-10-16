# Smoothing_Conditional_Pivot_Seed42

FF++ c23 Baseline에서 각각 시작하여 **동일 검증 기준의 Aug**와 **Aug + 클래스별 스무딩 의존 억제**를 학습하고 Celeb-DF 공식 test까지 평가한다. GitHub에 새 코드를 push한 뒤 사용한다. 새 ZIP/TAR 업로드는 필요 없다.

## 실험 조건

|방법|학습 목표|
|---|---|
|Aug · BER selection|원본 + 무작위 약·중·강 스무딩을 Real/Fake 정답으로 학습|
|Conditional pivot|같은 정답 학습 + 진짜·가짜 각각의 점수에서 스무딩 강도를 구분하기 어렵도록 학습|

- 기본 실행 모델은 **CLIP ViT-B/16**. 실행 순서만 CLIP 먼저이며 Cell 2의 MODELS를 바꾸면 Xception·DINOv2도 지원한다. 같은 모델·방법을 두 런타임에서 동시에 실행하지 않는다.
- 두 조건 모두 기존 FF++ Baseline의 **best.pt에서 독립적으로 시작**한다. B/C/Response에서 이어 학습하지 않는다. 기존 Aug는 clean AUC로 선택했으므로 같은 4조건 BER 선택의 새 Aug를 비교군으로 학습한다.
- FF++ source 6,000장/epoch: Real3,000 + Deepfakes·Face2Face·FaceSwap·NeuralTextures 각750. 같은 영상균형 샘플링, seed42+epoch×100003의 이미지 목록과 순서, 대응 view의 동일 좌우반전. 원본+스무딩 두 view로 epoch당12,000개 입력을 처리한다.
- 스무딩은 기존 native ROI의 masked bilateral blend25/50/75%, resize 전에 적용한다. 원래 마스크/경계 feathering·모델별 resize·정규화·fallback은 그대로다. 추가 JPEG 없음. Instagram 공식 필터 재현으로 주장하지 않는다.
- **백본도 업데이트한다.** Xception 전체, CLIP 마지막2블록·ln_post·visual projection, DINO 마지막2블록·최종 LayerNorm과 각 분류기. 기존 Aug의 학습 범위와 같다. 모든 BatchNorm running statistics는 고정한다.
- 보조 판별기: 진짜용/가짜용 각각 margin/10 → Linear32 → GELU → Linear4. 학습 때만 실제 FF++ 정답으로 판별기를 고르고 원본·약·중·강을 예측한다. 테스트 때는 정답·강도·보조 판별기가 필요 없으며 기존 탐지기1회 forward만 사용한다.
- 탐지기 CE는 두 view의 평균. 판별기는 강도를 맞히도록, 탐지기는 gradient reversal로 강도 의존을 줄이도록 학습한다. 최대 reversal 강도0.1, 처음2epoch에 선형 증가. 보조 판별기 손실 자체에는0.1을 곱하지 않아 보조 판별기는 충분히 학습하도록 한다. critic LR1e-3, 탐지기 encoder/head LR각1e-5, weight decay1e-4, batch16.
- 원본은50%, 약·중·강은각16.7%로 들어오므로 강도 손실에 고정 가중치0.5/1.5/1.5/1.5를 사용하여 기대 강도 비중을 맞춘다. **fallback은 진위 CE에 포함하되 강도 손실에서 제외**한다. 보조 정확도는 훈련 중 참고값이며 불변성이 달성됐다는 증거로 해석하지 않는다.
- **Validation:** 기존 고정 FF++1,200장×4조건. threshold0.5의 조건별 BER 평균 최소로 best 선택. epoch0도 후보, 최대10epoch/patience3, patience 최소 개선0.0001. 원본 FPR/FNR/AUC guard 없음. epoch0이 선택되면 Baseline과 동일한 결과이며 임의로 last로 대체하지 않는다.
- 검증용 native 스무딩 PNG는 한 번 만들어 두 조건·epoch에서 재사용한다. 모델 특징은 학습되므로 매번 추론한다. 전체 이미지 내용 해시를 전수 검사하지 않는다. 학습·검증·평가 진행률과 epoch 요약을 표시한다.
- CDF는 학습·모델 선택에 쓰지 않는다. **공식 test TAR만 필요**, non-test TAR 불필요. 평가:8,288장/518영상×4조건, FP32, frame FPR·FNR·BER·AUC. raw margin도 저장한다. 원본 성능 저하와 FPR/FNR 교환은 함께 확인한다.
- 같은 저장 예측으로 조건별 평균 AUC·25%씩 혼합한 AUC·4×4 교차 조건 AUC도 저장한다. 혼합 view를 독립 표본으로 해석하지 않으며, 혼합 BER은 네 조건 BER 평균과 같다.
- Baseline은 이전 동일 평가 JSON을 읽는다. 새 Aug와 Conditional pivot만 추론한다. 완료 run/조건은 재사용하고 중단 학습은 마지막 완료 epoch에서 재개한다. 기존 데이터·모델·결과를 변경하지 않는다.
- 현재 CDF test는 반복 확인한 개발 평가다. 이 구현의 GPU 실험 성능이나 새 학습법의 독창성이 확보됐다는 뜻은 아니다. 클래스별 nuisance 억제는 기존 pivotal classifier 계열 발상을 스무딩 문제에 적용한 비교 실험이다.

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_pivot.py').is_file(),'새 코드를 먼저 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['clip_b16']  # 'xception', 'clip_b16', 'dinov2_b14'
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_pivot_v1'
GROUP='_'.join(MODELS)
RUNS_ROOT=ROOT/'runs'/EXPERIMENT
MANIFEST_ROOT=ROOT/'manifests'/EXPERIMENT/'seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations'/EXPERIMENT/'seed42'/GROUP
VAL_CACHE=Path('/content/smoothing_pivot_validation_v1')
CACHE_ROOT=Path('/content/celebdf_official_smoothing_cache_v1')
BASELINE_EVALUATION=ROOT/'evaluations/ffpp_c23_baselines_celebdf_official_smoothing_v1/seed42'
MODEL_ARGS=' '.join(MODELS)
COMMON=f'--models {MODEL_ARGS} --strategies augmentation conditional_pivot --seed 42 --source-root "{SOURCE_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --validation-cache "{VAL_CACHE}" --workers 2'
print('Models:',MODELS,'| Aug / Conditional pivot',flush=True)
print('Runs:',RUNS_ROOT,flush=True)
```

## Cell 3 — FF++ Train/Val·Baseline 준비

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
```

## Cell 4 — 두 조건 학습

원본+스무딩을 정답으로 학습하는 Aug 대조군부터, 다음으로 클래스별 강도 의존 억제를 학습한다. 검증 스무딩 준비는 최초1회만 필요하다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_pivot.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --critic-lr 1e-3 --adversarial-weight 0.1
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 4 재실행'
```

## Cell 5 — CDF 공식 Test만 준비

```python
name='celebdf_v2_roi16_test_v1.tar'
expected_bytes=557015040
manifest=TARGET_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv'
marker=Path('/content')/f'.pivot_{name}.done'
old_markers=[Path('/content')/f'.{prefix}_{name}.done' for prefix in ['adapt','sda','sda_balance','mixed_uda','consistency_c','response']]
old_markers += [Path('/content/.celebdf_official_roi16_test_v1.done'),Path('/content/.preservation_b_celebdf_official_test.done')]
if manifest.is_file() and any(p.exists() for p in [marker,*old_markers]):
    print('Already extracted:',name,flush=True)
else:
    source=CDF_DATA/name
    assert source.is_file() and source.stat().st_size==expected_bytes,f'TAR 확인: {source}'
    assert shutil.disk_usage('/content').free>2*expected_bytes+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name
    print('Copying:',name,flush=True)
    shutil.copy2(source,local)
    assert local.stat().st_size==expected_bytes,'복사 미완료'
    print('Extracting:',name,flush=True)
    result=subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-5000:])
    assert manifest.is_file(),manifest
    marker.touch()
    local.unlink()
```

## Cell 6 — CDF 원본·스무딩 평가

모델당 새2방법×4조건=8회 평가. 기존 Baseline은 저장 결과를 읽으며 평가에는 보조 판별기·정답·강도 입력이 필요 없다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_pivot.py" evaluate {COMMON} --target-root "{TARGET_ROOT}" --cache-root "{CACHE_ROOT}" --output-root "{EVALUATION_ROOT}" --baseline-evaluation-root "{BASELINE_EVALUATION}" --batch-size 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표·선택 epoch

```python
import csv
from IPython.display import Markdown,display
labels={'baseline':'Baseline','augmentation':'Aug · BER selection','conditional_pivot':'Conditional pivot'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison.csv').open()))
means=list(csv.DictReader((EVALUATION_ROOT/'condition_means.csv').open()))
for model in MODELS:
    table=[f'### {model} — 조건별 결과','','|방법|조건|FPR|FNR|BER|AUC|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{conditions[r['condition']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    table=['### 네 조건 평균','','|방법|FPR|FNR|BER|AUC|','|---|---:|---:|---:|---:|']
    for r in means:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    for strategy in ['augmentation','conditional_pivot']:
        run=RUNS_ROOT/f'{model}_{strategy}_ffpp_c23_smoothing_pivot_v1_seed42'
        done=json.loads((run/'completed.json').read_text())
        print(model,strategy,'best epoch=',done['best_epoch'],'epochs=',done['epochs_completed'])
        if done['best_epoch']==0:
            print('epoch0 Baseline이 선택됨: 해당 방법은 validation에서 개선하지 못함')
mixtures=list(csv.DictReader((EVALUATION_ROOT/'mixture_summary.csv').open()))
table=['### 원본·약·중·강 25% 혼합','','|모델|방법|조건별 평균 AUC|혼합 AUC|혼합 BER|','|---|---|---:|---:|---:|']
for r in mixtures:
    table.append(f"|{r['model']}|{labels[r['strategy']]}|{float(r['condition_mean_auc']):.4f}|{float(r['pooled_auc']):.4f}|{float(r['balanced_error_rate']):.2%}|")
display(Markdown('\n'.join(table)))
print('Results:',EVALUATION_ROOT)
```
