# FFPP_c23_Smoothing_Preservation_B_Seed42

Xception / CLIP ViT-B/16 / DINOv2 ViT-B/14에 **B(원본 판단 보존 + 스무딩 점수 복원)**를 적용한다. 이 파일은 학습부터 CDF 평가·기존 Baseline/Aug 비교까지 포함한다. 먼저 GitHub에 새 코드를 push하고 위에서부터 실행한다.

- 세 모델은 각자의 기존 FF++ c23 best.pt로 시작한다. **인코더·원래 분류기·정규화 통계를 전부 고정**하고, 작은 보정 모듈만 학습한다. 모델별 전처리를 그대로 사용한다.
- 최종 로짓 차이 = 기존 로짓 차이 + 이미지별 보정값. 모듈은 `고정 특징 LayerNorm(affine 없음) + 기존 margin/scale → Linear32 → GELU → Linear1 → tanh`이며 마지막 층은 0으로 초기화한다.
- scale = 학습 첫 epoch 원본 6,000장의 |margin| 75백분위와 1 중 큰 값. 보정 한도는 ±2×scale. 검증·CDF로 scale을 맞추지 않는다.
- 손실 = 원본/스무딩 평균 정답 BCE + **스무딩 보정 margin을 고정 원본 margin에 가깝게 하는 Huber** + **원본 보정 margin을 고정 원본 margin에 가깝게 하는 Huber**. Huber는 scale로 나눈 margin에서 계산하고 두 가중치는 각각 1. 이는 사전 고정한 파일럿 설정이며 최적값이라는 뜻은 아니다.
- FF++ 원본 6,000장/epoch(Real3,000 + Fake4×750), 원본+무작위 약/중/강 두 view. 영상 균형/epoch별 이미지 회전/같은 좌우반전은 기존 방식이다. 새 원본 12,000장을 쓰는 것이 아니다.
- 기존 같은 FF++ validation 1,200장 × 원본·약·중·강으로 검증한다. **4조건 평균 BER 최소**로 선택하며, epoch0(보정 없음)도 후보에 포함한다. clean BER은 epoch0 대비 +1%p 이내, clean FPR/FNR은 각각 +2%p 이내, clean 및 4조건 평균 macro AUC는 −0.005 이내인 후보만 허용한다. 이 제한은 파일럿 전에 고정하며 CDF 결과로 조절하지 않는다.
- 선택 규칙이 이전 Aug의 clean AUC 규칙과 다르므로, 기존 Aug는 참고 비교다. B의 추가 항목 효과는 이후 동일 선택 기준의 A 비교가 필요하다.
- 최대10epoch / patience3 / seed42 / batch16 / 보정 모듈 LR1e-3 / weight decay1e-4. 본체는 FP32 forward만 하며 역전파하지 않는다. 고정 검증 특징은 한 번 계산하고, 학습 특징도 로컬 캐시하여 재실행 시 재사용한다. 캐시가 없어져도 마지막 완료 epoch부터 재개한다.
- **CDF는 학습에 사용하지 않는다. UDA가 아니다.** CDF 학습용 Real/Fake TAR도 불필요하다. 기존 공식 test TAR 1개만 사용한다.
- 평가: 같은 CDF 8,288장(518영상), 원본/약/중/강, threshold0.5, FP32, frame FPR/FNR/BER/AUC. 기존 스무딩과 fallback 정책 유지. 새 모델 총12조건을 평가하고 기존 Baseline·Aug 결과는 읽어서 비교한다.
- 저장: 새 `ffpp_c23_smoothing_preservation_b_v1` 경로. checkpoint에는 작은 보정 모듈만 저장하므로 원래 baseline best.pt도 보관해야 한다. 기존 결과/데이터를 수정하지 않는다. 이미 반복 확인한 CDF test는 개발 평가라는 한계를 유지한다.

## Cell 1 — 설치

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_preservation.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

MODELS만 바꿔 모델별로 나눠 실행해도 된다. 같은 모델의 run을 동시에 실행하지 않는다.

```python
MODELS=['xception','clip_b16','dinov2_b14']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
RUNS_ROOT=ROOT/'runs/ffpp_c23_smoothing_preservation_b_v1'
GROUP='_'.join(MODELS)
MANIFEST_ROOT=ROOT/'manifests/ffpp_c23_smoothing_preservation_b_v1/seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations/ffpp_c23_smoothing_preservation_b_v1/seed42'/GROUP
BASELINE_EVALUATION=ROOT/'evaluations/ffpp_c23_baselines_celebdf_official_smoothing_v1/seed42'
REFERENCE_EVALUATIONS=ROOT/'evaluations/ffpp_c23_smoothing_adaptation_v1/seed42'
OLD_VALIDATION=ROOT/'manifests/ffpp_c23_smoothing_adaptation_v1/seed42/xception_clip_b16/source_validation.csv'
FEATURE_CACHE=Path('/content/preservation_b_frozen_features_v1')
CACHE_ROOT=Path('/content/celebdf_official_smoothing_cache_v1')
MODEL_ARGS=' '.join(MODELS)
COMMON=f'--models {MODEL_ARGS} --seed 42 --source-root "{SOURCE_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --feature-cache-root "{FEATURE_CACHE}" --source-validation "{OLD_VALIDATION}" --workers 2'
print('Models:',MODELS,'| B only | CDF training: NONE',flush=True)
```

## Cell 3 — FF++ Train/Val만 준비

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{FF_DATA}" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'
```

## Cell 4 — 기존 Baseline 복사

```python
for model in MODELS:
    name=f'{model}_ffpp_c23_roi16_baseline_v1_seed42'
    source=BASELINE_DRIVE_RUNS/name
    dest=BASELINE_RUNS/name
    dest.mkdir(parents=True,exist_ok=True)
    for filename in ['best.pt','config.json','completed.json']:
        assert (source/filename).is_file(),f'Baseline 완료/경로 확인: {source/filename}'
        print('Copying:',model,filename,flush=True)
        shutil.copy2(source/filename,dest/filename)
    print('Baseline ready:',model,flush=True)
```

## Cell 5 — B 학습

처음 모델별 validation 특징 4조건을 한 번 계산한다. 이후 검증은 캐시로 빠르게 수행한다. 매 epoch 학습 특징 추출과 보정 모듈 학습에 진행률이 표시된다. 완료 run은 건너뛰며 중단 run은 마지막 완료 epoch부터 재개한다. selected epoch0이면 기준을 만족하는 개선이 없어 보정 없는 모델이 선택된 것이다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_preservation.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --batch-size 16 --learning-rate 1e-3 --pair-weight 1 --clean-weight 1 --bound 2 --clean-ber-tolerance 0.01 --clean-error-tolerance 0.02 --auc-tolerance 0.005
assert _exit_code==0,'B 학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — CDF 공식 Test만 준비

기존 test TAR만 사용한다. CDF non-test TAR는 받지 않는다.

```python
name='celebdf_v2_roi16_test_v1.tar'
source=CDF_DATA/name
manifest=TARGET_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv'
marker=Path('/content/.preservation_b_celebdf_official_test.done')
if manifest.is_file() and (marker.exists() or Path('/content/.celebdf_official_roi16_test_v1.done').exists()):
    print('CDF Test already extracted',flush=True)
else:
    assert source.is_file(),source
    assert source.stat().st_size==557015040,'TAR 업로드 미완료 또는 다른 버전'
    assert shutil.disk_usage('/content').free>2*source.stat().st_size+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name
    print('Copying CDF Test',flush=True)
    shutil.copy2(source,local)
    assert local.stat().st_size==source.stat().st_size,'로컬 복사 미완료'
    print('Extracting CDF Test',flush=True)
    result=subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-5000:])
    assert manifest.is_file(),manifest
    marker.touch()
    local.unlink()
```

## Cell 7 — B 평가 + 기존 Baseline·Aug 비교

원본·스무딩 3단계에 진행률을 표시한다. 이미 끝난 결과는 재사용한다. 기존 Aug 저장 폴더가 발견되면 자동으로 표에 추가한다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_preservation.py" evaluate {COMMON} --target-root "{TARGET_ROOT}" --cache-root "{CACHE_ROOT}" --output-root "{EVALUATION_ROOT}" --baseline-evaluation-root "{BASELINE_EVALUATION}" --reference-evaluations "{REFERENCE_EVALUATIONS}" --batch-size 32
assert _exit_code==0,'B 평가 실패: 위 오류 확인 후 Cell 7 재실행'
```

## Cell 8 — 결과 표

```python
import csv
from IPython.display import Markdown,display
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison.csv').open()))
labels={'baseline':'Baseline','augmentation':'기존 Aug','preservation_b':'B'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
for model in MODELS:
    done=json.loads((RUNS_ROOT/f'{model}_ffpp_c23_smoothing_preservation_b_v1_seed42/completed.json').read_text())
    table=[f'### {model} — B 선택 epoch {done["selected_epoch"]}', '', '|방법|조건|FPR|FNR|BER|AUC|','|---|---|---:|---:|---:|---:|']
    for strategy in ['baseline','augmentation','preservation_b']:
        for r in rows:
            if r['model']==model and r['strategy']==strategy:
                table.append(f"|{labels[strategy]}|{conditions[r['condition']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
print('결과:',EVALUATION_ROOT)
print('4조건 평균:',EVALUATION_ROOT/'condition_means.csv')
```

## Cell 9 — CLIP epoch3 별도 평가 (선택)

코드 push 후 같은 코랩 맨 아래 추가한다. 재학습 없이 CLIP last.pt가 실제 epoch3인지 확인하고 4조건을 평가한다. 기존 best.pt/last.pt와 기존 평가 결과는 보존한다. 별도 `diagnostic_epoch3` 폴더에 저장하며, 테스트로 best를 다시 선택하는 것이 아닌 진단 평가다. 다른 모델의 학습 코랩은 그대로 둔다.

```python
!git -C "{PROJECT}" pull --ff-only
assert _exit_code==0,'코드 갱신 실패'
!python -u "{PROJECT}/scripts/evaluate_preservation_b_epoch.py" evaluate {COMMON} --models clip_b16 --epoch 3 --target-root "{TARGET_ROOT}" --cache-root "{CACHE_ROOT}" --output-root "{EVALUATION_ROOT}" --baseline-evaluation-root "{BASELINE_EVALUATION}" --reference-evaluations "{REFERENCE_EVALUATIONS}" --batch-size 32
assert _exit_code==0,'epoch3 평가 실패: 위 오류 확인'
import csv
from IPython.display import Markdown,display
rows=list(csv.DictReader((EVALUATION_ROOT/'diagnostic_epoch3/comparison.csv').open()))
table=['|방법|조건|FPR|FNR|BER|AUC|','|---|---|---:|---:|---:|---:|']
for r in rows:
    label='B epoch3' if r['strategy']=='preservation_b' else r['strategy']
    table.append(f"|{label}|{r['condition']}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
display(Markdown('\n'.join(table)))
```
