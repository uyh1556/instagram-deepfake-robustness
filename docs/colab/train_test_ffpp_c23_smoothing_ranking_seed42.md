# CLIP_Cross_Quality_Ranking_Seed42

FF++ c23 Baseline의 백본을 고정하고 **같은 Current-only 분류기에서 BCE와 BCE+품질 간 순위 학습**을 비교한다. 학습과 Celeb-DF 공식 test 평가까지 포함한다. 코드를 GitHub에 push한 뒤 실행한다. 새 ZIP/TAR 업로드는 필요 없다.

|조건|학습 손실|
|---|---|
|Baseline|기존 FF++ 탐지기, 추가 학습 없음|
|Current-only / BCE|원본·스무딩을 실제 진짜/가짜 정답으로 분류|
|Quality ranking|같은 BCE + 가장 가짜처럼 보인 진짜의 점수보다 가장 진짜처럼 보인 가짜의 점수가 높도록 학습|

- 기본 CLIP, Cell 2의 MODELS를 바꾸면 Xception/DINOv2도 지원한다. 기존 encoder와 원래 classifier는 고정하고 Current-only와 동일한 hidden32 residual MLP만 학습한다. 추가 스무딩 probe는 만들지 않는다. 최종 출력은 기존 raw margin + MLP 보정이며, epoch0은 Baseline과 같다.
- 기존 FF++ split 유지. epoch당6,000장(Real3,000/Fake 네 기법 각750), 영상 균형·epoch별 회전 표본. 원본+무작위 약/중/강 view, 대응 좌우반전 동일. 스무딩과 모델별 전처리는 기존 그대로이며 fallback은 변형하지 않는다.
- **두 조건의 새로운 대조 비교:** source-image16쌍/배치(진짜8·가짜8), 총32개 view. 원본·스무딩 쌍을 유지한다. 이전 Current-only의 observed-view16 배치와 달라서, 이전 완료 모델을 공정 대조군으로 대신하지 않고 새 BCE 조건을 함께 학습한다. 두 조건의 초기값·표본·배치·검증은 동일하다.
- 후보 손실: `BCE + 0.5 × mean softplus(1 + worst_real/scale - worst_fake/scale)`. 각 이미지의 **원본과 이번에 뽑힌 스무딩 두 view 중에서만** 진짜는 최대, 가짜는 최소 점수를 선택하고 배치 내 모든 진짜–가짜 조합을 비교한다. 네 강도 전체의 최악값을 보장하는 학습은 아니다. scale은 첫 epoch 관측 margin 절댓값의75분위수와1 중 큰 값으로 고정한다. 강도 정보·CDF 정답은 입력하지 않는다. 가중치·gap은 이번 비교에서 고정하며 CDF로 선택하지 않는다.
- 특징 추출 FP32, 기본32장/추론 배치. 두 조건이 매 epoch의 특징을 공유하므로 원본6,000장당12,000 image forward만 필요하다. 강도별 특징은 epoch 목록대로 캐시하며 스무딩 bilateral도 native PNG로 캐시해 반복 계산을 줄인다. CDF의 기존 스무딩 PNG 캐시도 있으면 재사용한다. 첫 특징 추출은 여전히 시간이 걸린다. 임시 특징 캐시는 런타임 로컬, 학습 결과는 Drive에 저장된다.
- 최대10epoch/patience3, AdamW LR1e-3/weight decay1e-4. 기존 FF++ 검증1,200장×4조건의 **평균 BER 최소**로 선택한다. epoch0 후보, 별도 원본/AUC 제한 없음. epoch0이면 last도 진단 행으로 평가하되 정식 best는 대체하지 않는다.
- CDF 학습 없음. 공식test518영상/8,288장×원본·약·중·강 평가. Frame FPR/FNR/BER, threshold0.5. **주 AUC는 raw margin**, 확률 기반 AUC도 별도 저장한다. 확률0/1 포화에 따른 동점을 구분하기 위한 계산이며 AUC의 정의는 동일하다. 같은 frozen forward에서 Baseline도 얻는다. 추론은 이미지1개당 백본1회이며 원본 대응짝/정답/스무딩 강도 입력이 필요 없다.
- 저장: `comparison.csv`, `condition_means.csv`, raw margin 예측 CSV, 조건을25%씩 섞은 AUC 및4×4 교차 조건 AUC. 같은 이미지의 네 view는 독립 표본이 아니다. 현재 CDF test는 반복 확인한 개발 평가이며 이 방법의 개선·독창성을 미리 보장하지 않는다.
- 완료 run과 캐시를 재사용하며 중단 시 마지막 완료 epoch에서 재개한다. 과거 Aug/B/C/Response/Pivot 결과와 원본 데이터는 보존한다.

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_ranking.py').is_file(),'새 코드를 먼저 GitHub에 push하세요.'
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
EXPERIMENT='ffpp_c23_smoothing_ranking_v1'
GROUP='_'.join(MODELS)
RUNS_ROOT=ROOT/'runs'/EXPERIMENT
MANIFEST_ROOT=ROOT/'manifests'/EXPERIMENT/'seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations'/EXPERIMENT/'seed42'/GROUP
FEATURE_CACHE=Path('/content/smoothing_ranking_features_v1')
TRAIN_CACHE=Path('/content/smoothing_pivot_bilateral_cache_v1')
CDF_CACHE=Path('/content/celebdf_official_smoothing_cache_v1')
CDF_PIXEL_CACHE=Path('/content/smoothing_ranking_cdf_bilateral_cache_v1')
MODEL_ARGS=' '.join(MODELS)
COMMON=f'--models {MODEL_ARGS} --strategies current_only quality_ranking --seed 42 --source-root "{SOURCE_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --feature-cache-root "{FEATURE_CACHE}" --train-cache "{TRAIN_CACHE}" --feature-batch-size 32 --workers 2'
print('Models:',MODELS,'| BCE / BCE + Quality ranking',flush=True)
print('저장 경로:',RUNS_ROOT,flush=True)
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
        assert temp.stat().st_size==src.stat().st_size,'모델 복사 미완료'
        temp.replace(dst)
```

## Cell 4 — 두 조건 학습

특징 추출 진행률 다음에 작은 분류기 학습 진행률이 표시된다. 분류기 두 개는 같은 특징을 사용한다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_ranking.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --batch-size 16 --hidden 32 --learning-rate 1e-3 --weight-decay 1e-4 --ranking-weight 0.5 --ranking-gap 1.0
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 4 재실행'
```

## Cell 5 — CDF 공식 Test 준비

```python
name='celebdf_v2_roi16_test_v1.tar'
expected_bytes=557015040
manifest=TARGET_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv'
markers=[Path('/content')/f'.{prefix}_{name}.done' for prefix in ['adapt','sda','sda_balance','mixed_uda','consistency_c','response','pivot','ranking']]
markers += [Path('/content/.celebdf_official_roi16_test_v1.done'),Path('/content/.preservation_b_celebdf_official_test.done')]
if manifest.is_file() and any(p.exists() for p in markers):
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
    (Path('/content')/f'.ranking_{name}.done').touch()
    local.unlink()
```

## Cell 6 — CDF 원본·스무딩 평가

모델당 네 조건의 특징을 한 번씩만 추출하고 Baseline·BCE·Ranking이 공유한다. 학습 정답이나 강도 정보는 테스트 입력으로 사용하지 않는다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_ranking.py" evaluate {COMMON} --target-root "{TARGET_ROOT}" --output-root "{EVALUATION_ROOT}" --cache-root "{CDF_CACHE}" --evaluation-pixel-cache "{CDF_PIXEL_CACHE}"
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표

```python
import csv
from IPython.display import Markdown,display
labels={'baseline':'Baseline','current_only':'Current-only / BCE','quality_ranking':'BCE + Quality ranking'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison.csv').open()))
means=list(csv.DictReader((EVALUATION_ROOT/'condition_means.csv').open()))
for model in MODELS:
    table=[f'### {model} — 조건별 결과','','|방법|조건|FPR|FNR|BER|AUC|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['model']==model:
            table.append(f"|{labels.get(r['strategy'],r['strategy'])}|{conditions[r['condition']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    table=['### 원본·약·중·강 평균','','|방법|FPR|FNR|BER|AUC|','|---|---:|---:|---:|---:|']
    for r in means:
        if r['model']==model:
            table.append(f"|{labels.get(r['strategy'],r['strategy'])}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
    for strategy in ['current_only','quality_ranking']:
        done=json.loads((RUNS_ROOT/f'{model}_{strategy}_{EXPERIMENT}_seed42/completed.json').read_text())
        print(strategy,'selected epoch:',done['selected_epoch'],'/ trained epochs:',done['epochs_completed'])
print('AUC: raw margin 기준. last_diagnostic 행은 정식 best가 아닌 마지막 epoch 진단값입니다.')
print('결과:',EVALUATION_ROOT)
```
