# CLIP_Smoothing_Response_Learning_Seed42

FF++ c23로 학습한 CLIP ViT-B/16에서 시작하여 **현재 이미지 분류기와 스무딩 반응 분류기를 학습하고 Celeb-DF 공식 test까지 평가**한다. 새 코드를 GitHub에 push한 뒤 Cell 1부터 실행한다. ZIP 업로드와 새 TAR는 필요 없다.

|조건|분류기 입력|
|---|---|
|Baseline|기존 FF++ 탐지기의 판단|
|Current-only|현재 이미지의 frozen 특징과 기존 margin|
|Response|현재 특징·margin + 추가 약한 스무딩 전후 특징 차이·margin 차이|
|기존 Aug|저장된 결과가 있으면 참고 비교에 포함|

- CLIP은 순서상 먼저 실행한다. 스크립트는 Xception·DINOv2도 지원하며, 다른 모델에서도 같은 조건을 적용할 수 있다.
- 두 새 조건 모두 **기존 인코더·기존 분류기 고정**, hidden32 GELU MLP만 학습한다. 기존 margin에 MLP 출력을 더해 최종 판정을 만든다. 마지막 층을0으로 초기화하여 epoch0은 Baseline과 같다.
- 두 조건의 MLP 구조·초기 가중치·학습 표본·배치 순서는 같다. Current-only는 변화량 입력을0으로 가린다. 전체 parameter 수는 같지만 사용 가능한 입력 정보와 유효 자유도는 다르다.
- FF++ 기존 Train/Val 분할 유지. 매 epoch source6,000장: Real3,000 + Fake 네 기법 각750. 영상균형 표본을 seed42+epoch×100003으로 회전 선택한다. 각 이미지의 원본과 무작위 약·중·강 view를 만들고, 각 view에 추가 약한 스무딩0.25를 적용하여 반응을 측정한다. 좌우반전은 대응 view 모두 동일하게 적용한다.
- 특징 캐시는 두 조건이 공유한다. 원본6,000장당 observed12,000개와 probe12,000개, 합계24,000번의 이미지 forward가 필요하다. Current-only가 단독 추론할 때 필요한 forward는1회, Response는2회다. 추가 스무딩 준비 및 특징 추출에 시간이 소요되며 진행률이 표시된다.
- 원래 스무딩은 native ROI의 bilateral blend25/50/75%. 추가 probe는 **현재 관측 이미지에** blend25%를 한 번 더 적용한다. 강도가 단순히 합산되는 것으로 해석하지 않는다. 얼굴 검출 실패 fallback에는 두 변형 모두 적용하지 않는다.
- 전처리는 기존 CLIP의 RGB → bicubic short-side224 → center crop224 → CLIP 정규화. 스무딩은 resize 전에 적용한다. 특징 추출 FP32, 본체 eval 모드.
- 손실은 FF++ 실제 Real/Fake 정답에 대한 BCE만 사용한다. 원본 점수 복원·teacher·일관성 손실·CDF 학습은 없다. 특징 RMS와 margin scale은 첫 학습 표본에서 계산해 고정한다. 반응의 크기가 사라지지 않도록 특징 차이 자체를 이미지별 단위 벡터로 만들지 않는다.
- 최대10epoch, patience3, AdamW LR1e-3, weight decay1e-4, batch16. **기존 FF++ validation1,200장×원본·약·중·강 평균 BER만**으로 best를 선택한다. epoch0도 후보. 원본 FPR/FNR/AUC 제한은 두지 않으며, 성능 변화는 결과 표에서 확인한다. patience는0.0001 이상 개선 여부로 계산한다.
- 평가: 기존 Celeb-DF 공식test518영상/8,288장×4조건. frame FPR·FNR·BER·AUC, threshold0.5. 정답과 스무딩 강도는 분류기 입력으로 제공하지 않는다. 원본 CDF 대응 이미지를 추가 입력으로 주지 않는다. 입력과 그 입력에서 생성한 probe만 사용한다.
- Baseline 점수는 새 분류기 추론에 필요한 같은 frozen forward에서 함께 얻는다. 기존 Aug는 호환되는 저장 결과만 읽는다. 기존 Aug는 encoder 학습 범위·선택 기준이 달라 참고 비교이며, 반응 정보의 효과는 Current-only와 Response로 비교한다.
- best가 epoch0이면 last도 별도 진단 행으로 평가하며 best를 바꾸지 않는다. 완료 run/특징/평가 캐시는 재사용하고, 중단된 학습은 완료 epoch의 last에서 재개한다. 기존 모델·데이터·결과는 변경하지 않는다.
- 현재 CDF test는 반복 확인한 개발 평가다. 독립적인 최종 일반화 평가는 방법 확정 후 별도로 필요하다. 이 스무딩은 Instagram 공식 알고리즘이 아닌 기하 마스크 기반 근사 변형이다.

## Cell 1 — 설치

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_response.py').is_file(),'새 코드를 먼저 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; from PIL import Image; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0)); print("Pillow:",Image.__version__)'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['clip_b16']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_response_v1'
GROUP='_'.join(MODELS)
RUNS_ROOT=ROOT/'runs'/EXPERIMENT
MANIFEST_ROOT=ROOT/'manifests'/EXPERIMENT/'seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations'/EXPERIMENT/'seed42'/GROUP
FEATURE_CACHE=Path('/content/smoothing_response_features_v1')
REFERENCE_EVALUATIONS=ROOT/'evaluations/ffpp_c23_smoothing_adaptation_v1/seed42'
MODEL_ARGS=' '.join(MODELS)
COMMON=f'--models {MODEL_ARGS} --strategies current_only response --seed 42 --source-root "{SOURCE_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --feature-cache-root "{FEATURE_CACHE}" --extra-strength 0.25 --workers 2'
print('Models:',MODELS,'| Current-only / Response',flush=True)
print('저장 경로:',RUNS_ROOT)
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
        src=source/filename
        dst=dest/filename
        assert src.is_file(),src
        if dst.is_file() and dst.stat().st_size==src.stat().st_size:
            print('Already copied:',model,filename,flush=True)
            continue
        print('Copying:',model,filename,flush=True)
        temp=dst.with_suffix(dst.suffix+'.copying')
        shutil.copy2(src,temp)
        assert temp.stat().st_size==src.stat().st_size,'모델 복사 미완료'
        temp.replace(dst)
```

## Cell 4 — 두 조건 학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_response.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --batch-size 16 --hidden 32 --learning-rate 1e-3 --weight-decay 1e-4
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 4 재실행'
```

## Cell 5 — CDF 평가 데이터 준비

```python
name='celebdf_v2_roi16_test_v1.tar'
expected_bytes=557015040
manifest=TARGET_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv'
markers=[Path('/content')/f'.{prefix}_{name}.done' for prefix in ['adapt','sda','sda_balance','mixed_uda','consistency_c','response']]
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
    (Path('/content')/f'.response_{name}.done').touch()
    local.unlink()
```

## Cell 6 — CDF 원본·스무딩 평가

기존 Aug는 평가 추론을 완료한 뒤 별도 집계 명령으로 합친다. 과거 Aug JSON의 `images`·`precision` 필드가 없는 형식도 저장된 예측 CSV와 평가 프로토콜을 확인하여 읽는다. 이미 네 조건 평가를 마친 뒤 `KeyError: 'images'`로 중단했다면 **아래 두 번째 명령만 실행**하면 된다. 학습·평가 코드는 변경하지 않아 기존 체크포인트·특징 캐시·평가 결과의 identity를 유지한다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_response.py" evaluate {COMMON} --target-root "{TARGET_ROOT}" --output-root "{EVALUATION_ROOT}" --batch-size 32 --hidden 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'

!python -u "{PROJECT}/scripts/summarize_smoothing_response.py" --models {MODEL_ARGS} --output-root "{EVALUATION_ROOT}" --reference-evaluations "{REFERENCE_EVALUATIONS}"
assert _exit_code==0,'결과 집계 실패: 위 오류 확인'
```

## Cell 7 — 결과 표

```python
import csv
from IPython.display import Markdown,display
labels={'baseline':'Baseline','augmentation':'기존 Aug','current_only':'Current-only','response':'Response'}
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
    for strategy in ['current_only','response']:
        done=json.loads((RUNS_ROOT/f'{model}_{strategy}_{EXPERIMENT}_seed42/completed.json').read_text())
        print(strategy,'selected epoch:',done['selected_epoch'],'/ trained epochs:',done['epochs_completed'])
print('결과:',EVALUATION_ROOT)
```
