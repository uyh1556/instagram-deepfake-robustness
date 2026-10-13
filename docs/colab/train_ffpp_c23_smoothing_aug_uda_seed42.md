# FFPP_c23_Smoothing_Aug_UDA_Seed42

baseline 스무딩 평가 코랩과 **별도 런타임**에서 실행한다. 모델별 기존 FF++ best.pt에서 시작한다.
기존 baseline run을 변경하거나 다시 학습하지 않는다. 기본은 3모델 × 4방법 = 12개 추가학습 run이다.

## 비교 방법

|방법|학습 내용|
|---|---|
|source_ft|FF++ 원본으로 추가학습: 추가학습 자체의 효과를 구분하는 대조군|
|augmentation|FF++ 원본 + 랜덤 스무딩 25/50/75%, 두 view에 동일한 Real/Fake 정답 적용|
|uda_normal|FF++ 정답 있는 원본 + Celeb-DF 정답 없는 원본을 DANN으로 적응|
|uda_smoothing|FF++ 정답 있는 원본 + Celeb-DF 정답 없는 랜덤 스무딩을 DANN으로 적응|

- 모두 같은 초기 baseline, source 샘플 규칙, 최대 **10epoch / patience 3**. 종료 epoch는 검증 추이에 따라 달라질 수 있다.
- source는 epoch마다 **6,000장(Real 3,000 + Fake 각 기법 750)**. 원래 FF++ Train에서 영상별로 고르게 재선택한다.
- FF++ clean Validation에서 고정 **1,200장(Real 600 + Fake 각 150)**으로 비교한다. 속도를 위한 추가학습용 고정 부분집합이며 기존 baseline 검증 전체 목록은 바꾸지 않는다.
- checkpoint는 추가학습 epoch 중 최고 FF++ 검증 macro4 frame AUC. Celeb-DF 정답/테스트는 선택에 사용하지 않는다.
- UDA target은 **정답을 제거한 5,000장**. 클래스 비율을 맞춰 뽑지 않는다. 고정 target pool에서 영상별 균등하게 반복 샘플링한다. source 6,000장당 target 6,000회 노출이므로 일부 target은 재사용된다.
- 공식 테스트 영상 제외 + 파일명의 두 인물/영상 번호에 따른 원본 후보 연결까지 보수적으로 제외한다. 로컬 목록에서 적응 가능 pool은 **342영상 / 5,472장**이었다. 인물 자체를 분리하는 규칙이 아니며, 공식 생성 관계의 완전한 검증도 아니다.
- target의 자연스러운 클래스 불균형과 보수적 제외로 달라진 데이터 구성이 UDA 결과에 영향을 줄 수 있다. target 정답은 학습/균형화/모델 선택에 쓰지 않는다.
- 스무딩은 이전과 동일한 bilateral·기하 마스크, native ROI에서 모델별 resize 전 적용. fallback 그대로 포함, 추가 JPEG 없음.
- Xception 전체 / CLIP·DINO 마지막 2블록 등 기존 trainable 범위 유지. 인코더·분류기 LR1e-5, AdamW, batch16. warmup 재실행 없음.
- 모든 방법에서 BatchNorm running statistics 고정. UDA domain head는 LayerNorm→Linear128→ReLU→Linear2, domain loss 0.1, 점진적 GRL.
- source_ft/augmentation은 두 source view의 평균 CE, UDA는 source CE + domain CE. batch당 두 view와 source 이미지 수는 같지만 supervised view 수와 target 접근 여부는 다르다.
- BF16 지원 시 BF16, 아니면 FP32. FP16 없음. 비정상 loss/gradient는 batch를 건너뛰지 않고 중단한다. 완료 epoch부터 재개한다.
- **평가:** 같은 공식 test 8,288장 × 원본/스무딩3단계, 이미지 지표만. baseline 결과는 기존 평가 코랩의 파일을 읽고 재추론하지 않는다.

## 필요한 Drive 파일

기존 파일을 재사용하며 새 TAR 업로드는 필요 없다.

- `data/ffpp_c23_4methods_roi16_v1/`: 기법별 Train/Val TAR 5개 + method_inventory.json. FF++ Test TAR는 이 코랩에서 필요 없음.
- `data/celebdf_v2_face_crops16_v1/`: 공식 test TAR + non_test_real TAR + non_test_fake TAR.
- `runs/ffpp_c23_4methods_baseline_v1/`: 완료된 세 모델의 best.pt/config.json/completed.json.

기존 임의 80/10/10 TAR를 사용하지 않는다. 학습은 baseline 평가가 진행 중이어도 시작할 수 있다.
**Cell 6 평가만 baseline 평가 코랩 완료 후 실행한다.**

## Cell 1 — Drive·코드·설치

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_adaptation.py').is_file(),'새 코드 push 필요'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 변수

MODELS를 나누면 모델별로 다른 코랩에서 돌릴 수 있다. 같은 모델·방법의 run을 두 런타임에서 동시에 학습하지 않는다.

```python
MODELS=['xception','clip_b16','dinov2_b14']
STRATEGIES=['source_ft','augmentation','uda_normal','uda_smoothing']
VAL_IMAGES=1200
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
RUNS_ROOT=ROOT/'runs/ffpp_c23_smoothing_adaptation_v1'
GROUP='_'.join(MODELS)
MANIFEST_ROOT=ROOT/'manifests/ffpp_c23_smoothing_adaptation_v1/seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations/ffpp_c23_smoothing_adaptation_v1/seed42'/GROUP
BASELINE_EVALUATION=ROOT/'evaluations/ffpp_c23_baselines_celebdf_official_smoothing_v1/seed42'
CACHE_ROOT=Path('/content/celebdf_official_smoothing_cache_v1')
MODEL_ARGS=' '.join(MODELS)
STRATEGY_ARGS=' '.join(STRATEGIES)
COMMON=f'--models {MODEL_ARGS} --strategies {STRATEGY_ARGS} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --output-root "{EVALUATION_ROOT}" --cache-root "{CACHE_ROOT}" --baseline-evaluation-root "{BASELINE_EVALUATION}" --batch-size 16 --workers 2'
print('Models:',MODELS,'Strategies:',STRATEGIES)
```

## Cell 3 — FF++ Train/Val 데이터

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{FF_DATA}" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'
```

## Cell 4 — Celeb-DF·초기 모델 준비

Drive TAR를 로컬로 복사 후 추출한다. 큰 이미지 전수 해시 검사는 하지 않는다.

```python
archives={
 'celebdf_v2_roi16_test_v1.tar':557015040,
 'celebdf_v2_roi16_non_test_real_v1.tar':826142720,
 'celebdf_v2_roi16_non_test_fake_v1.tar':5679564800,
}
for name,size in archives.items():
    source=CDF_DATA/name
    assert source.is_file(),source
    manifest=TARGET_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv'
    marker=Path('/content')/f'.adapt_{name}.done'
    if marker.exists() and manifest.is_file():
        print('Already extracted:',name,flush=True)
        continue
    assert source.stat().st_size==size,f'업로드 미완료 또는 다른 TAR: {source}'
    assert shutil.disk_usage('/content').free>2*size+2*2**30,'Colab 로컬 디스크 공간 부족'
    local=Path('/content')/name
    print('Copying:',name,flush=True)
    shutil.copy2(source,local)
    assert local.stat().st_size==size,'로컬 복사 크기 불일치'
    print('Extracting:',name,flush=True)
    result=subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        print(result.stderr)
        raise RuntimeError('Celeb-DF 압축 해제 실패')
    assert manifest.is_file(),manifest
    marker.touch()
    local.unlink()
for model in MODELS:
    name=f'{model}_ffpp_c23_roi16_baseline_v1_seed42'
    source=BASELINE_DRIVE_RUNS/name
    dest=BASELINE_RUNS/name
    for filename in ['best.pt','config.json','completed.json']:
        assert (source/filename).is_file(),f'Baseline 학습 완료 확인: {source/filename}'
    dest.mkdir(parents=True,exist_ok=True)
    for filename in ['best.pt','config.json','completed.json']:
        shutil.copy2(source/filename,dest/filename)
    print('Baseline ready:',model,flush=True)
```

## Cell 5 — 추가학습

진행률·ETA와 epoch당 FF++ 검증 요약을 표시한다. 완료 run은 skip, 중단 run은 마지막 완료 epoch부터 재개한다.
기존 모델과 분할은 보존한다. 이 셀까지는 baseline 테스트 완료를 기다리지 않아도 된다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_adaptation.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images {VAL_IMAGES} --target-images 5000 --encoder-lr 1e-5 --head-lr 1e-5 --domain-weight 0.1
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — 공식 Test 평가·baseline 비교

baseline 평가 코랩 완료 후 실행. baseline은 파일을 재사용하고, 추가학습 모델만 원본/스무딩3단계에서 추론한다.
세 모델·네 추가학습 방법을 모두 선택했다면 48조건을 추론하므로 학습과 별개로 평가 시간이 필요하다.

```python
for model in MODELS:
    for condition in ['normal','smooth_weak','smooth_medium','smooth_strong']:
        path=BASELINE_EVALUATION/f'{model}_{condition}.json'
        assert path.is_file(),f'먼저 baseline 평가 코랩 완료: {path}'
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_adaptation.py" evaluate {COMMON}
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표

```python
import csv
from IPython.display import Markdown,display
rows=list(csv.DictReader((EVALUATION_ROOT/'frame_summary.csv').open()))
for model in MODELS:
    table=[f'### {model}', '', '|방법|조건|AUC|Real FPR|Fake FNR|', '|---|---|---:|---:|---:|']
    for r in rows:
        if r['model']==model:
            table.append(f"|{r['strategy']}|{r['condition']}|{float(r['auc']):.4f}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|")
    display(Markdown('\n'.join(table)))
print('Saved:',EVALUATION_ROOT)
```
