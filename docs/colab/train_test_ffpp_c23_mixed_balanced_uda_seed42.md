# FFPP_c23_Aug_UDA_MixedTarget_Balanced_Seed42

세 FF++ baseline(Xception, CLIP B/16, DINOv2)에서 각각 **UDA, Aug+UDA**를 새로 학습하고 평가한다. 총 6개 run. Aug는 기존 저장 결과를 재사용한다.

|방법|정답 있는 FF++ 입력|정답 없는 CDF 타깃 입력|
|---|---|---|
|기존 Aug 재사용|원본 + 약·중·강 중 하나|사용하지 않음|
|UDA|원본|원본·약·중·강 중 하나|
|Aug+UDA|원본 + 약·중·강 중 하나|원본·약·중·강 중 하나|

- 타깃은 기존 고정 5,000장을 재사용한다. **각 학습 배치의 Real/Fake 노출을 5:5**로 맞춘다. 영상은 각 클래스 내부에서 균등 선택하고 프레임을 반복 선택한다. 독립 이미지 2,500장씩을 새로 확보하는 실험은 아니다.
- CDF의 비테스트 정답은 **샘플링에만 사용**한다. 분류 손실에는 FF++ 정답만 사용하며, 모델에 CDF 정답·스무딩 여부·강도를 전달하지 않는다. 동일 CDF 이미지의 원본/스무딩 짝도 전달하지 않는다. 클래스 정보를 이용한 통제 실험으로 보고한다.
- 원본·약·중·강 선택 확률은 각각 25%. 실제 횟수는 무작위로 달라지고 epoch 기록에 저장된다. fallback은 그대로 입력하므로 실제 스무딩 비율은 이보다 작다.
- 기존 UDA 대비 타깃의 변형 혼합과 클래스 샘플링을 함께 변경한다. 기존 결과와의 차이만으로 두 변경의 개별 효과를 확정하지 않는다.
- FF++ 6,000장/epoch(Real3,000 + Fake4×750), 기존 clean FF++ validation1,200장. UDA는 타깃6,000회/epoch(Real3,000/Fake3,000).
- 최대10epoch, patience3, seed42, batch16, encoder/head LR1e-5. 기존 fine-tuning 범위·전처리·BN 통계 고정 정책 유지.
- 학습은 모든 방법에서 기존 FF++ baseline best.pt부터 독립적으로 시작한다. 검증 선택 기준은 기존 clean FF++ macro4 frame AUC를 유지한다. FPR·FNR도 기록한다. 타깃/테스트 정답으로 checkpoint를 선택하지 않는다.
- Aug는 source 두 view 평균CE. UDA는 source CE + 0.1 domain CE. Aug+UDA는 source 두 view 평균CE + 0.1 domain CE. source/target 도메인 손실은 각각 평균한 뒤 1/2씩 더해 source view 증가로 도메인 비중이 달라지지 않게 한다.
- Aug+UDA는 배치당 FF++ 두 view + CDF 한 view라 계산량이 더 크다. FF++ 고유 이미지 수, source epoch 목록, UDA 타깃 노출 수/목록은 일치한다. 계산량까지 동일한 비교라고 표현하지 않는다.
- 평가: 공식 CDF test8,288장 × 원본·스무딩3단계, 새 모델 총24조건. FPR·FNR·평균 오류율이 주요 지표이며 AUC는 보조 지표. threshold0.5, FP32, frame 지표만 사용한다. 기존 baseline·Aug 결과는 재사용한다.
- 기존 Aug와 새 학습의 source 이미지 예산은 같지만 배치 순서는 완전히 같지 않다. 기존 Aug는 참고 비교군으로 표시한다.
- 새 TAR 업로드는 필요 없다. 기존 결과는 보존하고 새 `ffpp_c23_mixed_balanced_uda_v1` 경로에 저장한다. 이 코드를 GitHub에 push한 뒤 실행한다.

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
assert (PROJECT/'scripts/ffpp_c23_mixed_balanced_uda.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

MODELS만 바꾸면 모델별로 코랩을 나눠 실행할 수 있다. 같은 run을 동시에 실행하지 않는다.
기존 UDA 목록 경로는 모델을 나누더라도 그대로 유지한다.

```python
MODELS=['xception','clip_b16','dinov2_b14']
STRATEGIES=['uda_mixed_balanced','uda_aug_mixed_balanced']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
UDA_MANIFEST_DIR=ROOT/'manifests/ffpp_c23_smoothing_adaptation_v1/seed42/xception_clip_b16'
RUNS_ROOT=ROOT/'runs/ffpp_c23_mixed_balanced_uda_v1'
GROUP='_'.join(MODELS)
MANIFEST_ROOT=ROOT/'manifests/ffpp_c23_mixed_balanced_uda_v1/seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations/ffpp_c23_mixed_balanced_uda_v1/seed42'/GROUP
BASELINE_EVALUATION=ROOT/'evaluations/ffpp_c23_baselines_celebdf_official_smoothing_v1/seed42'
REFERENCE_EVALUATIONS=ROOT/'evaluations/ffpp_c23_smoothing_adaptation_v1/seed42'
CACHE_ROOT=Path('/content/celebdf_official_smoothing_cache_v1')
MODEL_ARGS=' '.join(MODELS)
STRATEGY_ARGS=' '.join(STRATEGIES)
COMMON=f'--models {MODEL_ARGS} --strategies {STRATEGY_ARGS} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --uda-manifest-dir "{UDA_MANIFEST_DIR}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --output-root "{EVALUATION_ROOT}" --cache-root "{CACHE_ROOT}" --baseline-evaluation-root "{BASELINE_EVALUATION}" --batch-size 16 --workers 2'
for name in ['protocol.json','target_unlabelled.csv','source_validation.csv']:
    assert (UDA_MANIFEST_DIR/name).is_file(),f'기존 UDA 목록 경로 확인: {UDA_MANIFEST_DIR/name}'
print('Models:',MODELS,'Strategies:',STRATEGIES,flush=True)
```

## Cell 3 — FF++ Train/Val

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{FF_DATA}" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'
```

## Cell 4 — CDF·초기 모델

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
    marker=Path('/content')/f'.mixed_uda_{name}.done'
    available=any((Path('/content')/f'.{prefix}_{name}.done').exists() for prefix in ['adapt','sda','sda_balance','mixed_uda'])
    if name=='celebdf_v2_roi16_test_v1.tar':
        available=available or Path('/content/.celebdf_official_roi16_test_v1.done').exists()
    if available and manifest.is_file():
        print('Already extracted:',name,flush=True)
        continue
    assert source.stat().st_size==size,f'업로드 미완료 또는 다른 TAR: {source}'
    assert shutil.disk_usage('/content').free>2*size+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name
    print('Copying:',name,flush=True)
    shutil.copy2(source,local)
    assert local.stat().st_size==size,'로컬 복사 미완료'
    print('Extracting:',name,flush=True)
    result=subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        print(result.stderr)
        raise RuntimeError('CDF 압축 해제 실패')
    assert manifest.is_file(),manifest
    marker.touch()
    local.unlink()
for model in MODELS:
    name=f'{model}_ffpp_c23_roi16_baseline_v1_seed42'
    source=BASELINE_DRIVE_RUNS/name
    dest=BASELINE_RUNS/name
    dest.mkdir(parents=True,exist_ok=True)
    for filename in ['best.pt','config.json','completed.json']:
        assert (source/filename).is_file(),f'Baseline 완료 확인: {source/filename}'
        shutil.copy2(source/filename,dest/filename)
    print('Baseline ready:',model,flush=True)
```

## Cell 5 — UDA·Aug+UDA 학습

진행률은 15초 간격으로 갱신하며 epoch별 검증과 타깃 노출 수를 표시한다.
완료 run은 건너뛰고 중단 run은 last.pt의 마지막 완료 epoch부터 재개한다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_mixed_balanced_uda.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --encoder-lr 1e-5 --head-lr 1e-5 --domain-weight 0.1
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — CDF 원본·스무딩 3단계 평가

평가 진행률을 표시한다. 완료 결과는 재사용하고 baseline은 재추론하지 않는다.

```python
!python -u "{PROJECT}/scripts/ffpp_c23_mixed_balanced_uda.py" evaluate {COMMON} --batch-size 32
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 비교 표

```python
import csv
from IPython.display import Markdown,display
rows=list(csv.DictReader((EVALUATION_ROOT/'frame_summary.csv').open()))
current=json.loads((EVALUATION_ROOT/'evaluation_protocol.json').read_text())
for model in MODELS:
    candidates=sorted(REFERENCE_EVALUATIONS.rglob(f'{model}_augmentation_normal.json'))
    for candidate in candidates:
        folder=candidate.parent
        protocol=folder/'evaluation_protocol.json'
        if not protocol.is_file():
            continue
        previous=json.loads(protocol.read_text())
        if any(previous.get(k)!=current.get(k) for k in ['manifest_sha256','config','opencv','smoothing_code_sha256']):
            continue
        paths=[folder/f'{model}_augmentation_{c}.json' for c in ['normal','smooth_weak','smooth_medium','smooth_strong']]
        if not all(path.is_file() for path in paths):
            continue
        for condition,path in zip(['normal','smooth_weak','smooth_medium','smooth_strong'],paths):
            saved=json.loads(path.read_text())
            rows.append({'model':model,'strategy':'augmentation','condition':condition,
                         **{k:saved[k] for k in ['auc','accuracy','f1','real_fpr','fake_fnr']},
                         'balanced_error_rate':.5*(saved['real_fpr']+saved['fake_fnr'])})
        print('Reused Aug:',model,folder,flush=True)
        break
    else:
        print('기존 Aug 결과를 찾지 못해 새 결과만 표시:',model,flush=True)
fields=['model','strategy','condition','auc','accuracy','f1','real_fpr','fake_fnr','balanced_error_rate']
with (EVALUATION_ROOT/'with_existing_aug_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
labels={'baseline':'Baseline','augmentation':'기존 Aug','uda_mixed_balanced':'UDA','uda_aug_mixed_balanced':'Aug+UDA'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
for model in MODELS:
    table=[f'### {model}', '', '|방법|평가 조건|오탐 FPR|미탐 FNR|평균 오류율|AUC|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['model']==model:
            table.append(f"|{labels[r['strategy']]}|{conditions[r['condition']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
print('Saved:',EVALUATION_ROOT)
```
