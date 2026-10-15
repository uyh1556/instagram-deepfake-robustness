# FFPP_c23_Smoothing_Consistency_C_Seed42

세 모델의 **학습된 B → C 추가학습 → CDF 원본·스무딩 평가**. 새 코드를 GitHub에 push한 뒤 실행한다. MODELS만 바꾸면 모델별로 코랩을 나눌 수 있다.

- 시작점 자동 선택: B `best.pt`의 epoch가0이면 **B last.pt**, 아니면 **B best.pt**. 원본 파일은 그대로 보존하며 선택한 파일/epoch/SHA를 C config에 기록한다. 현재 완료 결과 기준 Xception best epoch2 / CLIP last epoch3 / DINO best epoch6이다.
- 기존 인코더·원래 분류기는 전부 고정. B에서 학습한 작은 보정 모듈만 계속 학습한다. B의 scale/bound도 그대로 가져온다. C는 새 AdamW(LR1e-4)로 시작하며 B optimizer를 재개하는 방식은 아니다.
- FF++ 정답 학습·스무딩 점수 복원·원본 판단 유지 손실은 B와 동일하게 유지한다. source6,000/epoch, Real3,000+Fake4×750, 원본/스무딩2view.
- CDF는 기존 고정 비테스트5,000장을 재사용하고6,000회/epoch 노출한다. **Real/Fake5:5는 기존 정답 보조 샘플링 통제 조건**이며 타깃 정답은 분류 손실에 사용하지 않는다. 순수하게 정답 없이 만든 타깃 표본이라고 주장하지 않는다.
- 타깃은 원본·약·중·강 중 하나를25%씩 선택한 observed image와, 거기에 약한 스무딩(strength0.25)을 추가한 이미지의2view. 모델/손실에 스무딩 여부/강도/타깃라벨을 전달하지 않는다. observed가 이미 스무딩일 수 있으며 깨끗한 원본짝을 받는 설정이 아니다. 두 번 적용한 스무딩은 원래4조건과 다른 변형일 수 있다. fallback은 그대로 둔다.
- C 손실 = B의 source 손실 + **0.1×target pair normalized Huber** + **1×observed target 점수를 고정 B-start 점수에 가깝게 하는 normalized Huber**. 보정 전후 두 target view에 gradient가 흐르고 B teacher는 고정한다. DANN/entropy minimization/pseudo-label/50:50 예측 강제는 사용하지 않는다. 이 일관성만으로 올바른 타깃 절대 점수가 식별되는 것은 아니다.
- 최대10epoch/patience3/seed42/batch16. frozen FP32 feature를 로컬 캐시한다. 계산량은 source2view+target2view로24,000views/epoch이며 B보다 크다. 검증 특징4조건은 한 번 계산하여 재사용한다. 학습·평가 진행률은10초 간격으로 갱신한다.
- 검증은 기존 동일 FF++1,200장×4조건. 평균BER최소, cleanBER+1%p/FPR·FNR각+2%p/AUC−0.005 guard는 **C 시작 B 모델 대비** 적용한다. **C epoch0은 이미 학습된 B 모델**이다. C에서 epoch0이 선택되더라도 보정값0 baseline을 뜻하지 않는다.
- CDF공식test8,288장×4조건, threshold0.5, FP32/frame FPR·FNR·BER·AUC. 같은 frozen backbone forward로 **실제 B-start와 C를 함께 평가**하므로 CLIP은 잘못된 B epoch0이 아닌 B epoch3과 비교한다. 기존Baseline/Aug는 저장결과를 읽는다.
- C와 B-start 차이에는 추가source학습 효과도 포함되므로, 타깃항목만의 기여라고 단정하지 않는다. 이미 반복 확인한 CDF test는 개발평가 한계를 유지한다. 기존 학습/평가/데이터는 보존한다.
- 새 TAR 업로드 없음. FF++ Train/Val5개와 기존CDF test/non-test Real/non-test Fake3개 TAR 및 기존 UDA manifest를 사용한다. 새 출력은 `ffpp_c23_smoothing_consistency_c_v1`이다.

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_consistency_c.py').is_file(),'새 코드를 먼저 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['xception','clip_b16','dinov2_b14']
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
B_DRIVE_RUNS=ROOT/'runs/ffpp_c23_smoothing_preservation_b_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
B_RUNS=Path('/content/preservation_b_initial_runs')
RUNS_ROOT=ROOT/'runs/ffpp_c23_smoothing_consistency_c_v1'
UDA_MANIFEST_DIR=ROOT/'manifests/ffpp_c23_smoothing_adaptation_v1/seed42/xception_clip_b16'
GROUP='_'.join(MODELS)
MANIFEST_ROOT=ROOT/'manifests/ffpp_c23_smoothing_consistency_c_v1/seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations/ffpp_c23_smoothing_consistency_c_v1/seed42'/GROUP
BASELINE_EVALUATION=ROOT/'evaluations/ffpp_c23_baselines_celebdf_official_smoothing_v1/seed42'
REFERENCE_EVALUATIONS=ROOT/'evaluations/ffpp_c23_smoothing_adaptation_v1/seed42'
FEATURE_CACHE=Path('/content/preservation_b_frozen_features_v1')
CACHE_ROOT=Path('/content/celebdf_official_smoothing_cache_v1')
MODEL_ARGS=' '.join(MODELS)
COMMON=f'--models {MODEL_ARGS} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --b-runs-root "{B_RUNS}" --runs-root "{RUNS_ROOT}" --uda-manifest-dir "{UDA_MANIFEST_DIR}" --manifest-root "{MANIFEST_ROOT}" --feature-cache-root "{FEATURE_CACHE}" --output-root "{EVALUATION_ROOT}" --cache-root "{CACHE_ROOT}" --baseline-evaluation-root "{BASELINE_EVALUATION}" --reference-evaluations "{REFERENCE_EVALUATIONS}" --workers 2'
for name in ['protocol.json','target_unlabelled.csv','source_validation.csv']:
    assert (UDA_MANIFEST_DIR/name).is_file(),UDA_MANIFEST_DIR/name
print('Models:',MODELS,'| B best epoch0 -> last.pt, otherwise best.pt',flush=True)
```

## Cell 3 — FF++ Train/Val

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{FF_DATA}" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'
```

## Cell 4 — CDF·기존 Baseline·B 모델 준비

```python
archives={'celebdf_v2_roi16_test_v1.tar':557015040,'celebdf_v2_roi16_non_test_real_v1.tar':826142720,'celebdf_v2_roi16_non_test_fake_v1.tar':5679564800}
for name,size in archives.items():
    manifest=TARGET_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv'
    markers=[Path('/content')/f'.{prefix}_{name}.done' for prefix in ['adapt','sda','sda_balance','mixed_uda','consistency_c']]
    if name=='celebdf_v2_roi16_test_v1.tar':
        markers += [Path('/content/.celebdf_official_roi16_test_v1.done'),Path('/content/.preservation_b_celebdf_official_test.done')]
    if manifest.is_file() and any(p.exists() for p in markers):
        print('Already extracted:',name,flush=True)
        continue
    source=CDF_DATA/name
    assert source.is_file() and source.stat().st_size==size,f'TAR 확인: {source}'
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
    (Path('/content')/f'.consistency_c_{name}.done').touch()
    local.unlink()
for model in MODELS:
    baseline_name=f'{model}_ffpp_c23_roi16_baseline_v1_seed42'
    b_name=f'{model}_ffpp_c23_smoothing_preservation_b_v1_seed42'
    b_source=B_DRIVE_RUNS/b_name
    if not (b_source/'completed.json').is_file():
        print('Finding B checkpoint:',model,flush=True)
        candidates=[p for p in (ROOT/'runs').rglob(b_name) if (p/'completed.json').is_file()]
        assert len(candidates)==1,f'B 경로를 B_DRIVE_RUNS에서 지정하세요: {candidates}'
        b_source=candidates[0]
    for source,dest,files in [(BASELINE_DRIVE_RUNS/baseline_name,BASELINE_RUNS/baseline_name,['best.pt','config.json','completed.json']),
                              (b_source,B_RUNS/b_name,['best.pt','last.pt','config.json','completed.json'])]:
        dest.mkdir(parents=True,exist_ok=True)
        for filename in files:
            assert (source/filename).is_file(),source/filename
            print('Copying model:',model,source.name,filename,flush=True)
            shutil.copy2(source/filename,dest/filename)
    done=json.loads((b_source/'completed.json').read_text())
    print(model,'B start:', 'last.pt' if done['selected_epoch']==0 else 'best.pt',
          'epoch',done['epochs_completed'] if done['selected_epoch']==0 else done['selected_epoch'],flush=True)
```

## Cell 5 — C 학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_consistency_c.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --learning-rate 1e-4 --target-weight 0.1 --trust-weight 1 --extra-strength 0.25
assert _exit_code==0,'C 학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — B-start vs C 평가

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_consistency_c.py" evaluate {COMMON} --batch-size 32
assert _exit_code==0,'C 평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표

```python
import csv
from IPython.display import Markdown,display
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison.csv').open()))
labels={'baseline':'Baseline','augmentation':'기존 Aug','b_start':'B 시작모델','preservation_c':'C'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
for model in MODELS:
    done=json.loads((RUNS_ROOT/f'{model}_ffpp_c23_smoothing_consistency_c_v1_seed42/completed.json').read_text())
    table=[f'### {model} — B epoch {done["b_start"]["epoch"]} → C 선택 epoch {done["selected_epoch"]}', '',
           '|방법|조건|FPR|FNR|BER|AUC|','|---|---|---:|---:|---:|---:|']
    for strategy in ['baseline','augmentation','b_start','preservation_c']:
        for r in rows:
            if r['model']==model and r['strategy']==strategy:
                table.append(f"|{labels[strategy]}|{conditions[r['condition']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
print('결과:',EVALUATION_ROOT)
print('4조건 평균:',EVALUATION_ROOT/'condition_means.csv')
```
