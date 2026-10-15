# FFPP_c23_Smoothing_Consistency_C_BER_Only_Seed42

세 모델의 **FF++ Baseline → 독립적인 C 학습 → CDF 원본·스무딩 평가**. 새 코드를 GitHub에 push한 뒤 실행한다. MODELS만 바꾸면 모델별로 코랩을 나눌 수 있다.

- B를 이어 학습하지 않는다. B와 동일한 FF++ Baseline에서 시작하여 출력이0인 보정 모듈을 새로 만든다. 인코더·원래 분류기는 고정하고 보정 모듈만 학습한다. scale/bound 초기화와 AdamW LR1e-3도 B와 맞춘다.
- FF++ 정답 학습·스무딩 점수 복원·원본 판단 유지 손실은 B와 동일하게 유지한다. source6,000/epoch, Real3,000+Fake4×750, 원본/스무딩2view.
- CDF는 기존 고정 비테스트5,000장을 재사용하고6,000회/epoch 노출한다. **Real/Fake5:5는 기존 정답 보조 샘플링 통제 조건**이며 타깃 정답은 분류 손실에 사용하지 않는다. 순수하게 정답 없이 만든 타깃 표본이라고 주장하지 않는다.
- 타깃은 원본·약·중·강 중 하나를25%씩 선택한 observed image와, 거기에 약한 스무딩(strength0.25)을 추가한 이미지의2view. 모델/손실에 스무딩 여부/강도/타깃라벨을 전달하지 않는다. observed가 이미 스무딩일 수 있으며 깨끗한 원본짝을 받는 설정이 아니다. 두 번 적용한 스무딩은 원래4조건과 다른 변형일 수 있다. fallback은 그대로 둔다.
- C 손실 = B의 source 손실 + **0.1×target pair normalized Huber**. 두 target view 모두에 gradient가 흐른다. B 점수 유지 손실은 제거한다. DANN/entropy minimization/pseudo-label/50:50 예측 강제는 사용하지 않는다. 일관성만으로 올바른 타깃 절대 점수가 식별되는 것은 아니다.
- 최대10epoch/patience3/seed42/batch16. frozen FP32 feature를 로컬 캐시한다. 계산량은 source2view+target2view로24,000views/epoch이며 B보다 크다. 검증 특징4조건은 한 번 계산하여 재사용한다. 학습·평가 진행률은10초 간격으로 갱신한다.
- 검증은 동일 FF++1,200장×4조건. **4조건 평균 BER만으로 best 선택·조기종료**한다. 원본 BER/FPR/FNR 및 AUC 제한은 모두 비활성화한다. epoch0도 후보이며, 0.0001 이상 개선이 3회 연속 없으면 종료한다. 원본 성능 변화와 AUC는 결과로 별도 보고한다.
- CDF공식test8,288장×4조건, threshold0.5, FP32/frame FPR·FNR·BER·AUC. Baseline/Aug/B는 기존 저장결과를 읽는다. B는 검증으로 선택된 결과만 비교하며 CLIP epoch3 진단 결과를 best로 대체하지 않는다.
- B와 초기화·source 목록·배치 순서·최대 epoch는 같지만, **이번 C는 선택 기준을 완화했으므로 기존 B와 공정한 동일 기준 비교가 아니다.** 실제 종료 시점/계산량도 다를 수 있다. 선택 기준 변경은 결과 확인 후의 탐색적 변경이며 CDF test는 개발평가로 취급한다.
- 새 TAR 업로드 없음. 기존 FF++/CDF TAR와 UDA manifest를 사용한다. 새 출력은 `ffpp_c23_smoothing_consistency_c_ber_only_v3`이다.

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
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
RUNS_ROOT=ROOT/'runs/ffpp_c23_smoothing_consistency_c_ber_only_v3'
UDA_MANIFEST_DIR=ROOT/'manifests/ffpp_c23_smoothing_adaptation_v1/seed42/xception_clip_b16'
GROUP='_'.join(MODELS)
MANIFEST_ROOT=ROOT/'manifests/ffpp_c23_smoothing_consistency_c_ber_only_v3/seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations/ffpp_c23_smoothing_consistency_c_ber_only_v3/seed42'/GROUP
B_EVALUATION=ROOT/'evaluations/ffpp_c23_smoothing_preservation_b_v1/seed42'
BASELINE_EVALUATION=ROOT/'evaluations/ffpp_c23_baselines_celebdf_official_smoothing_v1/seed42'
REFERENCE_EVALUATIONS=ROOT/'evaluations/ffpp_c23_smoothing_adaptation_v1/seed42'
FEATURE_CACHE=Path('/content/preservation_b_frozen_features_v1')
CACHE_ROOT=Path('/content/celebdf_official_smoothing_cache_v1')
MODEL_ARGS=' '.join(MODELS)
COMMON=f'--models {MODEL_ARGS} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --b-evaluation-root "{B_EVALUATION}" --runs-root "{RUNS_ROOT}" --uda-manifest-dir "{UDA_MANIFEST_DIR}" --manifest-root "{MANIFEST_ROOT}" --feature-cache-root "{FEATURE_CACHE}" --output-root "{EVALUATION_ROOT}" --cache-root "{CACHE_ROOT}" --baseline-evaluation-root "{BASELINE_EVALUATION}" --reference-evaluations "{REFERENCE_EVALUATIONS}" --selection-policy ber_only --workers 2'
for name in ['protocol.json','target_unlabelled.csv','source_validation.csv']:
    assert (UDA_MANIFEST_DIR/name).is_file(),UDA_MANIFEST_DIR/name
print('Models:',MODELS,'| C starts independently from FF++ Baseline',flush=True)
```

## Cell 3 — FF++ Train/Val

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{FF_DATA}" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'
```

## Cell 4 — CDF·기존 Baseline 준비

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
    for source,dest,files in [(BASELINE_DRIVE_RUNS/baseline_name,BASELINE_RUNS/baseline_name,['best.pt','config.json','completed.json'])]:
        dest.mkdir(parents=True,exist_ok=True)
        for filename in files:
            assert (source/filename).is_file(),source/filename
            print('Copying model:',model,source.name,filename,flush=True)
            shutil.copy2(source/filename,dest/filename)
```

## Cell 5 — C 학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_consistency_c.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --learning-rate 1e-3 --target-weight 0.1 --extra-strength 0.25
assert _exit_code==0,'C 학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — C 평가·기존 결과 비교

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_consistency_c.py" evaluate {COMMON} --batch-size 32
assert _exit_code==0,'C 평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표

```python
import csv
from IPython.display import Markdown,display
rows=list(csv.DictReader((EVALUATION_ROOT/'comparison.csv').open()))
labels={'baseline':'Baseline','augmentation':'기존 Aug','preservation_b':'B (검증 선택)','preservation_c':'C (BER 선택)'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
for model in MODELS:
    done=json.loads((RUNS_ROOT/f'{model}_ffpp_c23_smoothing_consistency_c_independent_v2_seed42/completed.json').read_text())
    table=[f'### {model} — BER-only C 선택 epoch {done["selected_epoch"]}', '',
           '|방법|조건|FPR|FNR|BER|AUC|','|---|---|---:|---:|---:|---:|']
    for strategy in dict.fromkeys(r['strategy'] for r in rows if r['model']==model):
        for r in rows:
            if r['model']==model and r['strategy']==strategy:
                table.append(f"|{labels.get(strategy,strategy+' (진단)')}|{conditions[r['condition']]}|{float(r['real_fpr']):.2%}|{float(r['fake_fnr']):.2%}|{float(r['balanced_error_rate']):.2%}|{float(r['auc']):.4f}|")
    display(Markdown('\n'.join(table)))
print('결과:',EVALUATION_ROOT)
print('4조건 평균:',EVALUATION_ROOT/'condition_means.csv')
```

